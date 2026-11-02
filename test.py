import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import IRFusionMapDataset, infer_channels
from models import IRFusionNet
from utils.metrics import average_metrics, evaluate_regression


def load_json(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    with open(path, "r") as handle:
        return json.load(handle)


def collate_batch(batch):
    features = torch.stack([item["feature"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    return {
        "feature": features,
        "label": labels,
        "name": [item["name"] for item in batch],
        "feature_path": [item["feature_path"] for item in batch],
        "label_path": [item["label_path"] for item in batch],
    }


def choose_device(args: argparse.Namespace) -> torch.device:
    if args.cpu:
        return torch.device("cpu")
    if args.gpu is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str, device: torch.device) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


def build_model_from_checkpoint(
    checkpoint: Dict[str, object],
    in_channels: int,
    out_channels: int,
    device: torch.device,
) -> torch.nn.Module:
    model = IRFusionNet(in_channels=in_channels, out_channels=out_channels).to(device)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint.get("model_state_dict")))
    if state_dict is None:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    hotspot_ratio: float,
    target_scale: float,
    save_predictions: bool,
    output_dir: Path,
) -> Dict[str, float]:
    model.eval()
    metrics = []
    per_sample_metrics = []
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "predictions"
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.perf_counter()
    inference_total = 0.0
    for batch in tqdm(loader, desc="test", ncols=100):
        feature = batch["feature"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_start = time.perf_counter()
        pred = model(feature)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_total += time.perf_counter() - inference_start
        pred_np = (pred / target_scale).cpu().numpy()
        label_np = label.cpu().numpy()
        for idx, name in enumerate(batch["name"]):
            sample_metrics = evaluate_regression(
                label_np[idx],
                pred_np[idx],
                hotspot_ratio=hotspot_ratio,
            )
            metrics.append(sample_metrics)
            per_sample_metrics.append({"name": name, **sample_metrics})
            if save_predictions:
                np.save(pred_dir / f"{name}.npy", pred_np[idx].astype(np.float32))

    results = average_metrics(metrics)
    results["num_samples"] = len(metrics)
    results["runtime_total_s"] = float(inference_total)
    results["runtime_per_sample_s"] = results["runtime_total_s"] / max(len(metrics), 1)
    results["evaluation_wall_total_s"] = float(time.perf_counter() - wall_start)
    results["evaluation_wall_per_sample_s"] = results["evaluation_wall_total_s"] / max(len(metrics), 1)
    with open(output_dir / "per_sample_metrics.json", "w") as handle:
        json.dump(per_sample_metrics, handle, indent=2)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained IR-Fusion model.")
    parser.add_argument("--config", default=None, help="JSON config file. CLI values override config values.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--test-csv", default="processed/splits/test.csv")
    parser.add_argument("--dataroot", default="processed")
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hotspot-ratio", type=float, default=0.9)
    parser.add_argument("--target-scale", type=float, default=None, help="Override target scale saved in checkpoint.")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    base_defaults = vars(parser.parse_args([]))
    initial, _ = parser.parse_known_args()
    config = load_json(initial.config)
    parser.set_defaults(**{key: value for key, value in config.items() if key in base_defaults})
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint is required, either on CLI or in --config")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    device = choose_device(args)

    dataset = IRFusionMapDataset(args.test_csv, dataroot=args.dataroot, require_label=True)
    in_channels, out_channels = infer_channels(dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model_from_checkpoint(checkpoint, in_channels, out_channels, device)
    checkpoint_scale = checkpoint.get("target_scale")
    if checkpoint_scale is None:
        checkpoint_scale = checkpoint.get("config", {}).get("target_scale", 1.0)
    target_scale = float(args.target_scale if args.target_scale is not None else checkpoint_scale)
    metrics = run_eval(
        model,
        loader,
        device,
        args.hotspot_ratio,
        target_scale,
        args.save_predictions,
        output_dir,
    )
    metrics["target_scale"] = target_scale
    metrics["hotspot_ratio"] = args.hotspot_ratio
    metrics["f1_definition"] = "per-map 90%-of-maximum thresholds"
    metrics["mirde_definition"] = "mean absolute error in the ground-truth 90%-of-maximum region"
    metrics["runtime_scope"] = "neural_network_forward_only"

    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
