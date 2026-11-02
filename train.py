import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from torch.amp import GradScaler, autocast

    AMP_HAS_DEVICE_TYPE = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    AMP_HAS_DEVICE_TYPE = False

from datasets import IRFusionMapDataset, infer_channels
from models import IRFusionNet
from utils.logger import build_logger, timestamped_dir
from utils.losses import build_loss
from utils.metrics import average_metrics, evaluate_regression


def load_json(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    with open(path, "r") as handle:
        return json.load(handle)


def dump_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(args: argparse.Namespace) -> torch.device:
    if args.cpu:
        return torch.device("cpu")
    if args.gpu is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_scaler(enabled: bool) -> GradScaler:
    if AMP_HAS_DEVICE_TYPE:
        return GradScaler("cuda", enabled=enabled)
    return GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    if AMP_HAS_DEVICE_TYPE:
        return autocast("cuda", enabled=enabled)
    return autocast(enabled=enabled)


def collate_batch(batch):
    features = torch.stack([item["feature"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    return {
        "feature": features,
        "label": labels,
        "name": [item["name"] for item in batch],
        "is_real": [item["is_real"] for item in batch],
        "difficulty": torch.tensor([item["difficulty"] for item in batch], dtype=torch.float32),
    }


def make_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    indices: Optional[list] = None,
) -> DataLoader:
    if indices is not None:
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
        drop_last=False,
    )


def build_model(in_channels: int, out_channels: int) -> IRFusionNet:
    return IRFusionNet(in_channels=in_channels, out_channels=out_channels)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    clip_grad_norm: float,
    target_scale: float,
    freeze_batchnorm: bool,
) -> float:
    model.train()
    if freeze_batchnorm:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    running_loss = 0.0
    seen = 0
    pbar = tqdm(loader, desc="train", ncols=100, leave=False)
    for batch in pbar:
        feature = batch["feature"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(enabled=use_amp):
            pred = model(feature)
            loss = criterion(pred, label * target_scale)

        scaler.scale(loss).backward()
        if clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(feature.shape[0])
        running_loss += float(loss.item()) * batch_size
        seen += batch_size
        pbar.set_postfix(loss=running_loss / max(seen, 1))
    return running_loss / max(seen, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    target_scale: float,
    hotspot_ratio: float,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    seen = 0
    metrics = []
    for batch in tqdm(loader, desc="eval", ncols=100, leave=False):
        feature = batch["feature"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        pred = model(feature)
        loss = criterion(pred, label * target_scale)
        running_loss += float(loss.item()) * int(feature.shape[0])
        seen += int(feature.shape[0])
        pred_np = (pred / target_scale).detach().cpu().numpy()
        label_np = label.detach().cpu().numpy()
        for idx in range(pred_np.shape[0]):
            sample_metrics = evaluate_regression(
                    label_np[idx],
                    pred_np[idx],
                    hotspot_ratio=hotspot_ratio,
                )
            metrics.append(sample_metrics)
    return running_loss / max(seen, 1), average_metrics(metrics)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_mae: float,
    best_f1: float,
    best_mirde: float,
    args: argparse.Namespace,
    in_channels: int,
    out_channels: int,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_mae": best_mae,
        "best_f1": best_f1,
        "best_mirde": best_mirde,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": vars(args),
        "model_kwargs": {
            "in_channels": in_channels,
            "out_channels": out_channels,
        },
        "target_scale": args.target_scale,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_model_weights(checkpoint_path: str, model: torch.nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get(
        "model", checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    )
    model.load_state_dict(state_dict, strict=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train IR-Fusion Inception Attention U-Net.")
    parser.add_argument("--config", default=None, help="JSON config file. CLI values override config values.")
    parser.add_argument("--train-csv", default="processed/splits/train.csv")
    parser.add_argument("--val-csv", default="processed/splits/val.csv")
    parser.add_argument("--dataroot", default="processed")
    parser.add_argument("--save-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-dir", default=None, help="Exact output directory. If set, --save-dir/--run-name timestamping is not used.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--loss",
        default="l1",
        choices=["l1", "hotspot_dice_l1"],
    )
    parser.add_argument("--dice-weight", type=float, default=0.05)
    parser.add_argument("--dice-temperature", type=float, default=0.02)
    parser.add_argument("--hotspot-ratio", type=float, default=0.9)
    parser.add_argument("--target-scale", type=float, default=1.0, help="Scale labels during loss computation; predictions are divided by this value for metrics.")
    parser.add_argument("--augment-rotations", action="store_true")
    parser.add_argument("--oversample-fake", type=int, default=2)
    parser.add_argument("--oversample-real", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--clip-grad-norm", type=float, default=0.0)
    parser.add_argument("--freeze-batchnorm", action="store_true")
    parser.add_argument("--init-checkpoint", default=None, help="Load model weights only and start a fresh optimizer/schedule.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    base_defaults = vars(parser.parse_args([]))
    initial, _ = parser.parse_known_args()
    config = load_json(initial.config)
    parser.set_defaults(**{key: value for key, value in config.items() if key in base_defaults})
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = choose_device(args)
    run_dir = Path(args.run_dir) if args.run_dir else Path(timestamped_dir(args.save_dir, args.run_name or "irfusion"))
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(str(run_dir), "train.log")
    dump_json(run_dir / "config.json", vars(args))

    train_dataset = IRFusionMapDataset(
        args.train_csv,
        dataroot=args.dataroot,
        require_label=True,
        augment_rotations=args.augment_rotations,
        oversample_fake=args.oversample_fake,
        oversample_real=args.oversample_real,
    )
    val_dataset = IRFusionMapDataset(args.val_csv, dataroot=args.dataroot, require_label=True)
    in_channels, out_channels = infer_channels(train_dataset)

    model = build_model(in_channels, out_channels).to(device)
    criterion = build_loss(
        args.loss,
        dice_weight=args.dice_weight,
        dice_temperature=args.dice_temperature,
        hotspot_ratio=args.hotspot_ratio,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    start_epoch = 0
    best_mae = float("inf")
    best_f1 = float("-inf")
    best_mirde = float("inf")
    if args.init_checkpoint:
        load_model_weights(args.init_checkpoint, model, device)

    scaler = make_scaler(enabled=args.amp and device.type == "cuda")
    logger.info("device=%s train_items=%d val_items=%d in_channels=%d", device, len(train_dataset), len(val_dataset), in_channels)

    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, shuffle=False)
    history = []
    if args.init_checkpoint:
        init_loss, init_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.target_scale,
            args.hotspot_ratio,
        )
        best_mae = float(init_metrics["mae"])
        best_f1 = float(init_metrics["f1"])
        best_mirde = float(init_metrics["mirde"])
        init_record = {
            "epoch": -1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": None,
            "val_loss": init_loss,
            **{f"val_{key}": value for key, value in init_metrics.items()},
        }
        history.append(init_record)
        save_checkpoint(
            run_dir / "best_mae.pth",
            model,
            optimizer,
            scheduler,
            -1,
            best_mae,
            best_f1,
            best_mirde,
            args,
            in_channels,
            out_channels,
        )
        logger.info(
            "initial val_loss=%.6g val_mae=%.6g val_f1=%.4f val_mirde=%.6g",
            init_loss,
            best_mae,
            best_f1,
            best_mirde,
        )
    for epoch in range(start_epoch, args.epochs):
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp=args.amp and device.type == "cuda",
            clip_grad_norm=args.clip_grad_norm,
            target_scale=args.target_scale,
            freeze_batchnorm=args.freeze_batchnorm,
        )
        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.target_scale,
            args.hotspot_ratio,
        )
        if scheduler is not None:
            scheduler.step()

        record = {"epoch": epoch, "lr": epoch_lr, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        dump_json(run_dir / "history.json", {"history": history})

        val_mae = float(val_metrics.get("mae", val_loss))
        val_f1 = float(val_metrics.get("f1", float("-inf")))
        val_mirde = float(val_metrics.get("mirde", float("inf")))
        is_best_mae = val_mae < best_mae
        is_best_f1 = val_f1 > best_f1
        is_best_mirde = val_mirde < best_mirde
        if is_best_mae:
            best_mae = val_mae
        if is_best_f1:
            best_f1 = val_f1
        if is_best_mirde:
            best_mirde = val_mirde
        if is_best_mae:
            save_checkpoint(
                run_dir / "best_mae.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_mae,
                best_f1,
                best_mirde,
                args,
                in_channels,
                out_channels,
            )
        save_checkpoint(
            run_dir / "last.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_mae,
            best_f1,
            best_mirde,
            args,
            in_channels,
            out_channels,
        )

        logger.info(
            "epoch=%03d lr=%.6g train_loss=%.6g val_loss=%.6g val_mae=%.6g val_f1=%.4f val_mirde=%.6g val_cc=%.4f best_mae=%s best_f1=%s best_mirde=%s",
            epoch,
            epoch_lr,
            train_loss,
            val_loss,
            val_metrics.get("mae", float("nan")),
            val_metrics.get("f1", float("nan")),
            val_metrics.get("mirde", float("nan")),
            val_metrics.get("cc", float("nan")),
            "yes" if is_best_mae else "no",
            "yes" if is_best_f1 else "no",
            "yes" if is_best_mirde else "no",
        )

    logger.info(
        "finished run_dir=%s best_mae=%.6g best_f1=%.4f best_mirde=%.6g",
        run_dir,
        best_mae,
        best_f1,
        best_mirde,
    )


if __name__ == "__main__":
    main()
