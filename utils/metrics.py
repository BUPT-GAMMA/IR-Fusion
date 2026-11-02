from typing import Dict, Iterable

import numpy as np


def _safe_corrcoef(target: np.ndarray, pred: np.ndarray) -> float:
    target = target.reshape(-1)
    pred = pred.reshape(-1)
    if np.std(target) == 0 or np.std(pred) == 0:
        return 0.0
    return float(np.corrcoef(target, pred)[0, 1])


def _binary_f1(target_hot: np.ndarray, pred_hot: np.ndarray) -> float:
    target_hot = np.asarray(target_hot, dtype=bool).reshape(-1)
    pred_hot = np.asarray(pred_hot, dtype=bool).reshape(-1)
    tp = float(np.logical_and(pred_hot, target_hot).sum())
    fp = float(np.logical_and(pred_hot, np.logical_not(target_hot)).sum())
    fn = float(np.logical_and(np.logical_not(pred_hot), target_hot).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def hotspot_f1(
    target: np.ndarray, pred: np.ndarray, hotspot_ratio: float = 0.9
) -> float:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target_threshold = float(np.max(target)) * hotspot_ratio
    pred_threshold = float(np.max(pred)) * hotspot_ratio
    return _binary_f1(target > target_threshold, pred > pred_threshold)


def evaluate_regression(
    target: np.ndarray,
    pred: np.ndarray,
    hotspot_ratio: float = 0.9,
) -> Dict[str, float]:
    target = np.asarray(target, dtype=np.float64).squeeze()
    pred = np.asarray(pred, dtype=np.float64).squeeze()
    error = pred - target
    mse = float(np.mean(error**2))
    hotspot_mask = target > hotspot_ratio * float(np.max(target))
    mirde = (
        float(np.mean(np.abs(error[hotspot_mask])))
        if np.any(hotspot_mask)
        else float(np.abs(error.reshape(-1)[int(np.argmax(target))]))
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "cc": _safe_corrcoef(target, pred),
        "f1": hotspot_f1(target, pred, hotspot_ratio),
        "mirde": mirde,
    }


def average_metrics(metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
    metrics = list(metrics)
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(np.mean([item[key] for item in metrics])) for key in keys}
