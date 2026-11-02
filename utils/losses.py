import torch
import torch.nn as nn


class HotspotDiceL1(nn.Module):
    def __init__(
        self,
        hotspot_ratio: float = 0.9,
        dice_weight: float = 0.05,
        temperature: float = 0.02,
    ) -> None:
        super().__init__()
        self.hotspot_ratio = float(hotspot_ratio)
        self.dice_weight = float(dice_weight)
        self.temperature = float(temperature)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_loss = torch.mean(torch.abs(pred - target))
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)
        pred_peak = pred_flat.amax(dim=1, keepdim=True).clamp_min(1.0e-8)
        target_peak = target_flat.detach().amax(dim=1, keepdim=True).clamp_min(1.0e-8)
        pred_threshold = self.hotspot_ratio * pred_peak
        target_hot = (target_flat >= self.hotspot_ratio * target_peak).to(pred.dtype)
        soft_pred = torch.sigmoid(
            (pred_flat - pred_threshold) / (self.temperature * pred_peak)
        )
        intersection = (soft_pred * target_hot).sum(dim=1)
        dice = (2.0 * intersection + 1.0e-6) / (
            soft_pred.sum(dim=1) + target_hot.sum(dim=1) + 1.0e-6
        )
        return base_loss + self.dice_weight * (1.0 - dice).mean()


def build_loss(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "l1":
        return nn.L1Loss()
    return HotspotDiceL1(
        hotspot_ratio=kwargs.get("hotspot_ratio", 0.9),
        dice_weight=kwargs.get("dice_weight", 0.05),
        temperature=kwargs.get("dice_temperature", 0.02),
    )
