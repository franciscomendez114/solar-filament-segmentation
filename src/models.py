import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.GroupNorm(8, out_channels), nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.GroupNorm(8, out_channels), nn.SiLU(inplace=True),
        )

    def forward(self, inputs):
        return self.block(inputs)


class CropUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        c = base_channels
        self.encoder1, self.encoder2 = DoubleConv(in_channels, c), DoubleConv(c, c * 2)
        self.encoder3, self.encoder4 = DoubleConv(c * 2, c * 4), DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(DoubleConv(c * 8, c * 16), nn.Dropout2d(0.20))
        self.up4, self.decoder4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2), DoubleConv(c * 16, c * 8)
        self.up3, self.decoder3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2), DoubleConv(c * 8, c * 4)
        self.up2, self.decoder2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2), DoubleConv(c * 4, c * 2)
        self.up1, self.decoder1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2), DoubleConv(c * 2, c)
        self.output = nn.Conv2d(c, 1, 1)

    def forward(self, inputs):
        e1 = self.encoder1(inputs)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))
        x = self.decoder4(torch.cat((self.up4(x), e4), dim=1))
        x = self.decoder3(torch.cat((self.up3(x), e3), dim=1))
        x = self.decoder2(torch.cat((self.up2(x), e2), dim=1))
        return self.output(self.decoder1(torch.cat((self.up1(x), e1), dim=1)))


class PretrainedCropUNet(nn.Module):
    def __init__(self, encoder_name="efficientnet-b5", encoder_weights="imagenet"):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.unet = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=1, decoder_attention_type="scse")
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))

    def forward(self, inputs):
        return self.unet((inputs - self.mean) / self.std)


class RotationTTA(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, inputs):
        probabilities = []
        for rotation in range(4):
            rotated = torch.rot90(inputs, rotation, (-2, -1))
            probabilities.append(torch.rot90(torch.sigmoid(self.unet(rotated)), -rotation, (-2, -1)))
            probabilities.append(torch.rot90(torch.sigmoid(self.unet(rotated.flip(-1))).flip(-1), -rotation, (-2, -1)))
        mean = torch.stack(probabilities).mean(0).clamp(1e-6, 1 - 1e-6)
        return torch.log(mean / (1 - mean))


class ModelAverage(nn.Module):
    # Averages probabilities, not logits, and returns a logit so predict_instances can apply its usual sigmoid
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, inputs):
        mean = torch.stack([torch.sigmoid(model(inputs)) for model in self.models]).mean(0).clamp(1e-6, 1 - 1e-6)
        return torch.log(mean / (1 - mean))


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.30, beta=0.70, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits, targets):
        probabilities, targets = torch.sigmoid(logits.float()), targets.float()
        dimensions = (1, 2, 3)
        tp = (probabilities * targets).sum(dimensions)
        fp = (probabilities * (1 - targets)).sum(dimensions)
        fn = ((1 - probabilities) * targets).sum(dimensions)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return ((1 - tversky) ** self.gamma).mean()


def build_model(model_name, checkpoint_config=None, pretrained=True):
    checkpoint_config = checkpoint_config or {}
    if model_name == "crop_unet":
        return CropUNet(base_channels=checkpoint_config.get("base_channels", 32))
    if model_name == "b5":
        return PretrainedCropUNet(encoder_name=checkpoint_config.get("encoder_name", "efficientnet-b5"), encoder_weights="imagenet" if pretrained else None)
    raise ValueError(f"Unknown model: {model_name}")


def load_checkpoint(checkpoint_path, model_name, device):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(model_name, checkpoint.get("config", {}), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    if "official_pq" in checkpoint:
        score = f"official PQ {checkpoint['official_pq']:.4f}"
    elif "validation_dice" in checkpoint:
        score = f"oracle validation Dice {checkpoint['validation_dice']:.4f}"
    else:
        score = "saved score unavailable"
    print(f"Loaded {checkpoint_path.name}: epoch {checkpoint.get('epoch', 'unknown')}, {score}")
    return model, checkpoint


def combined_loss(logits, targets, settings):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    tversky = FocalTverskyLoss(settings["tversky_alpha"], settings["tversky_beta"], settings["tversky_gamma"])(logits, targets)
    return settings["bce_weight"] * bce + settings["tversky_weight"] * tversky, bce, tversky


def train_crop_unet(model, train_loader, validation_loader, device, settings, checkpoint_path, history_path, pq_validator=None):
    # pq_validator(model) -> official-PQ summary. When given, it selects the checkpoint instead of oracle crop Dice,
    # which is blind to false positives because the validation crops are all positives with oracle prompts.
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    updates_per_epoch = math.ceil(len(train_loader) / settings["accumulation_steps"])
    warmup_updates = updates_per_epoch * settings["warmup_epochs"]
    total_updates = updates_per_epoch * settings["num_epochs"]
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, total_iters=warmup_updates)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates - warmup_updates, eta_min=settings["min_learning_rate"])
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_updates])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, best_score, stale_epochs = [], -1.0, 0

    for epoch in range(settings["num_epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss, train_samples = 0.0, 0
        for batch_index, (inputs, targets, _) in enumerate(tqdm(train_loader, desc=f"Training {epoch + 1}")):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            group_start = (batch_index // settings["accumulation_steps"]) * settings["accumulation_steps"]
            group_size = min(settings["accumulation_steps"], len(train_loader) - group_start)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss, _, _ = combined_loss(model(inputs), targets, settings)
            scaler.scale(loss / group_size).backward()
            train_loss += float(loss.detach()) * len(inputs)
            train_samples += len(inputs)
            if (batch_index + 1) % settings["accumulation_steps"] == 0 or batch_index + 1 == len(train_loader):
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= old_scale:
                    scheduler.step()

        model.eval()
        dice_sum, count = 0.0, 0
        with torch.inference_mode():
            for inputs, targets, _ in tqdm(validation_loader, desc=f"Validation {epoch + 1}"):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True).bool()
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    predictions = torch.sigmoid(model(inputs).float()) >= 0.50
                intersection = (predictions & targets).sum((1, 2, 3)).float()
                dice_sum += ((2 * intersection + 1e-6) / (predictions.sum((1, 2, 3)) + targets.sum((1, 2, 3)) + 1e-6)).sum().item()
                count += len(inputs)
        validation_dice = dice_sum / count
        summary = pq_validator(model) if pq_validator else {}
        score = summary["official_pq"] if pq_validator else validation_dice
        history.append({"epoch": epoch + 1, "train_loss": train_loss / train_samples, "validation_dice": validation_dice,
                        **summary, "learning_rate": scheduler.get_last_lr()[0]})
        pd.DataFrame(history).to_csv(history_path, index=False)
        if settings.get("stop_after_epoch"):
            # Full-data training: validation is inside the training set, so nothing can select; keep the epoch the split run chose
            if epoch + 1 == settings["stop_after_epoch"]:
                torch.save({"model_state_dict": model.state_dict(), "config": settings, "epoch": epoch + 1, "validation_dice": validation_dice}, checkpoint_path)
                break
            continue
        if score > best_score:
            best_score, stale_epochs = score, 0
            torch.save({"model_state_dict": model.state_dict(), "config": settings, "epoch": epoch + 1,
                        "validation_dice": validation_dice, **summary}, checkpoint_path)
        else:
            stale_epochs += 1
        selection = f"official PQ {summary['official_pq']:.4f} (TP {summary['official_tp']}, FP {summary['official_fp']}, FN {summary['official_fn']})" if pq_validator else f"oracle validation Dice {validation_dice:.4f}"
        print(f"Epoch {epoch + 1}: train loss {train_loss / train_samples:.4f}, {selection}")
        if stale_epochs >= settings["early_stopping_patience"]:
            break
    return pd.DataFrame(history)

