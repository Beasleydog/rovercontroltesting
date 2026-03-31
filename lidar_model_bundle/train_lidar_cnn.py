import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


SEED = 42
ROOT = Path("/content/cleancombineddata")
OUT_DIR = Path("/content/lidar_experiments")
MAX_LIDAR_CM = 5000.0
NUM_BEAMS = 17


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def beam_columns(columns: Sequence[str]) -> List[str]:
    return [c for c in columns if c.startswith("lidar_") and not c.endswith("_is_obstacle")]


def label_columns(columns: Sequence[str]) -> List[str]:
    return [c for c in columns if c.endswith("_is_obstacle")]


def normalize_heading_deg(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(values.astype(np.float32))
    return np.sin(radians), np.cos(radians)


def sanitize_lidar(distances_cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = (distances_cm > 0.0) & np.isfinite(distances_cm) & (distances_cm <= MAX_LIDAR_CM)
    clipped = np.where(valid, np.clip(distances_cm, 0.0, MAX_LIDAR_CM), 0.0).astype(np.float32)
    normalized = np.log1p(clipped) / np.log1p(MAX_LIDAR_CM)
    return normalized.astype(np.float32), valid.astype(np.float32)


@dataclass
class FileData:
    file_id: str
    parent_group: str
    lidar: np.ndarray
    valid: np.ndarray
    pose_xyz: np.ndarray
    orient: np.ndarray
    deltas: np.ndarray
    labels: np.ndarray
    steps: np.ndarray


def load_file_pair(raw_path: Path, label_path: Path) -> FileData:
    raw_df = pd.read_csv(raw_path)
    label_df = pd.read_csv(label_path)

    raw_df = raw_df.sort_values("step_idx").reset_index(drop=True)
    label_df = label_df.sort_values("step_idx").reset_index(drop=True)

    raw_beams = beam_columns(raw_df.columns)
    label_beams = label_columns(label_df.columns)
    expected_labels = [f"{c}_is_obstacle" for c in raw_beams]
    if label_beams != expected_labels:
        raise ValueError(f"Beam/label mismatch for {raw_path.name}")

    lidar_raw = raw_df[raw_beams].to_numpy(dtype=np.float32)
    lidar_norm, valid_mask = sanitize_lidar(lidar_raw)

    pose_xyz = raw_df[["rover_pos_x", "rover_pos_y", "rover_pos_z"]].to_numpy(dtype=np.float32)
    heading_sin, heading_cos = normalize_heading_deg(raw_df["heading"].to_numpy(dtype=np.float32))
    orient = np.stack(
        [
            heading_sin,
            heading_cos,
            raw_df["pitch"].to_numpy(dtype=np.float32) / 45.0,
            raw_df["roll"].to_numpy(dtype=np.float32) / 45.0,
        ],
        axis=1,
    ).astype(np.float32)

    deltas = np.zeros_like(pose_xyz, dtype=np.float32)
    deltas[1:] = pose_xyz[1:] - pose_xyz[:-1]
    deltas[:, 0] /= 1000.0
    deltas[:, 1] /= 1000.0
    deltas[:, 2] /= 100.0

    labels = label_df[label_beams].to_numpy(dtype=np.float32)
    steps = raw_df["step_idx"].to_numpy(dtype=np.int64)

    return FileData(
        file_id=raw_path.name,
        parent_group=str(raw_path.parent.relative_to(ROOT)),
        lidar=lidar_norm,
        valid=valid_mask,
        pose_xyz=pose_xyz,
        orient=orient,
        deltas=deltas,
        labels=labels,
        steps=steps,
    )


def discover_files(root: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for raw_path in sorted(root.glob("**/*.csv")):
        if "labeled_obstacles_liveexport" in str(raw_path) or raw_path.name == "manifest.json":
            continue
        label_path = raw_path.parent / "labeled_obstacles_liveexport" / raw_path.name
        if label_path.exists():
            pairs.append((raw_path, label_path))
    return pairs


def split_file_ids(files: List[FileData]) -> Dict[str, List[str]]:
    by_group: Dict[str, List[str]] = {}
    for file in files:
        by_group.setdefault(file.parent_group, []).append(file.file_id)

    split = {"train": [], "val": [], "test": []}
    rng = random.Random(SEED)
    for group, file_ids in by_group.items():
        ids = list(file_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(round(n * 0.7)))
        n_val = max(1, int(round(n * 0.15)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        split["train"].extend(ids[:n_train])
        split["val"].extend(ids[n_train:n_train + n_val])
        split["test"].extend(ids[n_train + n_val:])

    for key in split:
        split[key] = sorted(split[key])
    return split


class LidarSequenceDataset(Dataset):
    def __init__(
        self,
        files: Sequence[FileData],
        context_len: int,
        train_mode: bool,
    ) -> None:
        self.context_len = context_len
        self.train_mode = train_mode
        self.files = {f.file_id: f for f in files}
        self.samples: List[Tuple[str, int]] = []

        for f in files:
            for end_idx in range(context_len - 1, len(f.steps)):
                self.samples.append((f.file_id, end_idx))

        self.beam_index = np.linspace(-1.0, 1.0, NUM_BEAMS, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_id, end_idx = self.samples[idx]
        file = self.files[file_id]
        start_idx = end_idx - self.context_len + 1

        lidar = file.lidar[start_idx:end_idx + 1]
        valid = file.valid[start_idx:end_idx + 1]
        pose_xyz = file.pose_xyz[start_idx:end_idx + 1].copy()
        orient = file.orient[start_idx:end_idx + 1]
        deltas = file.deltas[start_idx:end_idx + 1]
        labels = file.labels[end_idx]

        if self.train_mode:
            pose_xyz[:, 0] += np.random.uniform(-10000.0, 10000.0)
            pose_xyz[:, 1] += np.random.uniform(-10000.0, 10000.0)
            pose_xyz[:, 2] += np.random.uniform(-50.0, 50.0)

        pose_scaled = pose_xyz.astype(np.float32).copy()
        pose_scaled[:, 0] /= 20000.0
        pose_scaled[:, 1] /= 20000.0
        pose_scaled[:, 2] /= 2000.0

        time_steps = lidar.shape[0]
        channels = []
        channels.append(lidar)
        channels.append(valid)

        for i in range(pose_scaled.shape[1]):
            channels.append(np.repeat(pose_scaled[:, i:i + 1], NUM_BEAMS, axis=1))
        for i in range(orient.shape[1]):
            channels.append(np.repeat(orient[:, i:i + 1], NUM_BEAMS, axis=1))
        for i in range(deltas.shape[1]):
            channels.append(np.repeat(deltas[:, i:i + 1], NUM_BEAMS, axis=1))
        channels.append(np.repeat(self.beam_index[None, :], time_steps, axis=0))

        x = np.stack(channels, axis=0).astype(np.float32)
        y = labels.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


class BeamTemporalCNN(nn.Module):
    def __init__(self, in_channels: int, widths: Sequence[int], kernel_t: int, kernel_b: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        c_in = in_channels
        for width in widths:
            blocks.extend(
                [
                    nn.Conv2d(
                        c_in,
                        width,
                        kernel_size=(kernel_t, kernel_b),
                        padding=(kernel_t // 2, kernel_b // 2),
                    ),
                    nn.BatchNorm2d(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            c_in = width
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv1d(c_in, c_in, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_in, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        current = features[:, :, -1, :]
        logits = self.head(current).squeeze(1)
        return logits


class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int, kernel_t: int, kernel_b: int, dropout: float, dilation_t: int = 1) -> None:
        super().__init__()
        padding = (dilation_t * (kernel_t // 2), kernel_b // 2)
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(kernel_t, kernel_b),
                padding=padding,
                dilation=(dilation_t, 1),
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(kernel_t, kernel_b),
                padding=padding,
                dilation=(dilation_t, 1),
            ),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResidualBeamCNN(nn.Module):
    def __init__(self, in_channels: int, width: int, depth: int, kernel_t: int, kernel_b: int, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=(kernel_t, kernel_b), padding=(kernel_t // 2, kernel_b // 2)),
            nn.BatchNorm2d(width),
            nn.GELU(),
        )
        blocks = []
        dilations = [1, 2, 4, 1, 2, 4]
        for i in range(depth):
            blocks.append(
                ResidualBlock2D(
                    width,
                    kernel_t=kernel_t,
                    kernel_b=kernel_b,
                    dropout=dropout,
                    dilation_t=dilations[i % len(dilations)],
                )
            )
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(self.stem(x))
        current = features[:, :, -1, :]
        return self.head(current).squeeze(1)


class FocalLoss(nn.Module):
    def __init__(self, pos_weight: float, gamma: float = 2.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        focal = (1.0 - pt).pow(self.gamma)
        alpha = torch.where(targets > 0.5, self.pos_weight, 1.0)
        loss = alpha * focal * bce
        return loss.mean()


@dataclass
class ExperimentConfig:
    name: str
    context_len: int
    widths: Tuple[int, ...]
    kernel_t: int
    kernel_b: int
    dropout: float
    model_type: str = "plain"
    loss_type: str = "bce"
    pos_weight_scale: float = 1.0
    focal_gamma: float = 2.0
    batch_size: int = 256
    epochs: int = 8
    patience: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4


def confusion_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(torch.int64)
    labels_i = labels.to(torch.int64)

    tp = int(((preds == 1) & (labels_i == 1)).sum().item())
    tn = int(((preds == 0) & (labels_i == 0)).sum().item())
    fp = int(((preds == 1) & (labels_i == 0)).sum().item())
    fn = int(((preds == 0) & (labels_i == 1)).sum().item())
    total = tp + tn + fp + fn

    obstacle_precision = tp / (tp + fp) if tp + fp else 0.0
    obstacle_recall = tp / (tp + fn) if tp + fn else 0.0
    obstacle_f1 = (
        2 * obstacle_precision * obstacle_recall / (obstacle_precision + obstacle_recall)
        if obstacle_precision + obstacle_recall
        else 0.0
    )

    non_precision = tn / (tn + fn) if tn + fn else 0.0
    non_recall = tn / (tn + fp) if tn + fp else 0.0
    non_f1 = (
        2 * non_precision * non_recall / (non_precision + non_recall)
        if non_precision + non_recall
        else 0.0
    )

    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "tp_obstacle": tp,
        "tn_non_obstacle": tn,
        "fp_obstacle": fp,
        "fn_obstacle": fn,
        "precision_obstacle": obstacle_precision,
        "recall_obstacle": obstacle_recall,
        "f1_obstacle": obstacle_f1,
        "precision_non_obstacle": non_precision,
        "recall_non_obstacle": non_recall,
        "f1_non_obstacle": non_f1,
        "positive_rate_pred": float(preds.float().mean().item()),
        "positive_rate_true": float(labels.float().mean().item()),
        "threshold": threshold,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    scaler: GradScaler | None,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n = 0
    all_logits = []
    all_labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)

        if train:
            if scaler is not None and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        n += batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = confusion_metrics(logits_cat, labels_cat)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics, logits_cat, labels_cat


def find_best_threshold(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics = confusion_metrics(logits, labels, threshold=0.5)
    best_score = best_metrics["f1_obstacle"]
    for threshold in np.linspace(0.2, 0.8, 25):
        metrics = confusion_metrics(logits, labels, threshold=float(threshold))
        if metrics["f1_obstacle"] > best_score:
            best_score = metrics["f1_obstacle"]
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def summarize_split(files: Sequence[FileData], split_ids: Sequence[str]) -> Dict[str, float]:
    total_steps = 0
    positive = 0
    total_labels = 0
    for f in files:
        if f.file_id not in split_ids:
            continue
        total_steps += len(f.steps)
        positive += int(f.labels.sum())
        total_labels += int(f.labels.size)
    return {
        "files": len(split_ids),
        "timesteps": total_steps,
        "positive_labels": positive,
        "positive_rate": positive / total_labels if total_labels else 0.0,
    }


def train_experiment(
    config: ExperimentConfig,
    files: Sequence[FileData],
    split_ids: Dict[str, List[str]],
    device: torch.device,
) -> Dict[str, object]:
    train_files = [f for f in files if f.file_id in split_ids["train"]]
    val_files = [f for f in files if f.file_id in split_ids["val"]]
    test_files = [f for f in files if f.file_id in split_ids["test"]]

    train_ds = LidarSequenceDataset(train_files, config.context_len, train_mode=True)
    val_ds = LidarSequenceDataset(val_files, config.context_len, train_mode=False)
    test_ds = LidarSequenceDataset(test_files, config.context_len, train_mode=False)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    positive_count = sum(int(f.labels.sum()) for f in train_files)
    total_count = sum(int(f.labels.size) for f in train_files)
    negative_count = total_count - positive_count
    pos_weight = (negative_count / max(positive_count, 1)) * config.pos_weight_scale

    sample_x, _ = train_ds[0]
    if config.model_type == "plain":
        model = BeamTemporalCNN(
            in_channels=sample_x.shape[0],
            widths=config.widths,
            kernel_t=config.kernel_t,
            kernel_b=config.kernel_b,
            dropout=config.dropout,
        ).to(device)
    elif config.model_type == "residual":
        model = ResidualBeamCNN(
            in_channels=sample_x.shape[0],
            width=config.widths[0],
            depth=len(config.widths),
            kernel_t=config.kernel_t,
            kernel_b=config.kernel_b,
            dropout=config.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    if config.loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device, dtype=torch.float32))
    elif config.loss_type == "focal":
        criterion = FocalLoss(pos_weight=pos_weight, gamma=config.focal_gamma)
    else:
        raise ValueError(f"Unknown loss_type: {config.loss_type}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    best_val = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        val_metrics, val_logits, val_labels = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False, scaler=None
        )
        tuned_threshold, tuned_val_metrics = find_best_threshold(val_logits, val_labels)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"[{config.name}] epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_f1_obs={train_metrics['f1_obstacle']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1_obs={val_metrics['f1_obstacle']:.4f} "
            f"val_f1_obs_tuned={tuned_val_metrics['f1_obstacle']:.4f} thr={tuned_threshold:.3f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if tuned_val_metrics["f1_obstacle"] > best_val:
            best_val = tuned_val_metrics["f1_obstacle"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_threshold = tuned_threshold
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("No best model state captured.")

    model.load_state_dict(best_state)
    val_best_default, val_logits, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False, scaler=None)
    test_default, test_logits, test_labels = run_epoch(model, test_loader, criterion, optimizer, device, train=False, scaler=None)
    val_best_tuned = confusion_metrics(val_logits, val_labels, threshold=best_threshold)
    val_best_tuned["loss"] = val_best_default["loss"]
    test_metrics = confusion_metrics(test_logits, test_labels, threshold=best_threshold)
    test_metrics["loss"] = test_default["loss"]

    return {
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "pos_weight": pos_weight,
        "best_val_metrics_default_threshold": val_best_default,
        "best_val_metrics_tuned_threshold": val_best_tuned,
        "test_metrics": test_metrics,
        "history": history,
    }


def build_report(dataset_summary: Dict[str, object], results: List[Dict[str, object]]) -> str:
    lines = []
    lines.append("# Lidar CNN Experiment Report")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- Total files: {dataset_summary['total_files']}")
    lines.append(f"- Total timesteps: {dataset_summary['total_timesteps']}")
    lines.append(f"- Beams per timestep: {dataset_summary['num_beams']}")
    lines.append(f"- Overall positive rate: {dataset_summary['overall_positive_rate']:.4f}")
    lines.append(f"- Split summary: {json.dumps(dataset_summary['splits'], indent=2)}")
    lines.append("")
    lines.append("## Experiments")
    for result in results:
        cfg = result["config"]
        test = result["test_metrics"]
        lines.append(
            f"- `{cfg['name']}`: context={cfg['context_len']}, widths={cfg['widths']}, "
            f"k_t={cfg['kernel_t']}, k_b={cfg['kernel_b']}, dropout={cfg['dropout']}, "
            f"model={cfg['model_type']}, loss={cfg['loss_type']}, pos_w_scale={cfg['pos_weight_scale']}, "
            f"best_epoch={result['best_epoch']}, thr={result['best_threshold']:.3f}"
        )
        lines.append(
            f"  test_acc={test['accuracy']:.4f}, obstacle_f1={test['f1_obstacle']:.4f}, "
            f"obstacle_precision={test['precision_obstacle']:.4f}, obstacle_recall={test['recall_obstacle']:.4f}, "
            f"fp={test['fp_obstacle']}, fn={test['fn_obstacle']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = discover_files(ROOT)
    files = [load_file_pair(raw, label) for raw, label in pairs]
    split_ids = split_file_ids(files)

    total_steps = sum(len(f.steps) for f in files)
    total_positive = sum(int(f.labels.sum()) for f in files)
    total_labels = sum(int(f.labels.size) for f in files)
    dataset_summary = {
        "total_files": len(files),
        "total_timesteps": total_steps,
        "num_beams": NUM_BEAMS,
        "overall_positive_rate": total_positive / total_labels,
        "splits": {
            split_name: summarize_split(files, ids) for split_name, ids in split_ids.items()
        },
    }

    experiments = [
        ExperimentConfig(name="ctx8_plain_bce_pw1", context_len=8, widths=(32, 64), kernel_t=3, kernel_b=3, dropout=0.10),
        ExperimentConfig(
            name="ctx8_plain_bce_pw05",
            context_len=8,
            widths=(32, 64),
            kernel_t=3,
            kernel_b=3,
            dropout=0.10,
            pos_weight_scale=0.5,
        ),
        ExperimentConfig(
            name="ctx8_plain_bce_pw2",
            context_len=8,
            widths=(32, 64),
            kernel_t=3,
            kernel_b=3,
            dropout=0.10,
            pos_weight_scale=2.0,
        ),
        ExperimentConfig(
            name="ctx8_residual_bce",
            context_len=8,
            widths=(64, 64, 64, 64),
            kernel_t=3,
            kernel_b=3,
            dropout=0.10,
            model_type="residual",
            epochs=10,
            patience=3,
        ),
        ExperimentConfig(
            name="ctx8_residual_focal",
            context_len=8,
            widths=(64, 64, 64, 64),
            kernel_t=3,
            kernel_b=3,
            dropout=0.10,
            model_type="residual",
            loss_type="focal",
            focal_gamma=2.0,
            epochs=10,
            patience=3,
        ),
        ExperimentConfig(
            name="ctx8_residual_focal_pw05",
            context_len=8,
            widths=(64, 64, 64, 64),
            kernel_t=3,
            kernel_b=3,
            dropout=0.10,
            model_type="residual",
            loss_type="focal",
            focal_gamma=2.0,
            pos_weight_scale=0.5,
            epochs=10,
            patience=3,
        ),
        ExperimentConfig(
            name="ctx8_residual_focal_pw2",
            context_len=8,
            widths=(64, 64, 64, 64),
            kernel_t=3,
            kernel_b=3,
            dropout=0.10,
            model_type="residual",
            loss_type="focal",
            focal_gamma=2.0,
            pos_weight_scale=2.0,
            epochs=10,
            patience=3,
        ),
        ExperimentConfig(
            name="ctx8_residual_bce_k5",
            context_len=8,
            widths=(64, 64, 64, 64),
            kernel_t=5,
            kernel_b=5,
            dropout=0.15,
            model_type="residual",
            loss_type="bce",
            epochs=10,
            patience=3,
        ),
        ExperimentConfig(
            name="ctx12_residual_bce",
            context_len=12,
            widths=(64, 64, 64, 64),
            kernel_t=3,
            kernel_b=3,
            dropout=0.10,
            model_type="residual",
            loss_type="bce",
            epochs=10,
            patience=3,
        ),
        ExperimentConfig(
            name="ctx12_residual_focal",
            context_len=12,
            widths=(64, 64, 64, 64),
            kernel_t=3,
            kernel_b=3,
            dropout=0.10,
            model_type="residual",
            loss_type="focal",
            focal_gamma=2.0,
            epochs=10,
            patience=3,
        ),
        ExperimentConfig(
            name="ctx10_residual_focal",
            context_len=10,
            widths=(64, 64, 64, 64),
            kernel_t=3,
            kernel_b=3,
            dropout=0.10,
            model_type="residual",
            loss_type="focal",
            focal_gamma=2.0,
            epochs=10,
            patience=3,
        ),
        ExperimentConfig(
            name="ctx12_residual_focal_g15",
            context_len=12,
            widths=(64, 64, 64, 64),
            kernel_t=3,
            kernel_b=3,
            dropout=0.10,
            model_type="residual",
            loss_type="focal",
            focal_gamma=1.5,
            epochs=10,
            patience=3,
        ),
        ExperimentConfig(
            name="ctx12_residual_focal_g25",
            context_len=12,
            widths=(64, 64, 64, 64),
            kernel_t=3,
            kernel_b=3,
            dropout=0.10,
            model_type="residual",
            loss_type="focal",
            focal_gamma=2.5,
            epochs=10,
            patience=3,
        ),
        ExperimentConfig(
            name="ctx12_residual_focal_pw15",
            context_len=12,
            widths=(64, 64, 64, 64),
            kernel_t=3,
            kernel_b=3,
            dropout=0.10,
            model_type="residual",
            loss_type="focal",
            focal_gamma=2.0,
            pos_weight_scale=1.5,
            epochs=10,
            patience=3,
        ),
        ExperimentConfig(
            name="ctx16_residual_focal",
            context_len=16,
            widths=(64, 64, 64, 64),
            kernel_t=3,
            kernel_b=3,
            dropout=0.15,
            model_type="residual",
            loss_type="focal",
            focal_gamma=2.0,
            epochs=10,
            patience=3,
        ),
        ExperimentConfig(
            name="ctx4_residual_focal_pw15",
            context_len=4,
            widths=(64, 64, 64, 64),
            kernel_t=5,
            kernel_b=3,
            dropout=0.10,
            model_type="residual",
            loss_type="focal",
            focal_gamma=1.5,
            pos_weight_scale=1.5,
            epochs=10,
            patience=3,
        ),
    ]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this experiment. No Colab GPU was detected.")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(json.dumps(dataset_summary, indent=2))

    results = []
    for config in experiments:
        result = train_experiment(config, files, split_ids, device)
        results.append(result)

        result_path = OUT_DIR / f"{config.name}.json"
        result_path.write_text(json.dumps(result, indent=2))
        interim_report = build_report(dataset_summary, results)
        (OUT_DIR / "report.md").write_text(interim_report)

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps({"dataset": dataset_summary, "results": results}, indent=2))
    (OUT_DIR / "report.md").write_text(build_report(dataset_summary, results))
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
