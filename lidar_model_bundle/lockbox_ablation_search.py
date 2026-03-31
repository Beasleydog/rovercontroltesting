import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from train_lidar_cnn import (
    ROOT,
    OUT_DIR,
    BeamTemporalCNN,
    FileData,
    FocalLoss,
    ResidualBeamCNN,
    confusion_metrics,
    discover_files,
    find_best_threshold,
    load_file_pair,
    set_seed,
)


LOCKBOX_DIR = OUT_DIR / "lockbox_search"


@dataclass
class CandidateConfig:
    name: str
    context_len: int
    model_type: str
    widths: Tuple[int, ...]
    kernel_t: int
    kernel_b: int
    dropout: float
    loss_type: str
    focal_gamma: float
    pos_weight_scale: float
    lidar_noise_std: float = 0.0
    beam_dropout: float = 0.0
    timestep_dropout: float = 0.0
    epochs: int = 8
    patience: int = 2
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4


class AugmentedLidarDataset(Dataset):
    def __init__(self, files: Sequence[FileData], context_len: int, train_mode: bool, cfg: CandidateConfig) -> None:
        self.files = {f.file_id: f for f in files}
        self.samples: List[Tuple[str, int]] = []
        self.context_len = context_len
        self.train_mode = train_mode
        self.cfg = cfg
        self.beam_index = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
        for f in files:
            for end_idx in range(context_len - 1, len(f.steps)):
                self.samples.append((f.file_id, end_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_id, end_idx = self.samples[idx]
        file = self.files[file_id]
        start_idx = end_idx - self.context_len + 1

        lidar = file.lidar[start_idx:end_idx + 1].copy()
        valid = file.valid[start_idx:end_idx + 1].copy()
        pose_xyz = file.pose_xyz[start_idx:end_idx + 1].copy()
        orient = file.orient[start_idx:end_idx + 1]
        deltas = file.deltas[start_idx:end_idx + 1]
        labels = file.labels[end_idx]

        if self.train_mode:
            pose_xyz[:, 0] += np.random.uniform(-10000.0, 10000.0)
            pose_xyz[:, 1] += np.random.uniform(-10000.0, 10000.0)
            pose_xyz[:, 2] += np.random.uniform(-50.0, 50.0)

            if self.cfg.lidar_noise_std > 0:
                noise = np.random.normal(0.0, self.cfg.lidar_noise_std, size=lidar.shape).astype(np.float32)
                lidar = np.clip(lidar + noise * valid, 0.0, 1.0)
            if self.cfg.beam_dropout > 0:
                mask = (np.random.rand(*lidar.shape) >= self.cfg.beam_dropout).astype(np.float32)
                lidar *= mask
                valid *= mask
            if self.cfg.timestep_dropout > 0 and lidar.shape[0] > 1:
                step_mask = (np.random.rand(lidar.shape[0], 1) >= self.cfg.timestep_dropout).astype(np.float32)
                step_mask[-1, :] = 1.0
                lidar *= step_mask
                valid *= step_mask

        pose_scaled = pose_xyz.astype(np.float32).copy()
        pose_scaled[:, 0] /= 20000.0
        pose_scaled[:, 1] /= 20000.0
        pose_scaled[:, 2] /= 2000.0

        channels = [lidar, valid]
        for i in range(pose_scaled.shape[1]):
            channels.append(np.repeat(pose_scaled[:, i:i + 1], 17, axis=1))
        for i in range(orient.shape[1]):
            channels.append(np.repeat(orient[:, i:i + 1], 17, axis=1))
        for i in range(deltas.shape[1]):
            channels.append(np.repeat(deltas[:, i:i + 1], 17, axis=1))
        channels.append(np.repeat(self.beam_index[None, :], lidar.shape[0], axis=0))
        x = np.stack(channels, axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(labels.astype(np.float32))


def build_model(cfg: CandidateConfig, in_channels: int, device: torch.device) -> nn.Module:
    if cfg.model_type == "plain":
        model = BeamTemporalCNN(
            in_channels=in_channels,
            widths=cfg.widths,
            kernel_t=cfg.kernel_t,
            kernel_b=cfg.kernel_b,
            dropout=cfg.dropout,
        )
    elif cfg.model_type == "residual":
        model = ResidualBeamCNN(
            in_channels=in_channels,
            width=cfg.widths[0],
            depth=len(cfg.widths),
            kernel_t=cfg.kernel_t,
            kernel_b=cfg.kernel_b,
            dropout=cfg.dropout,
        )
    else:
        raise ValueError(cfg.model_type)
    return model.to(device)


def build_criterion(cfg: CandidateConfig, pos_weight: float, device: torch.device) -> nn.Module:
    if cfg.loss_type == "focal":
        return FocalLoss(pos_weight=pos_weight, gamma=cfg.focal_gamma)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device, dtype=torch.float32))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    scaler: GradScaler | None,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    model.train(train)
    total_loss = 0.0
    n = 0
    all_logits = []
    all_labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = confusion_metrics(logits, labels, threshold=0.5)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics, logits, labels


def file_positive_rates(files: Sequence[FileData]) -> np.ndarray:
    rates = []
    for f in files:
        rates.append(float(f.labels.mean()))
    return np.asarray(rates)


def stratified_lockbox_split(files: Sequence[FileData], lockbox_frac: float = 0.18) -> Tuple[List[str], List[str]]:
    file_ids = [f.file_id for f in files]
    rates = file_positive_rates(files)
    bins = np.digitize(rates, np.quantile(rates, [0.25, 0.5, 0.75]), right=True)
    rng = np.random.RandomState(123)
    dev_ids, lockbox_ids = [], []
    for bucket in sorted(set(bins.tolist())):
        idxs = np.where(bins == bucket)[0].tolist()
        rng.shuffle(idxs)
        n_lock = max(1, int(round(len(idxs) * lockbox_frac)))
        lock_idxs = idxs[:n_lock]
        dev_idxs = idxs[n_lock:]
        lockbox_ids.extend(file_ids[i] for i in lock_idxs)
        dev_ids.extend(file_ids[i] for i in dev_idxs)
    return sorted(dev_ids), sorted(lockbox_ids)


def make_cv_splits(dev_files: Sequence[FileData], n_splits: int = 3) -> List[Tuple[List[str], List[str]]]:
    file_ids = np.asarray([f.file_id for f in dev_files])
    rates = file_positive_rates(dev_files)
    bins = np.digitize(rates, np.quantile(rates, [0.25, 0.5, 0.75]), right=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=321)
    splits = []
    for train_idx, val_idx in skf.split(file_ids, bins):
        splits.append((file_ids[train_idx].tolist(), file_ids[val_idx].tolist()))
    return splits


def train_one_split(
    cfg: CandidateConfig,
    train_files: Sequence[FileData],
    val_files: Sequence[FileData],
    device: torch.device,
) -> Dict[str, object]:
    train_ds = AugmentedLidarDataset(train_files, cfg.context_len, True, cfg)
    val_ds = AugmentedLidarDataset(val_files, cfg.context_len, False, cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    positive_count = sum(int(f.labels.sum()) for f in train_files)
    total_count = sum(int(f.labels.size) for f in train_files)
    negative_count = total_count - positive_count
    pos_weight = (negative_count / max(positive_count, 1)) * cfg.pos_weight_scale

    sample_x, _ = train_ds[0]
    model = build_model(cfg, sample_x.shape[0], device)
    criterion = build_criterion(cfg, pos_weight, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler("cuda", enabled=True)

    best_state = None
    best_threshold = 0.5
    best_score = -1.0
    best_val_metrics = None
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, True, scaler)
        val_metrics, val_logits, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, False, None)
        threshold, tuned_metrics = find_best_threshold(val_logits, val_labels)
        if tuned_metrics["f1_obstacle"] > best_score:
            best_score = tuned_metrics["f1_obstacle"]
            best_threshold = threshold
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_val_metrics = tuned_metrics
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
        print(
            f"[{cfg.name}] epoch={epoch} train_f1={train_metrics['f1_obstacle']:.4f} "
            f"val_f1_tuned={tuned_metrics['f1_obstacle']:.4f} thr={threshold:.3f}"
        )

    assert best_state is not None and best_val_metrics is not None
    return {
        "state": best_state,
        "best_threshold": best_threshold,
        "best_val_metrics": best_val_metrics,
    }


def evaluate_with_threshold(
    model: nn.Module,
    files: Sequence[FileData],
    cfg: CandidateConfig,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    ds = AugmentedLidarDataset(files, cfg.context_len, False, cfg)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    _, logits, labels = run_epoch(model, loader, criterion, optimizer, device, False, None)
    metrics = confusion_metrics(logits, labels, threshold=threshold)
    metrics["fpr_non_obstacle"] = metrics["fp_obstacle"] / (metrics["fp_obstacle"] + metrics["tn_non_obstacle"])
    metrics["fnr_obstacle"] = metrics["fn_obstacle"] / (metrics["fn_obstacle"] + metrics["tp_obstacle"])
    return metrics


def mean_metric(results: Sequence[Dict[str, float]], key: str) -> float:
    return float(np.mean([r[key] for r in results]))


def main() -> None:
    set_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    LOCKBOX_DIR.mkdir(parents=True, exist_ok=True)

    pairs = discover_files(ROOT)
    files = [load_file_pair(raw, label) for raw, label in pairs]
    dev_ids, lockbox_ids = stratified_lockbox_split(files)
    dev_files = [f for f in files if f.file_id in dev_ids]
    lockbox_files = [f for f in files if f.file_id in lockbox_ids]
    cv_splits = make_cv_splits(dev_files, n_splits=3)

    protocol = {
        "note": "Fresh lockbox split untouched during candidate selection. Candidates ranked by 3-fold CV on development files only.",
        "dev_files": dev_ids,
        "lockbox_files": lockbox_ids,
    }
    (LOCKBOX_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2))

    candidates = [
        CandidateConfig("base_ctx4_focal_pw15", 4, "residual", (64, 64, 64, 64), 5, 3, 0.10, "focal", 1.5, 1.5),
        CandidateConfig("ctx6_focal_pw15", 6, "residual", (64, 64, 64, 64), 5, 3, 0.10, "focal", 1.5, 1.5),
        CandidateConfig("ctx8_focal_pw10", 8, "residual", (64, 64, 64, 64), 3, 3, 0.10, "focal", 1.5, 1.0),
        CandidateConfig("ctx4_focal_noise", 4, "residual", (64, 64, 64, 64), 5, 3, 0.10, "focal", 1.5, 1.5, lidar_noise_std=0.02),
        CandidateConfig("ctx4_focal_beamdrop", 4, "residual", (64, 64, 64, 64), 5, 3, 0.10, "focal", 1.5, 1.5, beam_dropout=0.05),
        CandidateConfig("ctx4_focal_timestepdrop", 4, "residual", (64, 64, 64, 64), 5, 3, 0.10, "focal", 1.5, 1.5, timestep_dropout=0.10),
        CandidateConfig("ctx4_focal_noise_beamdrop", 4, "residual", (64, 64, 64, 64), 5, 3, 0.10, "focal", 1.5, 1.5, lidar_noise_std=0.02, beam_dropout=0.05),
        CandidateConfig("ctx4_focal_deeper", 4, "residual", (64, 64, 64, 64, 64, 64), 5, 3, 0.12, "focal", 1.5, 1.5),
        CandidateConfig("ctx4_focal_gamma2", 4, "residual", (64, 64, 64, 64), 5, 3, 0.10, "focal", 2.0, 1.5),
    ]

    candidate_summaries = []
    for cfg in candidates:
        fold_results = []
        print(f"\n=== Candidate: {cfg.name} ===")
        for fold_idx, (train_ids, val_ids) in enumerate(cv_splits, start=1):
            train_files = [f for f in dev_files if f.file_id in train_ids]
            val_files = [f for f in dev_files if f.file_id in val_ids]
            result = train_one_split(cfg, train_files, val_files, device)
            fold_metrics = result["best_val_metrics"]
            fold_metrics["threshold"] = result["best_threshold"]
            fold_results.append(fold_metrics)
            print(
                f"[{cfg.name}] fold={fold_idx} val_f1={fold_metrics['f1_obstacle']:.4f} "
                f"prec={fold_metrics['precision_obstacle']:.4f} rec={fold_metrics['recall_obstacle']:.4f} "
                f"thr={result['best_threshold']:.3f}"
            )
        summary = {
            "config": asdict(cfg),
            "cv_mean_f1_obstacle": mean_metric(fold_results, "f1_obstacle"),
            "cv_mean_precision_obstacle": mean_metric(fold_results, "precision_obstacle"),
            "cv_mean_recall_obstacle": mean_metric(fold_results, "recall_obstacle"),
            "cv_mean_accuracy": mean_metric(fold_results, "accuracy"),
            "cv_mean_threshold": mean_metric(fold_results, "threshold"),
            "fold_results": fold_results,
        }
        candidate_summaries.append(summary)

    candidate_summaries.sort(key=lambda x: x["cv_mean_f1_obstacle"], reverse=True)
    (LOCKBOX_DIR / "cv_results.json").write_text(json.dumps(candidate_summaries, indent=2))

    finalists = candidate_summaries[:3]
    final_reports = []
    for finalist in finalists:
        cfg = CandidateConfig(**finalist["config"])
        print(f"\n=== Finalist retrain on dev: {cfg.name} ===")
        holdout_pool = dev_files
        rates = file_positive_rates(holdout_pool)
        bins = np.digitize(rates, np.quantile(rates, [0.25, 0.5, 0.75]), right=True)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=222)
        train_idx, val_idx = next(iter(skf.split(np.asarray([f.file_id for f in holdout_pool]), bins)))
        train_files = [holdout_pool[i] for i in train_idx]
        val_files = [holdout_pool[i] for i in val_idx]
        train_result = train_one_split(cfg, train_files, val_files, device)

        full_train_ds = AugmentedLidarDataset(holdout_pool, cfg.context_len, True, cfg)
        positive_count = sum(int(f.labels.sum()) for f in holdout_pool)
        total_count = sum(int(f.labels.size) for f in holdout_pool)
        negative_count = total_count - positive_count
        pos_weight = (negative_count / max(positive_count, 1)) * cfg.pos_weight_scale
        sample_x, _ = full_train_ds[0]
        model = build_model(cfg, sample_x.shape[0], device)
        model.load_state_dict(train_result["state"])
        criterion = build_criterion(cfg, pos_weight, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        lockbox_metrics = evaluate_with_threshold(
            model, lockbox_files, cfg, criterion, optimizer, device, threshold=train_result["best_threshold"]
        )
        final_reports.append(
            {
                "config": asdict(cfg),
                "cv_summary": finalist,
                "lockbox_threshold": train_result["best_threshold"],
                "lockbox_metrics": lockbox_metrics,
            }
        )
        print(
            f"[LOCKBOX {cfg.name}] f1={lockbox_metrics['f1_obstacle']:.4f} "
            f"prec={lockbox_metrics['precision_obstacle']:.4f} rec={lockbox_metrics['recall_obstacle']:.4f} "
            f"fpr_non={lockbox_metrics['fpr_non_obstacle']:.4f} fnr_obs={lockbox_metrics['fnr_obstacle']:.4f}"
        )

    final_reports.sort(key=lambda x: x["lockbox_metrics"]["f1_obstacle"], reverse=True)
    (LOCKBOX_DIR / "lockbox_results.json").write_text(json.dumps(final_reports, indent=2))

    md_lines = [
        "# Lockbox Ablation Search",
        "",
        "Fresh lockbox files were held out during candidate selection. Candidates were ranked using 3-fold CV on development files only.",
        "",
        "## CV Ranking",
    ]
    for row in candidate_summaries:
        cfg = row["config"]
        md_lines.append(
            f"- `{cfg['name']}`: cv_f1={row['cv_mean_f1_obstacle']:.4f}, "
            f"cv_prec={row['cv_mean_precision_obstacle']:.4f}, cv_rec={row['cv_mean_recall_obstacle']:.4f}, "
            f"cv_thr={row['cv_mean_threshold']:.3f}"
        )
    md_lines.append("")
    md_lines.append("## Lockbox Results")
    for row in final_reports:
        cfg = row["config"]
        m = row["lockbox_metrics"]
        md_lines.append(
            f"- `{cfg['name']}`: lockbox_f1={m['f1_obstacle']:.4f}, prec={m['precision_obstacle']:.4f}, "
            f"rec={m['recall_obstacle']:.4f}, acc={m['accuracy']:.4f}, fpr_non={m['fpr_non_obstacle']:.4f}, "
            f"fnr_obs={m['fnr_obstacle']:.4f}, fp={m['fp_obstacle']}, fn={m['fn_obstacle']}, "
            f"threshold={row['lockbox_threshold']:.3f}"
        )
    (LOCKBOX_DIR / "report.md").write_text("\n".join(md_lines) + "\n")
    print(f"Saved lockbox search outputs to {LOCKBOX_DIR}")


if __name__ == "__main__":
    main()
