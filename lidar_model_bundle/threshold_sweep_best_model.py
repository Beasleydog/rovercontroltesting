import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from train_lidar_cnn import (
    ROOT,
    OUT_DIR,
    ExperimentConfig,
    FocalLoss,
    LidarSequenceDataset,
    ResidualBeamCNN,
    BeamTemporalCNN,
    autocast,
    confusion_metrics,
    discover_files,
    find_best_threshold,
    load_file_pair,
    run_epoch,
    set_seed,
    split_file_ids,
)


def build_model(config: ExperimentConfig, in_channels: int, device: torch.device) -> nn.Module:
    if config.model_type == "plain":
        model = BeamTemporalCNN(
            in_channels=in_channels,
            widths=config.widths,
            kernel_t=config.kernel_t,
            kernel_b=config.kernel_b,
            dropout=config.dropout,
        )
    elif config.model_type == "residual":
        model = ResidualBeamCNN(
            in_channels=in_channels,
            width=config.widths[0],
            depth=len(config.widths),
            kernel_t=config.kernel_t,
            kernel_b=config.kernel_b,
            dropout=config.dropout,
        )
    else:
        raise ValueError(config.model_type)
    return model.to(device)


def build_criterion(config: ExperimentConfig, pos_weight: float, device: torch.device) -> nn.Module:
    if config.loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device, dtype=torch.float32))
    if config.loss_type == "focal":
        return FocalLoss(pos_weight=pos_weight, gamma=config.focal_gamma)
    raise ValueError(config.loss_type)


def main() -> None:
    set_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    config = ExperimentConfig(
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
    )

    pairs = discover_files(ROOT)
    files = [load_file_pair(raw, label) for raw, label in pairs]
    split_ids = split_file_ids(files)

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
    model = build_model(config, sample_x.shape[0], device)
    criterion = build_criterion(config, pos_weight, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = GradScaler("cuda", enabled=True)

    best_state = None
    best_threshold = 0.5
    best_val_f1 = -1.0
    patience = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        val_metrics, val_logits, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False, scaler=None)
        tuned_threshold, tuned_val_metrics = find_best_threshold(val_logits, val_labels)
        print(
            f"epoch={epoch} train_f1={train_metrics['f1_obstacle']:.4f} "
            f"val_f1={val_metrics['f1_obstacle']:.4f} tuned_val_f1={tuned_val_metrics['f1_obstacle']:.4f} "
            f"thr={tuned_threshold:.3f}"
        )
        if tuned_val_metrics["f1_obstacle"] > best_val_f1:
            best_val_f1 = tuned_val_metrics["f1_obstacle"]
            best_threshold = tuned_threshold
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("No best state found.")
    model.load_state_dict(best_state)
    _, test_logits, test_labels = run_epoch(model, test_loader, criterion, optimizer, device, train=False, scaler=None)

    thresholds = np.arange(0.40, 0.901, 0.025)
    rows = []
    for threshold in thresholds:
        metrics = confusion_metrics(test_logits, test_labels, threshold=float(threshold))
        fp = metrics["fp_obstacle"]
        tn = metrics["tn_non_obstacle"]
        fn = metrics["fn_obstacle"]
        tp = metrics["tp_obstacle"]
        metrics["false_positive_rate_non_obstacle"] = fp / (fp + tn) if fp + tn else 0.0
        metrics["false_negative_rate_obstacle"] = fn / (fn + tp) if fn + tp else 0.0
        rows.append(metrics)

    out_csv = OUT_DIR / "best_model_threshold_sweep.csv"
    out_md = OUT_DIR / "best_model_threshold_sweep.md"
    out_json = OUT_DIR / "best_model_threshold_sweep.json"

    fieldnames = [
        "threshold",
        "accuracy",
        "precision_obstacle",
        "recall_obstacle",
        "f1_obstacle",
        "precision_non_obstacle",
        "recall_non_obstacle",
        "f1_non_obstacle",
        "false_positive_rate_non_obstacle",
        "false_negative_rate_obstacle",
        "fp_obstacle",
        "fn_obstacle",
        "tp_obstacle",
        "tn_non_obstacle",
        "positive_rate_pred",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    lines = [
        "# Threshold Sweep For Best Model",
        "",
        f"- Model: `{config.name}`",
        f"- Validation-selected threshold during training: `{best_threshold:.3f}`",
        "",
        "| threshold | acc | obs_prec | obs_rec | obs_f1 | non_obs_FPR | obs_FNR | FP | FN | TP | TN | pred_pos_rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']:.3f} | {row['accuracy']:.4f} | {row['precision_obstacle']:.4f} | "
            f"{row['recall_obstacle']:.4f} | {row['f1_obstacle']:.4f} | "
            f"{row['false_positive_rate_non_obstacle']:.4f} | {row['false_negative_rate_obstacle']:.4f} | "
            f"{row['fp_obstacle']} | {row['fn_obstacle']} | {row['tp_obstacle']} | {row['tn_non_obstacle']} | "
            f"{row['positive_rate_pred']:.4f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    out_json.write_text(json.dumps({"config": config.__dict__, "best_threshold": best_threshold, "rows": rows}, indent=2))

    print(f"Saved threshold sweep to {out_csv}")
    print(f"Saved threshold sweep to {out_md}")
    print(f"Saved threshold sweep to {out_json}")


if __name__ == "__main__":
    main()
