import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_lidar_cnn import (
    BeamTemporalCNN,
    ResidualBeamCNN,
    beam_columns,
    normalize_heading_deg,
    sanitize_lidar,
)


def build_features(df: pd.DataFrame, context_len: int) -> np.ndarray:
    raw_beams = beam_columns(df.columns)
    lidar_raw = df[raw_beams].to_numpy(dtype=np.float32)
    lidar, valid = sanitize_lidar(lidar_raw)

    pose_xyz = df[["rover_pos_x", "rover_pos_y", "rover_pos_z"]].to_numpy(dtype=np.float32)
    heading_sin, heading_cos = normalize_heading_deg(df["heading"].to_numpy(dtype=np.float32))
    orient = np.stack(
        [
            heading_sin,
            heading_cos,
            df["pitch"].to_numpy(dtype=np.float32) / 45.0,
            df["roll"].to_numpy(dtype=np.float32) / 45.0,
        ],
        axis=1,
    ).astype(np.float32)
    deltas = np.zeros_like(pose_xyz, dtype=np.float32)
    deltas[1:] = pose_xyz[1:] - pose_xyz[:-1]
    deltas[:, 0] /= 1000.0
    deltas[:, 1] /= 1000.0
    deltas[:, 2] /= 100.0

    beam_index = np.linspace(-1.0, 1.0, len(raw_beams), dtype=np.float32)
    samples = []
    step_idxs = []
    for end_idx in range(context_len - 1, len(df)):
        start_idx = end_idx - context_len + 1
        lidar_chunk = lidar[start_idx:end_idx + 1]
        valid_chunk = valid[start_idx:end_idx + 1]
        pose_scaled = pose_xyz[start_idx:end_idx + 1].copy()
        pose_scaled[:, 0] /= 20000.0
        pose_scaled[:, 1] /= 20000.0
        pose_scaled[:, 2] /= 2000.0
        orient_chunk = orient[start_idx:end_idx + 1]
        delta_chunk = deltas[start_idx:end_idx + 1]

        channels = [lidar_chunk, valid_chunk]
        for i in range(pose_scaled.shape[1]):
            channels.append(np.repeat(pose_scaled[:, i:i + 1], len(raw_beams), axis=1))
        for i in range(orient_chunk.shape[1]):
            channels.append(np.repeat(orient_chunk[:, i:i + 1], len(raw_beams), axis=1))
        for i in range(delta_chunk.shape[1]):
            channels.append(np.repeat(delta_chunk[:, i:i + 1], len(raw_beams), axis=1))
        channels.append(np.repeat(beam_index[None, :], context_len, axis=0))
        samples.append(np.stack(channels, axis=0).astype(np.float32))
        step_idxs.append(int(df.iloc[end_idx]["step_idx"]))
    return np.stack(samples), step_idxs, raw_beams


def make_model(meta: dict, in_channels: int) -> torch.nn.Module:
    cfg = meta["model_config"]
    if cfg["model_type"] == "plain":
        return BeamTemporalCNN(
            in_channels=in_channels,
            widths=tuple(cfg["widths"]),
            kernel_t=cfg["kernel_t"],
            kernel_b=cfg["kernel_b"],
            dropout=cfg["dropout"],
        )
    return ResidualBeamCNN(
        in_channels=in_channels,
        width=cfg["widths"][0],
        depth=len(cfg["widths"]),
        kernel_t=cfg["kernel_t"],
        kernel_b=cfg["kernel_b"],
        dropout=cfg["dropout"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--bundle_dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    meta = json.loads((bundle_dir / "model_metadata.json").read_text())
    checkpoint = torch.load(bundle_dir / "best_model_checkpoint.pt", map_location="cpu")

    df = pd.read_csv(args.input_csv).sort_values("step_idx").reset_index(drop=True)
    x, step_idxs, beam_names = build_features(df, meta["model_config"]["context_len"])
    model = make_model(meta, x.shape[1])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        probs = torch.sigmoid(logits).numpy()

    threshold = meta["recommended_threshold"]
    preds = (probs >= threshold).astype(int)

    out = pd.DataFrame({"step_idx": step_idxs})
    for i, beam in enumerate(beam_names):
        out[f"{beam}_prob_obstacle"] = probs[:, i]
        out[f"{beam}_pred_obstacle"] = preds[:, i]

    output_csv = Path(args.output_csv) if args.output_csv else Path(args.input_csv).with_suffix(".predictions.csv")
    out.to_csv(output_csv, index=False)
    print(output_csv)


if __name__ == "__main__":
    main()
