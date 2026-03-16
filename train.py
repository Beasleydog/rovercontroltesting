from __future__ import annotations

import argparse
import os
import csv
import json
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable, TextIO

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


@dataclass(frozen=True)
class WorldData:
    pose: np.ndarray
    lidar_cm: np.ndarray
    lidar_class: np.ndarray
    teleport_flag: np.ndarray


@dataclass(frozen=True)
class NormStats:
    pose_mean: np.ndarray
    pose_std: np.ndarray
    dist_mean: float
    dist_std: float


# Fixed rover sensor geometry used to inject explicit physical structure.
# These match the simulator layout in environment_demo.py.
MODEL_SENSOR_COORDS_CM = np.array(
    [
        (250.0, 245.0, 50.0),
        (325.0, 75.0, 130.0),
        (325.0, 0.0, 130.0),
        (325.0, -75.0, 130.0),
        (250.0, -245.0, 50.0),
        (325.0, 75.0, 130.0),
        (325.0, -75.0, 130.0),
        (40.0, 235.0, 100.0),
        (40.0, -235.0, 100.0),
        (-215.0, 270.0, 70.0),
        (-320.0, 80.0, 10.0),
        (-320.0, -50.0, 10.0),
        (-215.0, -215.0, 70.0),
        (325.0, 75.0, 130.0),
        (325.0, -75.0, 130.0),
        (250.0, 245.0, 50.0),
        (250.0, -245.0, 50.0),
    ],
    dtype=np.float32,
)
MODEL_SENSOR_YAW_PITCH_DEG = np.array(
    [
        (30.0, 0.0),
        (20.0, -20.0),
        (0.0, 0.0),
        (-20.0, -20.0),
        (-30.0, 0.0),
        (0.0, -25.0),
        (0.0, -25.0),
        (90.0, -20.0),
        (-90.0, -20.0),
        (140.0, 0.0),
        (180.0, 0.0),
        (180.0, 0.0),
        (-140.0, 0.0),
        (20.0, -10.0),
        (-20.0, -10.0),
        (15.0, 0.0),
        (-15.0, 0.0),
    ],
    dtype=np.float32,
)


def _sensor_dirs_from_yaw_pitch_deg(yaw_pitch_deg: np.ndarray) -> np.ndarray:
    yaw = np.deg2rad(yaw_pitch_deg[:, 0].astype(np.float32))
    pitch = np.deg2rad(yaw_pitch_deg[:, 1].astype(np.float32))
    cp = np.cos(pitch)
    return np.stack(
        [
            cp * np.cos(yaw),
            cp * np.sin(yaw),
            np.sin(pitch),
        ],
        axis=1,
    ).astype(np.float32)


MODEL_SENSOR_DIRS_LOCAL = _sensor_dirs_from_yaw_pitch_deg(MODEL_SENSOR_YAW_PITCH_DEG)
LEGACY_FEATURE_MODE = "legacy_normalized"
EGO_MAP_FEATURE_MODE = "ego_map_raw"
DEFAULT_LIDAR_MAX_RANGE_CM = 1000.0
POSE_FEATURE_DIM_LEGACY = 4
POSE_FEATURE_DIM_EGO_MAP = 12


_LOG_FILE: TextIO | None = None


def set_log_file(log_file: TextIO | None) -> None:
    global _LOG_FILE
    _LOG_FILE = log_file


def _emit_log_line(line: str) -> None:
    print(line, flush=True)
    if _LOG_FILE is not None:
        _LOG_FILE.write(line + "\n")
        _LOG_FILE.flush()


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    _emit_log_line(f"[{stamp}] {message}")


def log_plain(message: str) -> None:
    _emit_log_line(message)


def resolve_data_loader_workers(requested: int) -> int:
    req = int(requested)
    if req >= 0:
        return req
    cpu_count = max(int(os.cpu_count() or 1), 1)
    return max(1, min(8, cpu_count - 1))


def configure_runtime_for_device(device_kind: str) -> None:
    if device_kind != "cuda":
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is not None and hasattr(cuda_backend, "matmul"):
        cuda_backend.matmul.allow_tf32 = True
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        cudnn_backend.allow_tf32 = True
        # Prefer deterministic kernel selection over aggressive autotuning here:
        # some CUDA stacks pick unstable/unsupported kernels for this small conv net.
        cudnn_backend.benchmark = False


def select_runtime_device(preferred: str | None = None) -> tuple[torch.device, str]:
    choice = "auto" if preferred is None else str(preferred).strip().lower()
    valid = {"auto", "cpu", "cuda", "mps", "xla"}
    if choice not in valid:
        raise ValueError(f"Unsupported device selection '{preferred}'. Choose from {sorted(valid)}")

    if choice in {"auto", "xla"}:
        try:
            import torch_xla.core.xla_model as xm

            return xm.xla_device(), "xla"
        except Exception:
            if choice == "xla":
                raise

    if choice in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if choice in {"auto", "mps"} and mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"

    if choice == "auto" or choice == "cpu":
        return torch.device("cpu"), "cpu"

    raise RuntimeError(f"Requested device '{choice}' is not available in this environment")


def optimizer_step_for_device(optimizer: torch.optim.Optimizer, device_kind: str) -> None:
    if device_kind == "xla":
        import torch_xla.core.xla_model as xm

        xm.optimizer_step(optimizer, barrier=False)
        xm.mark_step()
        return
    optimizer.step()


def save_checkpoint_for_device(checkpoint: dict, path: Path, device_kind: str) -> None:
    if device_kind == "xla":
        import torch_xla.core.xla_model as xm

        xm.save(checkpoint, path)
        return
    torch.save(checkpoint, path)


def load_checkpoint_for_device(path: str | Path, device: torch.device, device_kind: str) -> dict:
    map_location: str | torch.device = "cpu" if device_kind == "xla" else device
    return torch.load(path, map_location=map_location)


def load_model_state_with_compat(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    log_fn=None,
) -> None:
    load_result = model.load_state_dict(state_dict, strict=False)
    if log_fn is None:
        return
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    if not missing and not unexpected:
        return
    parts: list[str] = []
    if missing:
        parts.append(f"missing={len(missing)}")
    if unexpected:
        parts.append(f"unexpected={len(unexpected)}")
    log_fn(
        "Loaded checkpoint with compatibility fallback "
        f"({', '.join(parts)})."
    )


def _split_single_world_for_validation(world: WorldData, val_fraction: float) -> tuple[WorldData, WorldData]:
    t_steps = world.pose.shape[0]
    if t_steps < 2:
        # Not enough timeline for a clean temporal split; fallback to shared world.
        return world, world
    split_idx = int(round(t_steps * (1.0 - val_fraction)))
    split_idx = max(1, min(t_steps - 1, split_idx))
    train_world = WorldData(
        pose=world.pose[:split_idx].copy(),
        lidar_cm=world.lidar_cm[:split_idx].copy(),
        lidar_class=world.lidar_class[:split_idx].copy(),
        teleport_flag=world.teleport_flag[:split_idx].copy(),
    )
    val_world = WorldData(
        pose=world.pose[split_idx:].copy(),
        lidar_cm=world.lidar_cm[split_idx:].copy(),
        lidar_class=world.lidar_class[split_idx:].copy(),
        teleport_flag=world.teleport_flag[split_idx:].copy(),
    )
    return train_world, val_world


def _sensor_keys(fieldnames: list[str], prefix: str) -> list[str]:
    keys = [k for k in fieldnames if k.startswith(prefix)]
    keys.sort(key=lambda k: int(k.rsplit("_", 1)[1]))
    return keys


def _basis_fieldnames() -> list[str]:
    return [
        "basis_xx",
        "basis_xy",
        "basis_xz",
        "basis_yx",
        "basis_yy",
        "basis_yz",
        "basis_zx",
        "basis_zy",
        "basis_zz",
    ]


def _flat_basis_from_yaw_deg(yaw_deg: np.ndarray) -> np.ndarray:
    yaw_rad = np.deg2rad(yaw_deg.astype(np.float32))
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    basis = np.zeros((yaw_deg.shape[0], 3, 3), dtype=np.float32)
    basis[:, 0, 0] = cos_yaw
    basis[:, 1, 0] = sin_yaw
    basis[:, 0, 1] = -sin_yaw
    basis[:, 1, 1] = cos_yaw
    basis[:, 2, 2] = 1.0
    return basis


def pose_array_to_legacy_xyzyaw(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 2:
        raise ValueError("pose must be shape [T,P]")
    if pose.shape[1] == POSE_FEATURE_DIM_LEGACY:
        return pose
    if pose.shape[1] != POSE_FEATURE_DIM_EGO_MAP:
        raise ValueError(f"Unsupported pose feature dim {pose.shape[1]}; expected 4 or 12")
    origin = pose[:, :3]
    basis = pose[:, 3:].reshape(-1, 3, 3)
    yaw = np.rad2deg(np.arctan2(basis[:, 1, 0], basis[:, 0, 0])).astype(np.float32)
    return np.concatenate([origin, yaw[:, None]], axis=1).astype(np.float32)


def pose_array_to_geometry_features(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 2:
        raise ValueError("pose must be shape [T,P]")
    if pose.shape[1] == POSE_FEATURE_DIM_EGO_MAP:
        return pose
    if pose.shape[1] != POSE_FEATURE_DIM_LEGACY:
        raise ValueError(f"Unsupported pose feature dim {pose.shape[1]}; expected 4 or 12")
    origin = pose[:, :3]
    yaw = pose[:, 3]
    basis = _flat_basis_from_yaw_deg(yaw).reshape(-1, 9)
    return np.concatenate([origin, basis], axis=1).astype(np.float32)


def assemble_pose_features(
    pose_values: np.ndarray,
    target_pose_dim: int,
    basis: np.ndarray | None = None,
) -> np.ndarray:
    pose_arr = np.asarray(pose_values, dtype=np.float32).reshape(-1)
    if pose_arr.shape[0] == int(target_pose_dim):
        return pose_arr.astype(np.float32, copy=False)
    if int(target_pose_dim) == POSE_FEATURE_DIM_LEGACY:
        if pose_arr.shape[0] == 3 and basis is not None:
            basis_arr = np.asarray(basis, dtype=np.float32).reshape(3, 3)
            yaw = float(np.rad2deg(np.arctan2(basis_arr[1, 0], basis_arr[0, 0])))
            return np.concatenate([pose_arr, np.asarray([yaw], dtype=np.float32)], axis=0).astype(np.float32)
        if pose_arr.shape[0] == POSE_FEATURE_DIM_EGO_MAP:
            return pose_array_to_legacy_xyzyaw(pose_arr[None, :])[0]
    if int(target_pose_dim) == POSE_FEATURE_DIM_EGO_MAP:
        if pose_arr.shape[0] == 3 and basis is not None:
            basis_arr = np.asarray(basis, dtype=np.float32).reshape(9)
            return np.concatenate([pose_arr, basis_arr], axis=0).astype(np.float32)
        if pose_arr.shape[0] == POSE_FEATURE_DIM_LEGACY:
            return pose_array_to_geometry_features(pose_arr[None, :])[0]
    raise ValueError(
        f"Cannot assemble pose features with input dim={pose_arr.shape[0]} target dim={target_pose_dim}"
    )


def load_world_file(path: Path) -> WorldData:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        cm_keys = _sensor_keys(reader.fieldnames, "lidar_cm_")
        class_keys = _sensor_keys(reader.fieldnames, "lidar_class_")
        if len(cm_keys) == 0 or len(cm_keys) != len(class_keys):
            raise ValueError(f"{path} has invalid lidar columns")
        basis_keys = [name for name in _basis_fieldnames() if name in reader.fieldnames]
        has_basis_cols = len(basis_keys) == 9

        pose_rows: list[list[float]] = []
        cm_rows: list[list[float]] = []
        cls_rows: list[list[int]] = []
        teleport_rows: list[int] = []
        has_teleport_col = "teleport_flag" in reader.fieldnames
        for row in reader:
            base_pose = [
                float(row["x_cm"]),
                float(row["y_cm"]),
                float(row["z_cm"]),
            ]
            if has_basis_cols:
                base_pose.extend(float(row[name]) for name in basis_keys)
            else:
                base_pose.append(float(row["yaw_deg"]))
            pose_rows.append(base_pose)
            cm_rows.append([float(row[k]) for k in cm_keys])
            cls_rows.append([int(row[k]) for k in class_keys])
            teleport_rows.append(int(float(row["teleport_flag"])) if has_teleport_col else 0)

    if len(pose_rows) < 1:
        raise ValueError(f"{path} needs at least 1 timestep for current-step classification")
    return WorldData(
        pose=np.asarray(pose_rows, dtype=np.float32),
        lidar_cm=np.asarray(cm_rows, dtype=np.float32),
        lidar_class=np.asarray(cls_rows, dtype=np.int64),
        teleport_flag=np.asarray(teleport_rows, dtype=np.int64),
    )


def compute_norm_stats(worlds: Iterable[WorldData], feature_mode: str) -> NormStats:
    if feature_mode == EGO_MAP_FEATURE_MODE:
        return NormStats(
            pose_mean=np.zeros((POSE_FEATURE_DIM_EGO_MAP,), dtype=np.float32),
            pose_std=np.ones((POSE_FEATURE_DIM_EGO_MAP,), dtype=np.float32),
            dist_mean=0.0,
            dist_std=1.0,
        )

    pose_all = np.concatenate([pose_array_to_legacy_xyzyaw(w.pose) for w in worlds], axis=0)
    pose_mean = pose_all.mean(axis=0)
    pose_std = pose_all.std(axis=0) + 1e-6

    dists = np.concatenate([w.lidar_cm for w in worlds], axis=0)
    valid = dists >= 0.0
    if np.any(valid):
        dist_mean = float(dists[valid].mean())
        dist_std = float(dists[valid].std() + 1e-6)
    else:
        dist_mean = 0.0
        dist_std = 1.0
    return NormStats(
        pose_mean=pose_mean.astype(np.float32),
        pose_std=pose_std.astype(np.float32),
        dist_mean=dist_mean,
        dist_std=dist_std,
    )


def world_to_features(
    world: WorldData,
    stats: NormStats,
    feature_mode: str,
    no_hit_range_cm: float = DEFAULT_LIDAR_MAX_RANGE_CM,
) -> tuple[np.ndarray, np.ndarray]:
    hit = (world.lidar_cm >= 0.0).astype(np.float32)

    if feature_mode == EGO_MAP_FEATURE_MODE:
        pose = pose_array_to_geometry_features(world.pose)
        dist = np.where(hit > 0.0, world.lidar_cm, float(no_hit_range_cm)).astype(np.float32)
    else:
        pose_raw = pose_array_to_legacy_xyzyaw(world.pose)
        pose = (pose_raw - stats.pose_mean[None, :]) / stats.pose_std[None, :]
        dist = np.where(hit > 0.0, world.lidar_cm, 0.0)
        dist = (dist - stats.dist_mean) / stats.dist_std
        dist = np.where(hit > 0.0, dist, 0.0).astype(np.float32)

    features = np.concatenate([pose.astype(np.float32), dist, hit], axis=1)
    targets = world.lidar_class.astype(np.int64)
    return features, targets


def featurize_timestep(
    pose_values: np.ndarray,
    lidar_cm: np.ndarray,
    stats: NormStats,
    basis: np.ndarray | None = None,
    no_hit_range_cm: float = DEFAULT_LIDAR_MAX_RANGE_CM,
) -> np.ndarray:
    target_pose_dim = int(stats.pose_mean.shape[0])
    pose = assemble_pose_features(pose_values, target_pose_dim, basis=basis)
    hit = (lidar_cm >= 0.0).astype(np.float32)
    if target_pose_dim == POSE_FEATURE_DIM_EGO_MAP:
        dist = np.where(hit > 0.0, lidar_cm.astype(np.float32), float(no_hit_range_cm))
        return np.concatenate([pose.astype(np.float32), dist.astype(np.float32), hit], axis=0).astype(np.float32)

    pose = (pose.astype(np.float32) - stats.pose_mean) / stats.pose_std
    dist = np.where(hit > 0.0, lidar_cm.astype(np.float32), 0.0)
    dist = (dist - stats.dist_mean) / stats.dist_std
    dist = np.where(hit > 0.0, dist, 0.0)
    return np.concatenate([pose, dist.astype(np.float32), hit], axis=0).astype(np.float32)


class SequencePieceDataset(Dataset):
    def __init__(
        self,
        features_by_world: list[np.ndarray],
        targets_by_world: list[np.ndarray],
        teleport_flags_by_world: list[np.ndarray],
        max_history: int,
        min_history: int = 1,
        histories_per_target: int = 3,
        exclude_after_teleport_steps: int = 1,
        history_step_min: int = 1,
        history_step_max: int = 4,
        seed: int = 0,
    ):
        self.features_by_world = features_by_world
        self.targets_by_world = targets_by_world
        self.min_history = int(max(min_history, 1))
        self.histories_per_target = int(max(histories_per_target, 1))
        self.exclude_after_teleport_steps = int(max(exclude_after_teleport_steps, 0))
        self.history_step_min = int(max(history_step_min, 1))
        self.history_step_max = int(max(history_step_max, self.history_step_min))
        self.index: list[tuple[int, np.ndarray, int]] = []
        self.sample_has_obstacle_target: list[bool] = []
        rng = np.random.default_rng(seed)

        for wi, feat in enumerate(features_by_world):
            t_steps = feat.shape[0]
            tele_flags = teleport_flags_by_world[wi]
            invalid_targets = np.zeros((t_steps,), dtype=bool)
            tele_indices = np.flatnonzero(tele_flags > 0)
            for idx in tele_indices:
                start_excl = int(idx)
                end_excl = min(t_steps - 1, int(idx + self.exclude_after_teleport_steps))
                invalid_targets[start_excl : end_excl + 1] = True
            for end in range(t_steps):
                if invalid_targets[end]:
                    continue
                history_max = self._max_history_length_for_end(
                    end,
                    max_history,
                    self.history_step_min,
                )
                if history_max < self.min_history:
                    continue
                history_lengths = self._sample_history_lengths(
                    self.min_history,
                    history_max,
                    self.histories_per_target,
                    rng,
                )
                for hist_len in history_lengths:
                    time_indices = self._sample_history_indices(
                        end,
                        hist_len,
                        self.history_step_min,
                        self.history_step_max,
                        rng,
                    )
                    self.index.append((wi, time_indices, end))
                    self.sample_has_obstacle_target.append(bool(np.any(self.targets_by_world[wi][end] == 1)))

    @staticmethod
    def _max_history_length_for_end(end: int, max_history: int, step_min: int) -> int:
        feasible_from_start = (int(end) // max(int(step_min), 1)) + 1
        if max_history <= 0:
            return feasible_from_start
        return min(feasible_from_start, int(max_history))

    @staticmethod
    def _sample_history_lengths(
        history_min: int,
        history_max: int,
        k: int,
        rng: np.random.Generator,
    ) -> list[int]:
        history_min_i = int(max(history_min, 1))
        history_max_i = int(max(history_max, history_min_i))
        if history_max_i < history_min_i:
            return []
        total_choices = history_max_i - history_min_i + 1
        if total_choices <= k:
            return list(range(history_min_i, history_max_i + 1))
        if k <= 1:
            return [history_max_i]

        # Always include the shortest and longest allowed contexts.
        lengths = [history_min_i, history_max_i]
        remaining = k - len(lengths)
        if remaining <= 0:
            return sorted(set(lengths))

        candidates = np.arange(history_min_i + 1, history_max_i, dtype=np.int32)
        if candidates.size > 0:
            sampled = rng.choice(candidates, size=min(remaining, candidates.size), replace=False)
            lengths.extend(int(v) for v in sampled.tolist())
        return sorted(set(lengths))

    @staticmethod
    def _sample_history_indices(
        end: int,
        hist_len: int,
        step_min: int,
        step_max: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if hist_len <= 1:
            return np.asarray([int(end)], dtype=np.int64)

        step_min_i = max(int(step_min), 1)
        step_max_i = max(int(step_max), step_min_i)
        cursor = int(end)
        reverse_indices = [cursor]
        total_back_steps = hist_len - 1

        for pick_idx in range(total_back_steps):
            remaining_after_pick = total_back_steps - pick_idx - 1
            min_prev_index = remaining_after_pick * step_min_i
            max_allowed_step = min(step_max_i, cursor - min_prev_index)
            if max_allowed_step <= step_min_i:
                step = max_allowed_step
            else:
                step = int(rng.integers(step_min_i, max_allowed_step + 1))
            cursor -= max(step, 1)
            reverse_indices.append(cursor)

        reverse_indices.reverse()
        return np.asarray(reverse_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wi, time_indices, end = self.index[idx]
        x = self.features_by_world[wi][time_indices]
        y = self.targets_by_world[wi][end]
        return torch.from_numpy(x), torch.from_numpy(y)

    def obstacle_target_mask(self) -> np.ndarray:
        return np.asarray(self.sample_has_obstacle_target, dtype=bool)


def collate_padded(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([x.shape[0] for x, _ in batch], dtype=torch.long)
    feat_dim = batch[0][0].shape[1]
    max_len = int(lengths.max().item())
    x_pad = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
    y = torch.stack([yb for _, yb in batch], dim=0).long()

    for i, (xb, _) in enumerate(batch):
        x_pad[i, : xb.shape[0], :] = xb
    return x_pad, lengths, y


def _default_region_sensor_groups(num_sensors: int) -> dict[str, list[int]]:
    if int(num_sensors) == 17:
        return {
            "front": [0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16],
            "left": [7],
            "right": [8],
            "rear": [9, 10, 11, 12],
        }
    chunks = np.array_split(np.arange(int(num_sensors), dtype=np.int64), 4)
    return {
        "front": [int(v) for v in chunks[0].tolist()],
        "left": [int(v) for v in chunks[1].tolist()],
        "right": [int(v) for v in chunks[2].tolist()],
        "rear": [int(v) for v in chunks[3].tolist()],
    }


def _normalize_region_sensor_groups(
    region_sensor_groups: dict[str, list[int]] | None,
    num_sensors: int,
) -> dict[str, list[int]]:
    base = _default_region_sensor_groups(num_sensors) if region_sensor_groups is None else region_sensor_groups
    ordered_names = ("front", "left", "right", "rear")
    normalized: dict[str, list[int]] = {}
    used: set[int] = set()
    for name in ordered_names:
        raw = base.get(name, [])
        ids: list[int] = []
        for idx in raw:
            idx_i = int(idx)
            if 0 <= idx_i < int(num_sensors) and idx_i not in used:
                ids.append(idx_i)
                used.add(idx_i)
        if ids:
            normalized[name] = ids

    remaining = [i for i in range(int(num_sensors)) if i not in used]
    if remaining:
        normalized.setdefault("front", []).extend(remaining)
    if not normalized:
        normalized["front"] = list(range(int(num_sensors)))
    return normalized


def _default_region_hidden_dims(
    region_sensor_groups: dict[str, list[int]],
    hidden_dim: int,
) -> dict[str, int]:
    active_items = [(name, ids) for name, ids in region_sensor_groups.items() if ids]
    min_hidden = 8
    if not active_items:
        return {"front": max(min_hidden, int(hidden_dim))}

    target_budget = max(min_hidden * len(active_items), int(round(max(int(hidden_dim), 1) * 1.0)))
    weights = np.sqrt(np.asarray([len(ids) for _, ids in active_items], dtype=np.float32))
    weights /= max(float(weights.sum()), 1e-8)

    region_hidden_dims: dict[str, int] = {}
    for (name, _), weight in zip(active_items, weights):
        region_hidden_dims[name] = max(min_hidden, int(round(target_budget * float(weight))))
    return region_hidden_dims


def _build_sensor_processing_order(region_sensor_groups: dict[str, list[int]], num_sensors: int) -> list[int]:
    preferred = ("front", "left", "rear", "right")
    used: set[int] = set()
    ordered: list[int] = []
    for name in preferred:
        for sensor_id in region_sensor_groups.get(name, []):
            sid = int(sensor_id)
            if 0 <= sid < int(num_sensors) and sid not in used:
                ordered.append(sid)
                used.add(sid)
    for name, sensor_ids in region_sensor_groups.items():
        if name in preferred:
            continue
        for sensor_id in sensor_ids:
            sid = int(sensor_id)
            if 0 <= sid < int(num_sensors) and sid not in used:
                ordered.append(sid)
                used.add(sid)
    for sid in range(int(num_sensors)):
        if sid not in used:
            ordered.append(sid)
    return ordered


class CausalTimeSensorBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.temporal_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(3, 1),
            dilation=(int(dilation), 1),
            padding=(int(dilation) * 2, 0),
        )
        self.temporal_norm = nn.GroupNorm(1, channels)
        self.sensor_conv = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))
        self.sensor_norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        t = self.temporal_conv(x)
        t = t[:, :, : x.shape[2], :]
        t = F.gelu(self.temporal_norm(t))
        s = self.sensor_conv(t)
        s = self.sensor_norm(s)
        s = self.dropout(F.gelu(s))
        return residual + s


class TemporalDownsampleBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.temporal_stride = nn.Conv2d(
            channels,
            channels,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
            groups=channels,
            bias=False,
        )
        self.mix = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        with torch.no_grad():
            self.temporal_stride.weight.zero_()
            self.temporal_stride.weight[:, 0, 1, 0] = 1.0
            self.mix.weight.zero_()
            if self.mix.bias is not None:
                self.mix.bias.zero_()
            eye = torch.eye(
                self.mix.out_channels,
                self.mix.in_channels,
                device=self.mix.weight.device,
                dtype=self.mix.weight.dtype,
            )
            self.mix.weight[:, :, 0, 0] = eye

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_stride(x)
        x = self.mix(x)
        return F.gelu(self.norm(x))


class TemporalUNetFuse(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=(2, 1),
            stride=(2, 1),
            groups=channels,
            bias=False,
        )
        self.proj = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        with torch.no_grad():
            self.up.weight.zero_()
            self.up.weight[:, 0, :, 0] = 1.0

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return F.gelu(self.norm(self.proj(x)))


class SensorRefinementBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        with torch.no_grad():
            self.conv2.weight.zero_()
            if self.conv2.bias is not None:
                self.conv2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2).contiguous()
        y = F.gelu(self.norm1(self.conv1(y)))
        y = self.dropout(F.gelu(self.norm2(self.conv2(y))))
        return residual + y.transpose(1, 2).contiguous()


class MapResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        pad = int(max(dilation, 1))
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=pad, dilation=pad)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=pad, dilation=pad)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout2d(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        with torch.no_grad():
            self.conv2.weight.zero_()
            if self.conv2.bias is not None:
                self.conv2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.dropout(F.gelu(self.norm2(self.conv2(y))))
        return x + y


class EgocentricMapLidarClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_sensors: int,
        num_classes: int,
        dropout: float,
        region_sensor_groups: dict[str, list[int]] | None = None,
        region_hidden_dims: dict[str, int] | None = None,
        fusion_hidden_dim: int | None = None,
        sensor_embed_dim: int | None = None,
        decoder_hidden_dim: int | None = None,
        attention_ff_dim: int | None = None,
        attention_heads: int | None = None,
        region_context_dim: int | None = None,
        map_half_extent_cm: float = 3000.0,
        map_cell_size_cm: float = 100.0,
        map_ray_write_samples: int = 8,
        query_ray_samples: int = 8,
        max_range_cm: float = DEFAULT_LIDAR_MAX_RANGE_CM,
    ):
        del region_sensor_groups, region_hidden_dims, attention_ff_dim, attention_heads, region_context_dim
        super().__init__()
        self.model_type = "obstacle_first_ego_map_cnn"
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(max(num_layers, 1))
        self.num_sensors = int(num_sensors)
        self.num_classes = int(num_classes)
        self.pose_dim = int(self.input_dim - 2 * self.num_sensors)
        # Legacy compatibility: older checkpoints may include one trailing feature.
        self.legacy_extra_feature_dim = 0
        if self.pose_dim == POSE_FEATURE_DIM_EGO_MAP + 1:
            self.pose_dim -= 1
            self.legacy_extra_feature_dim = 1
        if self.pose_dim != POSE_FEATURE_DIM_EGO_MAP:
            raise ValueError(
                "obstacle_first_ego_map_cnn expects input_dim=12+2*num_sensors "
                "(or +1 for legacy compatibility); "
                f"got input_dim={self.input_dim} for num_sensors={self.num_sensors}"
            )
        self.zero_bev_channels: tuple[int, ...] = ()

        self.map_half_extent_cm = float(max(map_half_extent_cm, max_range_cm))
        self.map_cell_size_cm = float(max(map_cell_size_cm, 10.0))
        self.max_range_cm = float(max(max_range_cm, 1.0))
        self.map_ray_write_samples = int(max(map_ray_write_samples, 1))
        self.query_ray_samples = int(max(query_ray_samples, 1))
        self.map_size = int(max(8, round((2.0 * self.map_half_extent_cm) / self.map_cell_size_cm)))
        self.map_channels = 5
        self.sensor_embed_dim = (
            max(4, min(16, int(round(self.hidden_dim * 0.25))))
            if sensor_embed_dim is None
            else max(2, int(sensor_embed_dim))
        )
        self.fusion_hidden_dim = (
            max(32, self.hidden_dim)
            if fusion_hidden_dim is None
            else max(8, int(fusion_hidden_dim))
        )
        self.decoder_hidden_dim = (
            max(48, self.hidden_dim * 2)
            if decoder_hidden_dim is None
            else max(8, int(decoder_hidden_dim))
        )

        self.register_buffer(
            "sensor_coords_local",
            torch.tensor(MODEL_SENSOR_COORDS_CM[: self.num_sensors], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sensor_dirs_local",
            torch.tensor(MODEL_SENSOR_DIRS_LOCAL[: self.num_sensors], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "write_fractions",
            torch.linspace(0.08, 0.92, self.map_ray_write_samples, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "query_fractions",
            torch.linspace(0.05, 1.00, self.query_ray_samples, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sensor_ids",
            torch.arange(self.num_sensors, dtype=torch.long),
            persistent=False,
        )

        self.sensor_embedding = nn.Embedding(self.num_sensors, self.sensor_embed_dim)
        self.map_stem = nn.Conv2d(self.map_channels, self.hidden_dim, kernel_size=3, padding=1)
        self.map_stem_norm = nn.GroupNorm(1, self.hidden_dim)
        self.map_blocks = nn.ModuleList(
            [MapResidualBlock(self.hidden_dim, 2 ** min(i, 2), dropout) for i in range(self.num_layers + 1)]
        )
        self.map_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.fusion_hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, self.fusion_hidden_dim),
            nn.GELU(),
        )
        query_in_dim = (
            2 * self.fusion_hidden_dim
            + self.sensor_embed_dim
            + 8
        )
        self.query_decoder = nn.Sequential(
            nn.Linear(query_in_dim, self.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.GELU(),
        )
        self.safe_type_decoder = nn.Sequential(
            nn.Linear(self.decoder_hidden_dim + 3, self.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.GELU(),
        )
        self.obstacle_head = nn.Linear(self.decoder_hidden_dim, 1)
        self.safe_type_head = nn.Linear(self.decoder_hidden_dim, 1)

    def _gather_last_valid(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_idx = torch.arange(values.shape[0], device=values.device)
        last_idx = lengths.to(device=values.device).clamp_min(1) - 1
        return values[batch_idx, last_idx]

    def _xy_to_grid_indices(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gx = torch.floor((xy[..., 0] + self.map_half_extent_cm) / self.map_cell_size_cm).long()
        gy = torch.floor((xy[..., 1] + self.map_half_extent_cm) / self.map_cell_size_cm).long()
        valid = (
            (gx >= 0)
            & (gx < self.map_size)
            & (gy >= 0)
            & (gy < self.map_size)
        )
        return gx + gy * self.map_size, valid

    def _scatter_add_channel(
        self,
        bev: torch.Tensor,
        channel_idx: int,
        xy: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        flat_index, in_bounds = self._xy_to_grid_indices(xy)
        scatter_weights = torch.where(in_bounds, weights, torch.zeros_like(weights))
        scatter_index = torch.where(in_bounds, flat_index, torch.zeros_like(flat_index))
        bev[:, channel_idx, :, :].flatten(1).scatter_add_(1, scatter_index, scatter_weights)

    def _gather_map_samples(
        self,
        feature_map: torch.Tensor,
        xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_index, in_bounds = self._xy_to_grid_indices(xy)
        gather_index = torch.where(in_bounds, flat_index, torch.zeros_like(flat_index))
        flat_map = feature_map.flatten(2)
        gather = flat_map.gather(2, gather_index.unsqueeze(1).expand(-1, feature_map.shape[1], -1))
        gather = gather * in_bounds.unsqueeze(1).to(dtype=gather.dtype)
        return gather, in_bounds

    def _parse_inputs(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del lengths
        pose = x[:, :, : self.pose_dim]
        dist = x[:, :, self.pose_dim : self.pose_dim + self.num_sensors]
        hit = x[:, :, self.pose_dim + self.num_sensors : self.pose_dim + 2 * self.num_sensors]
        pose_origin = pose[:, :, :3]
        pose_basis = pose[:, :, 3:].reshape(x.shape[0], x.shape[1], 3, 3)
        return pose_origin, pose_basis, dist, hit

    def _build_bev(
        self,
        pose_origin: torch.Tensor,
        pose_basis: torch.Tensor,
        dist: torch.Tensor,
        hit: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, time_steps = dist.shape[:2]
        sensor_count = dist.shape[2]
        time_mask = (
            torch.arange(time_steps, device=dist.device).unsqueeze(0) < lengths.to(device=dist.device).unsqueeze(1)
        )
        valid_rays = time_mask.unsqueeze(-1).expand(-1, -1, sensor_count).to(dtype=dist.dtype)

        current_origin = self._gather_last_valid(pose_origin, lengths)
        current_basis = self._gather_last_valid(pose_basis, lengths)
        inv_current_basis = current_basis.transpose(1, 2)
        current_dist = self._gather_last_valid(dist, lengths).clamp(min=0.0, max=self.max_range_cm)
        current_hit = self._gather_last_valid(hit, lengths)

        sensor_coords = self.sensor_coords_local.view(1, 1, sensor_count, 3).to(dtype=dist.dtype)
        sensor_dirs = self.sensor_dirs_local.view(1, 1, sensor_count, 3).to(dtype=dist.dtype)
        pose_basis_exp = pose_basis.unsqueeze(2)

        origin_world = pose_origin.unsqueeze(2) + torch.matmul(pose_basis_exp, sensor_coords.unsqueeze(-1)).squeeze(-1)
        dir_world = torch.matmul(pose_basis_exp, sensor_dirs.unsqueeze(-1)).squeeze(-1)

        rel_origin = origin_world - current_origin[:, None, None, :]
        origin_now = torch.matmul(inv_current_basis[:, None, None, :, :], rel_origin.unsqueeze(-1)).squeeze(-1)
        dir_now = torch.matmul(inv_current_basis[:, None, None, :, :], dir_world.unsqueeze(-1)).squeeze(-1)
        ranges = dist.clamp(min=0.0, max=self.max_range_cm)
        endpoint_now = origin_now + dir_now * ranges.unsqueeze(-1)

        time_idx = torch.arange(time_steps, device=dist.device).unsqueeze(0)
        time_offset = (lengths.to(device=dist.device).unsqueeze(1) - 1 - time_idx).clamp_min(0).to(dtype=dist.dtype)
        history_scale = torch.maximum(
            (lengths.to(device=dist.device).unsqueeze(1) - 1).to(dtype=dist.dtype),
            torch.ones((batch_size, 1), device=dist.device, dtype=dist.dtype),
        )
        recency = torch.exp(-2.0 * time_offset / history_scale) * time_mask.to(dtype=dist.dtype)

        bev = dist.new_zeros((batch_size, self.map_channels, self.map_size, self.map_size))

        write_fracs = self.write_fractions.to(device=dist.device, dtype=dist.dtype).view(1, 1, 1, -1, 1)
        free_points = origin_now.unsqueeze(3) + dir_now.unsqueeze(3) * (ranges.unsqueeze(-1).unsqueeze(-1) * write_fracs)
        free_xy = free_points[..., :2].reshape(batch_size, -1, 2)
        free_weights = (
            valid_rays.unsqueeze(-1).expand(-1, -1, -1, self.map_ray_write_samples) * (1.0 / float(self.map_ray_write_samples))
        ).reshape(batch_size, -1)
        free_recent = (
            recency.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, sensor_count, self.map_ray_write_samples)
            * (1.0 / float(self.map_ray_write_samples))
        ).reshape(batch_size, -1)
        self._scatter_add_channel(bev, 0, free_xy, free_weights)
        self._scatter_add_channel(bev, 1, free_xy, free_recent)

        hit_weights = (valid_rays * hit).reshape(batch_size, -1)
        hit_recent = (recency.unsqueeze(-1) * valid_rays * hit).reshape(batch_size, -1)
        hit_xy = endpoint_now[..., :2].reshape(batch_size, -1, 2)
        hit_heights = (endpoint_now[..., 2] * valid_rays * hit).reshape(batch_size, -1)
        self._scatter_add_channel(bev, 2, hit_xy, hit_weights)
        self._scatter_add_channel(bev, 3, hit_xy, hit_recent)
        self._scatter_add_channel(bev, 4, hit_xy, hit_heights)
        if self.zero_bev_channels:
            for channel_idx in self.zero_bev_channels:
                if 0 <= int(channel_idx) < self.map_channels:
                    bev[:, int(channel_idx), :, :] = 0.0

        return bev, current_dist, current_hit

    def _encode_map(self, bev: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.map_stem_norm(self.map_stem(bev)))
        for block in self.map_blocks:
            h = block(h)
        return self.map_head(h)

    def _decode_queries(
        self,
        encoded_map: torch.Tensor,
        current_dist: torch.Tensor,
        current_hit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = encoded_map.shape[0]
        sensor_coords_xy = self.sensor_coords_local[:, :2].to(device=encoded_map.device, dtype=encoded_map.dtype)
        sensor_dirs_xy = self.sensor_dirs_local[:, :2].to(device=encoded_map.device, dtype=encoded_map.dtype)
        query_fracs = self.query_fractions.to(device=encoded_map.device, dtype=encoded_map.dtype).view(1, 1, -1, 1)

        query_ranges = current_dist.clamp(min=0.0, max=self.max_range_cm)
        query_points = (
            sensor_coords_xy.view(1, self.num_sensors, 1, 2)
            + sensor_dirs_xy.view(1, self.num_sensors, 1, 2) * query_ranges.unsqueeze(-1).unsqueeze(-1) * query_fracs
        )
        flat_query_points = query_points.reshape(batch_size, -1, 2)
        gathered, in_bounds = self._gather_map_samples(encoded_map, flat_query_points)
        gathered = gathered.view(batch_size, encoded_map.shape[1], self.num_sensors, self.query_ray_samples)
        gathered = gathered.permute(0, 2, 3, 1).contiguous()
        in_bounds = in_bounds.view(batch_size, self.num_sensors, self.query_ray_samples)

        valid_count_raw = in_bounds.sum(dim=2, keepdim=True)
        valid_count = valid_count_raw.clamp_min(1)
        query_mean = (gathered * in_bounds.unsqueeze(-1).to(dtype=gathered.dtype)).sum(dim=2) / valid_count.to(
            dtype=gathered.dtype
        )
        query_max = gathered.masked_fill(~in_bounds.unsqueeze(-1), -1e9).max(dim=2).values
        no_valid = valid_count_raw.squeeze(-1) <= 0
        query_max = torch.where(no_valid.unsqueeze(-1), torch.zeros_like(query_max), query_max)

        sensor_embed = self.sensor_embedding(self.sensor_ids.to(device=encoded_map.device)).unsqueeze(0)
        sensor_embed = sensor_embed.expand(batch_size, -1, -1).to(dtype=encoded_map.dtype)
        endpoint_local = (
            self.sensor_coords_local.to(device=encoded_map.device, dtype=encoded_map.dtype).unsqueeze(0)
            + self.sensor_dirs_local.to(device=encoded_map.device, dtype=encoded_map.dtype).unsqueeze(0)
            * query_ranges.unsqueeze(-1)
        )
        decoder_in = torch.cat(
            [
                query_mean,
                query_max,
                sensor_embed,
                (query_ranges / self.max_range_cm).unsqueeze(-1),
                current_hit.unsqueeze(-1),
                endpoint_local,
                self.sensor_dirs_local.to(device=encoded_map.device, dtype=encoded_map.dtype).unsqueeze(0).expand(
                    batch_size, -1, -1
                ),
            ],
            dim=-1,
        )
        decoded = self.query_decoder(decoder_in)
        obstacle_logits = self.obstacle_head(decoded).squeeze(-1)
        safe_hidden = self.safe_type_decoder(
            torch.cat(
                [
                    decoded,
                    (query_ranges / self.max_range_cm).unsqueeze(-1),
                    current_hit.unsqueeze(-1),
                    endpoint_local[:, :, 2:3],
                ],
                dim=-1,
            )
        )
        ground_none_logits = self.safe_type_head(safe_hidden).squeeze(-1)
        return obstacle_logits, ground_none_logits

    def _forward_details(self, x: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        pose_origin, pose_basis, dist, hit = self._parse_inputs(x, lengths)
        bev, current_dist, current_hit = self._build_bev(pose_origin, pose_basis, dist, hit, lengths)
        encoded_map = self._encode_map(bev)
        obstacle_logits, ground_none_logits = self._decode_queries(encoded_map, current_dist, current_hit)
        class_logits = obstacle_first_binary_logits_to_class_logits(obstacle_logits, ground_none_logits)
        return {
            "class_logits": class_logits,
            "obstacle_logits": obstacle_logits,
            "ground_none_logits": ground_none_logits,
            "aux_obstacle_logits": [],
        }

    def forward_with_training_details(self, x: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        return self._forward_details(x, lengths)

    def forward_with_aux(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        details = self._forward_details(x, lengths)
        return details["class_logits"], details["obstacle_logits"], details["ground_none_logits"]

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        class_logits, _, _ = self.forward_with_aux(x, lengths)
        return class_logits


class GRULidarClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_sensors: int,
        num_classes: int,
        dropout: float,
        region_sensor_groups: dict[str, list[int]] | None = None,
        region_hidden_dims: dict[str, int] | None = None,
        fusion_hidden_dim: int | None = None,
        sensor_embed_dim: int | None = None,
        decoder_hidden_dim: int | None = None,
        attention_ff_dim: int | None = None,
        attention_heads: int | None = None,
        region_context_dim: int | None = None,
    ):
        super().__init__()
        self.model_type = "obstacle_first_tsunet"
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_sensors = int(num_sensors)
        self.num_classes = int(num_classes)

        if self.input_dim != 4 + 2 * self.num_sensors:
            raise ValueError(
                f"obstacle_first_tsunet expects input_dim=4+2*num_sensors; got input_dim={self.input_dim} "
                f"for num_sensors={self.num_sensors}"
            )

        self.region_sensor_groups = _normalize_region_sensor_groups(region_sensor_groups, self.num_sensors)
        self.sensor_order_list = _build_sensor_processing_order(self.region_sensor_groups, self.num_sensors)
        self.sensor_restore_list = [0] * self.num_sensors
        for ordered_idx, sensor_id in enumerate(self.sensor_order_list):
            self.sensor_restore_list[int(sensor_id)] = int(ordered_idx)
        self.fusion_hidden_dim = (
            max(48, self.hidden_dim)
            if fusion_hidden_dim is None
            else max(8, int(fusion_hidden_dim))
        )
        self.sensor_embed_dim = (
            max(4, min(16, int(round(self.hidden_dim * 0.25))))
            if sensor_embed_dim is None
            else max(2, int(sensor_embed_dim))
        )
        ff_ref = attention_ff_dim if attention_ff_dim is not None else region_context_dim
        self.attention_ff_dim = (
            max(48, max(self.hidden_dim, self.fusion_hidden_dim))
            if ff_ref is None
            else max(8, int(ff_ref))
        )
        self.decoder_hidden_dim = (
            max(48, max(self.hidden_dim, self.fusion_hidden_dim))
            if decoder_hidden_dim is None
            else max(8, int(decoder_hidden_dim))
        )

        self.register_buffer("sensor_order", torch.tensor(self.sensor_order_list, dtype=torch.long), persistent=False)
        self.register_buffer(
            "sensor_restore",
            torch.tensor(self.sensor_restore_list, dtype=torch.long),
            persistent=False,
        )
        self.sensor_embedding = nn.Embedding(self.num_sensors, self.sensor_embed_dim)
        self.input_feature_dim = 17 + self.sensor_embed_dim
        self.stem = nn.Conv2d(self.input_feature_dim, self.hidden_dim, kernel_size=1)
        self.stem_norm = nn.GroupNorm(1, self.hidden_dim)
        self.unet_depth = max(2, min(4, max(1, self.num_layers)))
        enc_dilations = [2 ** min(i, 3) for i in range(self.unet_depth)]
        self.encoder_blocks = nn.ModuleList(
            [CausalTimeSensorBlock(self.hidden_dim, d, dropout) for d in enc_dilations]
        )
        self.downsample_blocks = nn.ModuleList(
            [TemporalDownsampleBlock(self.hidden_dim) for _ in range(self.unet_depth - 1)]
        )
        bottleneck_dilation = 2 ** min(self.unet_depth + 1, 4)
        self.bottleneck_blocks = nn.ModuleList(
            [
                CausalTimeSensorBlock(self.hidden_dim, bottleneck_dilation, dropout),
                CausalTimeSensorBlock(self.hidden_dim, bottleneck_dilation, dropout),
                CausalTimeSensorBlock(self.hidden_dim, bottleneck_dilation * 2, dropout),
            ]
        )
        self._zero_init_residual_block(self.bottleneck_blocks[-1])
        self.decoder_fuse = nn.ModuleList(
            [TemporalUNetFuse(self.hidden_dim) for _ in range(self.unet_depth - 1)]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                CausalTimeSensorBlock(
                    self.hidden_dim,
                    enc_dilations[self.unet_depth - 2 - i],
                    dropout,
                )
                for i in range(self.unet_depth - 1)
            ]
        )
        self.aux_decoder_levels = min(2, self.unet_depth - 1)
        self.aux_sensor_refiners = nn.ModuleList(
            [SensorRefinementBlock(self.hidden_dim, dropout * 0.5) for _ in range(self.aux_decoder_levels)]
        )
        self.aux_obstacle_heads = nn.ModuleList(
            [nn.Linear(self.hidden_dim, 1) for _ in range(self.aux_decoder_levels)]
        )
        self.backbone_dropout = nn.Dropout2d(dropout)
        self.global_proj = nn.Linear(self.hidden_dim, self.fusion_hidden_dim)
        self.region_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.neighbor_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.final_sensor_refiner = SensorRefinementBlock(self.hidden_dim, dropout)

        decoder_in_dim = (
            self.hidden_dim  # last-time conv features
            + self.hidden_dim  # neighbor latent
            + self.hidden_dim  # region latent
            + self.fusion_hidden_dim  # global latent
            + self.sensor_embed_dim
            + 8  # pose + pose delta
            + 9  # local temporal/spatial range features
        )
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, self.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.GELU(),
        )
        self.obstacle_context_dim = self.hidden_dim + 3
        self.safe_type_decoder = nn.Sequential(
            nn.Linear(self.decoder_hidden_dim + self.obstacle_context_dim, self.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.GELU(),
        )
        self.obstacle_head = nn.Linear(self.decoder_hidden_dim, 1)
        boundary_hidden_dim = max(8, min(64, self.decoder_hidden_dim // 2))
        self.boundary_refine = nn.Sequential(
            nn.Linear(12, boundary_hidden_dim),
            nn.GELU(),
            nn.Linear(boundary_hidden_dim, 1),
        )
        nn.init.zeros_(self.boundary_refine[-1].weight)
        nn.init.zeros_(self.boundary_refine[-1].bias)
        self.safe_type_head = nn.Linear(self.decoder_hidden_dim, 1)
        self.region_attention_heads = {
            name: max(1, min(len(ids), 1)) for name, ids in self.region_sensor_groups.items()
        }

    def _zero_init_residual_block(self, block: CausalTimeSensorBlock) -> None:
        with torch.no_grad():
            block.temporal_conv.weight.zero_()
            if block.temporal_conv.bias is not None:
                block.temporal_conv.bias.zero_()
            block.sensor_conv.weight.zero_()
            if block.sensor_conv.bias is not None:
                block.sensor_conv.bias.zero_()

    def _reorder_sensor_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return values.index_select(dim=2, index=self.sensor_order.to(device=values.device))

    def _restore_sensor_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return values.index_select(dim=1, index=self.sensor_restore.to(device=values.device))

    def _shift_time(self, values: torch.Tensor, steps: int) -> torch.Tensor:
        steps_i = max(1, int(steps))
        prefix = values[:, :1, :].expand(-1, steps_i, -1)
        # Keep the shifted tensor the same length as the input, even when the
        # requested shift is longer than the available history.
        return torch.cat([prefix, values], dim=1)[:, : values.shape[1], :]

    def _shift_sensor(self, values: torch.Tensor, offset: int) -> torch.Tensor:
        if offset < 0:
            steps = -int(offset)
            prefix = values[:, :, :1].expand(-1, -1, steps)
            return torch.cat([prefix, values[:, :, :-steps]], dim=2)
        if offset > 0:
            steps = int(offset)
            suffix = values[:, :, -1:].expand(-1, -1, steps)
            return torch.cat([values[:, :, steps:], suffix], dim=2)
        return values

    def _build_spatiotemporal_input(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pose = x[:, :, :4]
        dist = x[:, :, 4 : 4 + self.num_sensors]
        hit = x[:, :, 4 + self.num_sensors : 4 + 2 * self.num_sensors]

        prev1_dist = self._shift_time(dist, 1)
        prev2_dist = self._shift_time(dist, 2)
        prev4_dist = self._shift_time(dist, 4)
        prev1_hit = self._shift_time(hit, 1)
        prev1_pose = self._shift_time(pose, 1)

        delta1_dist = dist - prev1_dist
        delta2_dist = dist - prev2_dist
        delta4_dist = dist - prev4_dist
        delta1_hit = hit - prev1_hit
        pose_delta = pose - prev1_pose

        dist_ord = self._reorder_sensor_tensor(dist)
        hit_ord = self._reorder_sensor_tensor(hit)
        delta1_ord = self._reorder_sensor_tensor(delta1_dist)
        delta2_ord = self._reorder_sensor_tensor(delta2_dist)
        delta4_ord = self._reorder_sensor_tensor(delta4_dist)

        left_dist = self._shift_sensor(dist_ord, -1)
        right_dist = self._shift_sensor(dist_ord, 1)
        neighbor_mean_dist = 0.5 * (left_dist + right_dist)
        neighbor_delta = dist_ord - neighbor_mean_dist
        left_delta = dist_ord - left_dist
        right_delta = dist_ord - right_dist

        batch_size, time_steps = x.shape[:2]
        pose_expand = pose.unsqueeze(2).expand(-1, -1, self.num_sensors, -1)
        pose_delta_expand = pose_delta.unsqueeze(2).expand(-1, -1, self.num_sensors, -1)
        sensor_embed = self.sensor_embedding(self.sensor_order.to(device=x.device))
        sensor_embed = sensor_embed.view(1, 1, self.num_sensors, self.sensor_embed_dim)
        sensor_embed = sensor_embed.expand(batch_size, time_steps, -1, -1)
        feats = torch.cat(
            [
                dist_ord.unsqueeze(-1),
                hit_ord.unsqueeze(-1),
                delta1_ord.unsqueeze(-1),
                delta1_hit.index_select(2, self.sensor_order.to(device=x.device)).unsqueeze(-1),
                delta2_ord.unsqueeze(-1),
                delta4_ord.unsqueeze(-1),
                neighbor_delta.unsqueeze(-1),
                left_delta.unsqueeze(-1),
                right_delta.unsqueeze(-1),
                pose_expand,
                pose_delta_expand,
                sensor_embed,
            ],
            dim=-1,
        )
        current_local = torch.cat(
            [
                dist_ord[:, -1, :].unsqueeze(-1),
                hit_ord[:, -1, :].unsqueeze(-1),
                delta1_ord[:, -1, :].unsqueeze(-1),
                delta1_hit.index_select(2, self.sensor_order.to(device=x.device))[:, -1, :].unsqueeze(-1),
                delta2_ord[:, -1, :].unsqueeze(-1),
                delta4_ord[:, -1, :].unsqueeze(-1),
                neighbor_delta[:, -1, :].unsqueeze(-1),
                left_delta[:, -1, :].unsqueeze(-1),
                right_delta[:, -1, :].unsqueeze(-1),
            ],
            dim=-1,
        )
        current_pose = pose[:, -1, :]
        current_pose_delta = pose_delta[:, -1, :]
        return feats, current_local, current_pose, current_pose_delta

    def _encode(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        del lengths  # conv backbone uses zero-padded histories directly
        feats, current_local, current_pose, current_pose_delta = self._build_spatiotemporal_input(x)
        conv_in = feats.permute(0, 3, 1, 2).contiguous()
        h = F.gelu(self.stem_norm(self.stem(conv_in)))
        h = self.backbone_dropout(h)
        skips: list[torch.Tensor] = []
        for block_idx, block in enumerate(self.encoder_blocks):
            h = block(h)
            skips.append(h)
            if block_idx < len(self.encoder_blocks) - 1 and h.shape[2] > 1:
                h = self.downsample_blocks[block_idx](h)
        for block in self.bottleneck_blocks:
            h = block(h)
        decoder_last_features: list[torch.Tensor] = []
        for up_idx, skip_idx in enumerate(range(len(skips) - 2, -1, -1)):
            h = self.decoder_fuse[up_idx](h, skips[skip_idx])
            h = self.decoder_blocks[up_idx](h)
            if up_idx < self.aux_decoder_levels:
                decoder_last_features.append(h.permute(0, 2, 3, 1).contiguous()[:, -1, :, :])
        h_bt = h.permute(0, 2, 3, 1).contiguous()
        last_features = h_bt[:, -1, :, :]
        current_local = current_local.to(dtype=last_features.dtype)
        current_pose = current_pose.to(dtype=last_features.dtype)
        current_pose_delta = current_pose_delta.to(dtype=last_features.dtype)
        return last_features, decoder_last_features, current_local, current_pose, current_pose_delta

    def _build_decoder_inputs(
        self,
        last_features: torch.Tensor,
        current_local: torch.Tensor,
        current_pose: torch.Tensor,
        current_pose_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = last_features.shape[0]
        neighbor_latent = 0.5 * (
            self._shift_sensor(last_features.transpose(1, 2), -1).transpose(1, 2)
            + self._shift_sensor(last_features.transpose(1, 2), 1).transpose(1, 2)
        )
        region_latent = last_features.new_zeros(last_features.shape)
        for name, sensor_ids in self.region_sensor_groups.items():
            ordered_positions = [
                self.sensor_order_list.index(int(sensor_id))
                for sensor_id in sensor_ids
                if int(sensor_id) in self.sensor_order_list
            ]
            if not ordered_positions:
                continue
            region_tokens = last_features[:, ordered_positions, :]
            region_mean = region_tokens.mean(dim=1, keepdim=True)
            region_latent[:, ordered_positions, :] = region_mean.expand(-1, len(ordered_positions), -1)

        global_latent = F.gelu(self.global_proj(last_features.mean(dim=1)))
        global_expand = global_latent.unsqueeze(1).expand(-1, self.num_sensors, -1)
        sensor_embed = self.sensor_embedding(self.sensor_order.to(device=last_features.device)).unsqueeze(0)
        sensor_embed = sensor_embed.expand(batch_size, -1, -1).to(dtype=last_features.dtype)
        pose_expand = current_pose.unsqueeze(1).expand(-1, self.num_sensors, -1)
        pose_delta_expand = current_pose_delta.unsqueeze(1).expand(-1, self.num_sensors, -1)

        decoder_in = torch.cat(
            [
                last_features,
                self.neighbor_proj(neighbor_latent),
                self.region_proj(region_latent),
                global_expand,
                sensor_embed,
                pose_expand,
                pose_delta_expand,
                current_local,
            ],
            dim=-1,
        )
        obstacle_context = torch.cat(
            [
                self.neighbor_proj(neighbor_latent),
                current_local[:, :, -3:],
            ],
            dim=-1,
        )
        return decoder_in, obstacle_context

    def _forward_details(self, x: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        last_features, decoder_last_features, current_local, current_pose, current_pose_delta = self._encode(x, lengths)
        refined_last_features = self.final_sensor_refiner(last_features)
        decoder_in, obstacle_context = self._build_decoder_inputs(
            refined_last_features,
            current_local,
            current_pose,
            current_pose_delta,
        )
        decoder_out = self.decoder(decoder_in)
        base_obstacle_logits_ordered = self.obstacle_head(decoder_out).squeeze(-1)
        ordered_logits_bt = base_obstacle_logits_ordered.unsqueeze(1)
        left_neighbor_logits = self._shift_sensor(ordered_logits_bt, -1).squeeze(1)
        right_neighbor_logits = self._shift_sensor(ordered_logits_bt, 1).squeeze(1)
        boundary_inputs = torch.cat(
            [
                base_obstacle_logits_ordered.unsqueeze(-1),
                left_neighbor_logits.unsqueeze(-1),
                right_neighbor_logits.unsqueeze(-1),
                current_local,
            ],
            dim=-1,
        )
        obstacle_logits_ordered = base_obstacle_logits_ordered + self.boundary_refine(boundary_inputs).squeeze(-1)
        safe_type_hidden = self.safe_type_decoder(torch.cat([decoder_out, obstacle_context], dim=-1))
        ground_none_logits_ordered = self.safe_type_head(safe_type_hidden).squeeze(-1)
        aux_obstacle_logits: list[torch.Tensor] = []
        for head_idx, features in enumerate(decoder_last_features[: self.aux_decoder_levels]):
            aux_features = self.aux_sensor_refiners[head_idx](features)
            aux_logits_ordered = self.aux_obstacle_heads[head_idx](aux_features).squeeze(-1)
            aux_obstacle_logits.append(self._restore_sensor_tensor(aux_logits_ordered))
        obstacle_logits = self._restore_sensor_tensor(obstacle_logits_ordered)
        ground_none_logits = self._restore_sensor_tensor(ground_none_logits_ordered)
        class_logits = obstacle_first_binary_logits_to_class_logits(obstacle_logits, ground_none_logits)
        return {
            "class_logits": class_logits,
            "obstacle_logits": obstacle_logits,
            "ground_none_logits": ground_none_logits,
            "aux_obstacle_logits": aux_obstacle_logits,
        }

    def forward_with_training_details(self, x: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        return self._forward_details(x, lengths)

    def forward_with_aux(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        details = self._forward_details(x, lengths)
        class_logits = details["class_logits"]
        obstacle_logits = details["obstacle_logits"]
        ground_none_logits = details["ground_none_logits"]
        return class_logits, obstacle_logits, ground_none_logits

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        class_logits, _, _ = self.forward_with_aux(x, lengths)
        return class_logits


class LegacyCausalTSCNNClassifier(GRULidarClassifier):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_sensors: int,
        num_classes: int,
        dropout: float,
        region_sensor_groups: dict[str, list[int]] | None = None,
        region_hidden_dims: dict[str, int] | None = None,
        fusion_hidden_dim: int | None = None,
        sensor_embed_dim: int | None = None,
        decoder_hidden_dim: int | None = None,
        attention_ff_dim: int | None = None,
        attention_heads: int | None = None,
        region_context_dim: int | None = None,
    ):
        del region_hidden_dims, attention_heads
        nn.Module.__init__(self)
        self.model_type = "obstacle_first_causal_tscnn"
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_sensors = int(num_sensors)
        self.num_classes = int(num_classes)

        if self.input_dim != 4 + 2 * self.num_sensors:
            raise ValueError(
                f"obstacle_first_causal_tscnn expects input_dim=4+2*num_sensors; got input_dim={self.input_dim} "
                f"for num_sensors={self.num_sensors}"
            )

        self.region_sensor_groups = _normalize_region_sensor_groups(region_sensor_groups, self.num_sensors)
        self.sensor_order_list = _build_sensor_processing_order(self.region_sensor_groups, self.num_sensors)
        self.sensor_restore_list = [0] * self.num_sensors
        for ordered_idx, sensor_id in enumerate(self.sensor_order_list):
            self.sensor_restore_list[int(sensor_id)] = int(ordered_idx)
        self.fusion_hidden_dim = (
            max(48, self.hidden_dim)
            if fusion_hidden_dim is None
            else max(8, int(fusion_hidden_dim))
        )
        self.sensor_embed_dim = (
            max(4, min(16, int(round(self.hidden_dim * 0.25))))
            if sensor_embed_dim is None
            else max(2, int(sensor_embed_dim))
        )
        ff_ref = attention_ff_dim if attention_ff_dim is not None else region_context_dim
        self.attention_ff_dim = (
            max(48, max(self.hidden_dim, self.fusion_hidden_dim))
            if ff_ref is None
            else max(8, int(ff_ref))
        )
        self.decoder_hidden_dim = (
            max(48, max(self.hidden_dim, self.fusion_hidden_dim))
            if decoder_hidden_dim is None
            else max(8, int(decoder_hidden_dim))
        )

        self.register_buffer("sensor_order", torch.tensor(self.sensor_order_list, dtype=torch.long), persistent=False)
        self.register_buffer(
            "sensor_restore",
            torch.tensor(self.sensor_restore_list, dtype=torch.long),
            persistent=False,
        )
        self.sensor_embedding = nn.Embedding(self.num_sensors, self.sensor_embed_dim)
        self.input_feature_dim = 17 + self.sensor_embed_dim
        self.stem = nn.Conv2d(self.input_feature_dim, self.hidden_dim, kernel_size=1)
        self.stem_norm = nn.GroupNorm(1, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [CausalTimeSensorBlock(self.hidden_dim, 2 ** min(i, 3), dropout) for i in range(max(1, self.num_layers))]
        )
        self.backbone_depth = len(self.blocks)
        self.backbone_dropout = nn.Dropout2d(dropout)
        self.global_proj = nn.Linear(self.hidden_dim, self.fusion_hidden_dim)
        self.region_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.neighbor_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        decoder_in_dim = (
            self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim
            + self.fusion_hidden_dim
            + self.sensor_embed_dim
            + 8
            + 9
        )
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, self.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.GELU(),
        )
        self.obstacle_context_dim = self.hidden_dim + 3
        self.safe_type_decoder = nn.Sequential(
            nn.Linear(self.decoder_hidden_dim + self.obstacle_context_dim, self.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.GELU(),
        )
        self.obstacle_head = nn.Linear(self.decoder_hidden_dim, 1)
        self.safe_type_head = nn.Linear(self.decoder_hidden_dim, 1)
        self.region_attention_heads = {
            name: max(1, min(len(ids), 1)) for name, ids in self.region_sensor_groups.items()
        }

    def _encode(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del lengths
        feats, current_local, current_pose, current_pose_delta = self._build_spatiotemporal_input(x)
        conv_in = feats.permute(0, 3, 1, 2).contiguous()
        h = F.gelu(self.stem_norm(self.stem(conv_in)))
        h = self.backbone_dropout(h)
        for block in self.blocks:
            h = block(h)
        h_bt = h.permute(0, 2, 3, 1).contiguous()
        last_features = h_bt[:, -1, :, :]
        current_local = current_local.to(dtype=last_features.dtype)
        current_pose = current_pose.to(dtype=last_features.dtype)
        current_pose_delta = current_pose_delta.to(dtype=last_features.dtype)
        return last_features, current_local, current_pose, current_pose_delta

    def _forward_details(self, x: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        last_features, current_local, current_pose, current_pose_delta = self._encode(x, lengths)
        decoder_in, obstacle_context = self._build_decoder_inputs(
            last_features,
            current_local,
            current_pose,
            current_pose_delta,
        )
        decoder_out = self.decoder(decoder_in)
        obstacle_logits_ordered = self.obstacle_head(decoder_out).squeeze(-1)
        safe_type_hidden = self.safe_type_decoder(torch.cat([decoder_out, obstacle_context], dim=-1))
        ground_none_logits_ordered = self.safe_type_head(safe_type_hidden).squeeze(-1)
        obstacle_logits = self._restore_sensor_tensor(obstacle_logits_ordered)
        ground_none_logits = self._restore_sensor_tensor(ground_none_logits_ordered)
        class_logits = obstacle_first_binary_logits_to_class_logits(obstacle_logits, ground_none_logits)
        return {
            "class_logits": class_logits,
            "obstacle_logits": obstacle_logits,
            "ground_none_logits": ground_none_logits,
            "aux_obstacle_logits": [],
        }


def infer_checkpoint_model_type(
    model_config: dict,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> str:
    cfg_type = str(model_config.get("model_type", "")).strip()
    if state_dict:
        state_keys = list(state_dict.keys())
        if any(key.startswith("map_stem.") or key.startswith("map_blocks.") for key in state_keys):
            return "obstacle_first_ego_map_cnn"
        if any(key.startswith("encoder_blocks.") or key.startswith("downsample_blocks.") for key in state_keys):
            return "obstacle_first_tsunet"
        if any(key.startswith("blocks.") for key in state_keys):
            return "obstacle_first_causal_tscnn"
    if cfg_type in {
        "obstacle_first_ego_map_cnn",
        "obstacle_first_tsunet",
        "obstacle_first_causal_tscnn",
    }:
        return cfg_type
    raise ValueError(
        f"Checkpoint model_type={cfg_type!r} is not supported by this build. "
        "Supported types: obstacle_first_causal_tscnn, obstacle_first_ego_map_cnn, obstacle_first_tsunet."
    )


def build_lidar_model_from_config(
    model_config: dict,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> nn.Module:
    resolved_model_type = infer_checkpoint_model_type(model_config, state_dict)
    common_kwargs = dict(
        input_dim=int(model_config["input_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        num_sensors=int(model_config["num_sensors"]),
        num_classes=int(model_config["num_classes"]),
        dropout=float(model_config["dropout"]),
        region_sensor_groups=model_config.get("region_sensor_groups"),
        fusion_hidden_dim=model_config.get("fusion_hidden_dim"),
        sensor_embed_dim=model_config.get("sensor_embed_dim"),
        decoder_hidden_dim=model_config.get("decoder_hidden_dim"),
        attention_ff_dim=model_config.get("attention_ff_dim", model_config.get("region_context_dim")),
        attention_heads=model_config.get("attention_heads"),
    )
    if resolved_model_type == "obstacle_first_tsunet":
        return GRULidarClassifier(**common_kwargs)
    if resolved_model_type == "obstacle_first_causal_tscnn":
        return LegacyCausalTSCNNClassifier(**common_kwargs)
    if resolved_model_type == "obstacle_first_ego_map_cnn":
        return EgocentricMapLidarClassifier(
            **common_kwargs,
            map_half_extent_cm=float(model_config.get("map_half_extent_cm", 3000.0)),
            map_cell_size_cm=float(model_config.get("map_cell_size_cm", 100.0)),
            map_ray_write_samples=int(model_config.get("map_ray_write_samples", 8)),
            query_ray_samples=int(model_config.get("query_ray_samples", 8)),
            max_range_cm=float(model_config.get("no_hit_range_cm", DEFAULT_LIDAR_MAX_RANGE_CM)),
        )
    raise ValueError(f"Unsupported resolved model_type={resolved_model_type!r}")


def obstacle_first_binary_logits_to_class_logits(
    obstacle_logits: torch.Tensor,
    ground_none_logits: torch.Tensor,
) -> torch.Tensor:
    # Compose score-like logits from the two binary heads without giving obstacle
    # an automatic advantage at zero-initialization. When both heads are near 0,
    # all three class scores tie instead of defaulting to "obstacle".
    obstacle_score = obstacle_logits
    ground_score = -obstacle_logits + ground_none_logits
    none_score = -obstacle_logits - ground_none_logits
    return torch.stack([ground_score, obstacle_score, none_score], dim=-1)


def predict_eval_logits(
    model: nn.Module,
    x: torch.Tensor,
    lengths: torch.Tensor,
    obstacle_logit_bias: float = 0.0,
) -> torch.Tensor:
    if hasattr(model, "forward_with_aux"):
        out = model.forward_with_aux(x, lengths)
        if isinstance(out, tuple) and len(out) == 3:
            class_logits, obstacle_logits, ground_none_logits = out
            bias = float(obstacle_logit_bias)
            if abs(bias) > 1e-12:
                class_logits = obstacle_first_binary_logits_to_class_logits(
                    obstacle_logits + bias,
                    ground_none_logits,
                )
            return class_logits
    return model(x, lengths)


def predict_eval_obstacle_logits(
    model: nn.Module,
    x: torch.Tensor,
    lengths: torch.Tensor,
    obstacle_logit_bias: float = 0.0,
) -> torch.Tensor:
    if hasattr(model, "forward_with_aux"):
        out = model.forward_with_aux(x, lengths)
        if isinstance(out, tuple) and len(out) == 3:
            _, obstacle_logits, _ = out
            bias = float(obstacle_logit_bias)
            if abs(bias) > 1e-12:
                obstacle_logits = obstacle_logits + bias
            return obstacle_logits
    logits = model(x, lengths)
    if logits.ndim == 3 and logits.shape[-1] >= 2:
        return logits[..., 1] - logits[..., 0]
    raise ValueError("Model does not expose obstacle logits for binary evaluation.")


class GRULidarInferencer:
    def __init__(
        self,
        model: nn.Module,
        stats: NormStats,
        device: torch.device,
        num_sensors: int,
        input_dim: int,
        max_history: int = 64,
        binary_obstacle_only: bool = False,
        no_hit_range_cm: float = DEFAULT_LIDAR_MAX_RANGE_CM,
    ):
        self.model = model.eval()
        self.stats = stats
        self.device = device
        self.num_sensors = int(num_sensors)
        self.input_dim = int(input_dim)
        self.max_history = int(max_history)
        self.binary_obstacle_only = bool(binary_obstacle_only)
        self.no_hit_range_cm = float(no_hit_range_cm)
        self.pose_dim = int(self.stats.pose_mean.shape[0])
        self.base_input_dim = int(self.pose_dim + 2 * self.num_sensors)

    def featurize_timestep(
        self,
        pose_values: np.ndarray,
        lidar_cm: np.ndarray,
        basis: np.ndarray | None = None,
    ) -> np.ndarray:
        return featurize_timestep(
            pose_values,
            lidar_cm,
            self.stats,
            basis=basis,
            no_hit_range_cm=self.no_hit_range_cm,
        )

    def _prepare_history_tensor(self, feature_history: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if feature_history.ndim != 2:
            raise ValueError("feature_history must be shape [T, F]")
        if feature_history.shape[0] < 1:
            raise ValueError("feature_history must have at least 1 timestep")

        feat = feature_history.astype(np.float32, copy=False)
        # Legacy compatibility: older checkpoints may expect one trailing feature.
        if feat.shape[1] == self.base_input_dim and self.input_dim == self.base_input_dim + 1:
            feat = np.concatenate([feat, np.zeros((feat.shape[0], 1), dtype=np.float32)], axis=1)
        elif feat.shape[1] == self.base_input_dim + 1 and self.input_dim == self.base_input_dim:
            feat = feat[:, : self.base_input_dim]
        elif feat.shape[1] != self.input_dim:
            raise ValueError(f"feature_history feature dim {feat.shape[1]} != expected {self.input_dim}")

        if self.max_history > 0 and feat.shape[0] > self.max_history:
            feat = feat[-self.max_history :]

        x = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=self.device)
        return x, lengths

    def predict_current_from_feature_history(self, feature_history: np.ndarray) -> np.ndarray:
        x, lengths = self._prepare_history_tensor(feature_history)
        with torch.no_grad():
            logits = self.model(x, lengths)
            pred = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.int64)
        if pred.shape[0] != self.num_sensors:
            raise RuntimeError(f"predicted {pred.shape[0]} sensors, expected {self.num_sensors}")
        return pred

    def predict_current_obstacle_mask_from_feature_history(self, feature_history: np.ndarray) -> np.ndarray:
        return self.predict_current_obstacle_mask_from_feature_history_with_bias(feature_history, obstacle_logit_bias=0.0)

    def predict_current_obstacle_logits_from_feature_history(self, feature_history: np.ndarray) -> np.ndarray:
        x, lengths = self._prepare_history_tensor(feature_history)
        with torch.no_grad():
            obstacle_logits = predict_eval_obstacle_logits(self.model, x, lengths)
            pred = obstacle_logits.squeeze(0).cpu().numpy().astype(np.float32)
        if pred.shape[0] != self.num_sensors:
            raise RuntimeError(f"predicted {pred.shape[0]} sensors, expected {self.num_sensors}")
        return pred

    def predict_current_obstacle_mask_from_feature_history_with_bias(
        self,
        feature_history: np.ndarray,
        obstacle_logit_bias: float = 0.0,
    ) -> np.ndarray:
        x, lengths = self._prepare_history_tensor(feature_history)
        with torch.no_grad():
            obstacle_logits = predict_eval_obstacle_logits(self.model, x, lengths, obstacle_logit_bias=obstacle_logit_bias)
            pred = (obstacle_logits > 0).squeeze(0).cpu().numpy().astype(bool)
        if pred.shape[0] != self.num_sensors:
            raise RuntimeError(f"predicted {pred.shape[0]} sensors, expected {self.num_sensors}")
        return pred

    def predict_current_from_history(self, pose_history: np.ndarray, lidar_cm_history: np.ndarray) -> np.ndarray:
        if pose_history.ndim != 2 or pose_history.shape[1] not in {3, 4, 12}:
            raise ValueError("pose_history must be shape [T,3], [T,4], or [T,12]")
        if lidar_cm_history.ndim != 2 or lidar_cm_history.shape[1] != self.num_sensors:
            raise ValueError(f"lidar_cm_history must be shape [T,{self.num_sensors}]")
        if pose_history.shape[0] != lidar_cm_history.shape[0]:
            raise ValueError("pose_history and lidar_cm_history must have matching T")
        if pose_history.shape[0] < 1:
            raise ValueError("history must have at least 1 timestep")

        feats = np.stack(
            [self.featurize_timestep(pose_history[i], lidar_cm_history[i]) for i in range(pose_history.shape[0])],
            axis=0,
        )
        return self.predict_current_from_feature_history(feats)

    def predict_next_from_feature_history(self, feature_history: np.ndarray) -> np.ndarray:
        # Backward-compatible alias retained for callers that still use the old name.
        return self.predict_current_from_feature_history(feature_history)

    def predict_next_from_history(self, pose_history: np.ndarray, lidar_cm_history: np.ndarray) -> np.ndarray:
        # Backward-compatible alias retained for callers that still use the old name.
        return self.predict_current_from_history(pose_history, lidar_cm_history)


def load_gru_lidar_inferencer(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    max_history: int | None = None,
) -> GRULidarInferencer:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    if device is None:
        device_t, device_kind = select_runtime_device("auto")
    elif isinstance(device, str) and device.strip().lower() in {"auto", "cpu", "cuda", "mps", "xla"}:
        device_t, device_kind = select_runtime_device(device)
    else:
        device_t = torch.device(device)
        device_kind = str(device_t.type)

    ckpt = load_checkpoint_for_device(checkpoint_path, device_t, device_kind)
    cfg = dict(ckpt["model_config"])
    cfg["model_type"] = infer_checkpoint_model_type(cfg, ckpt.get("model_state_dict"))
    model = build_lidar_model_from_config(cfg, ckpt.get("model_state_dict")).to(device_t)
    load_model_state_with_compat(model, ckpt["model_state_dict"])
    model.eval()

    norm = ckpt["norm"]
    stats = NormStats(
        pose_mean=np.asarray(norm["pose_mean"], dtype=np.float32),
        pose_std=np.asarray(norm["pose_std"], dtype=np.float32),
        dist_mean=float(norm["dist_mean"]),
        dist_std=float(norm["dist_std"]),
    )
    if max_history is None:
        max_history = int(ckpt.get("meta", {}).get("max_history", 64))
    return GRULidarInferencer(
        model=model,
        stats=stats,
        device=device_t,
        num_sensors=int(cfg["num_sensors"]),
        input_dim=int(cfg["input_dim"]),
        max_history=int(max_history),
        binary_obstacle_only=bool(ckpt.get("meta", {}).get("binary_obstacle_only", False)),
        no_hit_range_cm=float(ckpt.get("meta", {}).get("no_hit_range_cm", DEFAULT_LIDAR_MAX_RANGE_CM)),
    )


def focal_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    class_weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    # Standard multi-class focal loss applied per lidar ray.
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    pt = log_pt.exp()
    focal = (1.0 - pt).clamp(min=0.0).pow(float(max(gamma, 0.0)))

    nll = -log_pt
    if class_weights is not None:
        sample_w = class_weights.to(logits.device).gather(0, targets.reshape(-1)).reshape_as(targets)
        nll = nll * sample_w
    loss = focal * nll
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported focal loss reduction: {reduction}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    device_kind: str,
    num_classes: int,
    epoch_idx: int,
    total_epochs: int,
    phase_name: str,
    log_every_batches: int,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    grad_clip_norm: float = 0.0,
    focal_gamma: float = 2.0,
    obstacle_class_focus_weight: float = 1.0,
    obstacle_aux_weight: float = 1.0,
    obstacle_pos_weight: float = 1.0,
    hard_example_fraction: float = 0.25,
    obstacle_primary_loss_weight: float = 2.0,
    safe_type_loss_weight: float = 0.2,
    binary_obstacle_only: bool = False,
    use_amp: bool = False,
    grad_scaler=None,
) -> tuple[float, float]:
    def fmt_acc(correct: int, total: int) -> str:
        if total <= 0:
            return "n/a"
        return f"{(correct / total):.3f}"

    is_train = optimizer is not None
    model.train(is_train)
    hard_example_fraction = float(np.clip(hard_example_fraction, 0.0, 1.0))

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    seen_sequences = 0
    class_totals = np.zeros((num_classes,), dtype=np.int64)
    class_correct = np.zeros((num_classes,), dtype=np.int64)

    batch_count = len(loader)
    phase_t0 = perf_counter()
    non_blocking = device_kind == "cuda"
    amp_enabled = bool(use_amp and device_kind == "cuda")
    for batch_idx, (x, lengths, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
            if amp_enabled
            else nullcontext()
        )
        with amp_context:
            details = model.forward_with_training_details(x, lengths)
            logits = details["class_logits"]
            obstacle_logits = details["obstacle_logits"]
            ground_none_logits = details["ground_none_logits"]
            aux_obstacle_logits = details["aux_obstacle_logits"]
            obstacle_targets = (y == 1).float()
            effective_obstacle_pos_weight = 1.0 if binary_obstacle_only else float(obstacle_pos_weight)
            pos_weight_t = torch.tensor(
                effective_obstacle_pos_weight,
                device=obstacle_logits.device,
                dtype=obstacle_logits.dtype,
            )
            if binary_obstacle_only:
                obstacle_loss = F.binary_cross_entropy_with_logits(
                    obstacle_logits,
                    obstacle_targets,
                    reduction="mean",
                )
            else:
                obstacle_loss = balanced_hard_obstacle_bce_loss(
                    obstacle_logits,
                    obstacle_targets,
                    pos_weight=pos_weight_t,
                    hard_fraction=hard_example_fraction,
                )
            aux_loss_terms: list[torch.Tensor] = []
            for aux_logits in aux_obstacle_logits:
                if binary_obstacle_only:
                    aux_loss_terms.append(
                        F.binary_cross_entropy_with_logits(
                            aux_logits,
                            obstacle_targets,
                            reduction="mean",
                        )
                    )
                else:
                    aux_loss_terms.append(
                        balanced_hard_obstacle_bce_loss(
                            aux_logits,
                            obstacle_targets,
                            pos_weight=pos_weight_t,
                            hard_fraction=hard_example_fraction,
                        )
                    )
            if aux_loss_terms:
                obstacle_aux_loss = torch.stack(aux_loss_terms).mean()
            else:
                obstacle_aux_loss = obstacle_loss.new_zeros(())
            safe_mask = y != 1
            if safe_mask.any():
                ground_none_targets = (y[safe_mask] == 0).float()
                safe_type_loss = F.binary_cross_entropy_with_logits(
                    ground_none_logits[safe_mask],
                    ground_none_targets,
                    reduction="mean",
                )
            else:
                safe_type_loss = obstacle_loss.new_zeros(())
            if binary_obstacle_only:
                loss = obstacle_loss + (float(obstacle_aux_weight) * obstacle_aux_loss)
            else:
                loss = (
                    (float(obstacle_primary_loss_weight) * obstacle_loss)
                    + (float(obstacle_aux_weight) * obstacle_aux_loss)
                    + (float(safe_type_loss_weight) * safe_type_loss)
                )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler_enabled = bool(grad_scaler is not None and getattr(grad_scaler, "is_enabled", lambda: False)())
            if scaler_enabled:
                grad_scaler.scale(loss).backward()
                if grad_clip_norm > 0.0:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer_step_for_device(optimizer, device_kind)

        if binary_obstacle_only:
            binary_targets = (y == 1).long()
            preds = (obstacle_logits > 0).long()
            total_correct += int((preds == binary_targets).sum().item())
            total_count += int(binary_targets.numel())
            y_flat = binary_targets.reshape(-1).detach().cpu().numpy()
            p_flat = preds.reshape(-1).detach().cpu().numpy()
        else:
            preds = torch.argmax(logits, dim=-1)
            total_correct += int((preds == y).sum().item())
            total_count += int(y.numel())
            y_flat = y.reshape(-1).detach().cpu().numpy()
            p_flat = preds.reshape(-1).detach().cpu().numpy()

        batch_sequences = int(y.shape[0])
        total_loss += float(loss.item()) * float(batch_sequences)
        seen_sequences += batch_sequences

        for cls_id in range(num_classes):
            cls_mask = y_flat == cls_id
            cls_count = int(np.sum(cls_mask))
            if cls_count > 0:
                class_totals[cls_id] += cls_count
                class_correct[cls_id] += int(np.sum(p_flat[cls_mask] == cls_id))

        if log_every_batches > 0 and (batch_idx % log_every_batches == 0 or batch_idx == batch_count):
            running_loss = total_loss / max(seen_sequences, 1)
            running_acc = total_correct / max(total_count, 1)
            if binary_obstacle_only:
                non_obstacle_acc = fmt_acc(int(class_correct[0]), int(class_totals[0]))
                obstacle_acc = fmt_acc(int(class_correct[1]), int(class_totals[1]))
                log(
                    f"{phase_name} epoch {epoch_idx:03d}/{total_epochs:03d} "
                    f"batch {batch_idx:04d}/{batch_count:04d} "
                    f"loss={running_loss:.5f} acc={running_acc:.4f} "
                    f"binary_acc(non/o)={non_obstacle_acc}/{obstacle_acc}"
                )
            else:
                obstacle_acc = fmt_acc(int(class_correct[1]), int(class_totals[1]))
                ground_acc = fmt_acc(int(class_correct[0]), int(class_totals[0]))
                nothing_acc = fmt_acc(int(class_correct[2]), int(class_totals[2]))
                log(
                    f"{phase_name} epoch {epoch_idx:03d}/{total_epochs:03d} "
                    f"batch {batch_idx:04d}/{batch_count:04d} "
                    f"loss={running_loss:.5f} acc={running_acc:.4f} "
                    f"class_acc(o/g/n)={obstacle_acc}/{ground_acc}/{nothing_acc}"
                )

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = total_correct / max(total_count, 1)
    phase_s = perf_counter() - phase_t0
    log(
        f"{phase_name} epoch {epoch_idx:03d}/{total_epochs:03d} complete "
        f"loss={avg_loss:.5f} acc={acc:.4f} time={phase_s:.2f}s"
    )
    return avg_loss, acc


def evaluate_detailed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    device_kind: str,
    num_classes: int,
    obstacle_logit_bias: float = 0.0,
    binary_obstacle_only: bool = False,
) -> dict:
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for x, lengths, y in loader:
            x = x.to(device)
            y = y.to(device)
            if binary_obstacle_only:
                obstacle_logits = predict_eval_obstacle_logits(
                    model,
                    x,
                    lengths,
                    obstacle_logit_bias=obstacle_logit_bias,
                )
                preds = (obstacle_logits > 0).long()
                y_eval = (y == 1).long()
            else:
                logits = predict_eval_logits(model, x, lengths, obstacle_logit_bias=obstacle_logit_bias)
                preds = torch.argmax(logits, dim=-1)
                y_eval = y

            y_flat = y_eval.reshape(-1).cpu().numpy()
            p_flat = preds.reshape(-1).cpu().numpy()
            for yt, yp in zip(y_flat, p_flat):
                confusion[int(yt), int(yp)] += 1
            if device_kind == "xla":
                import torch_xla.core.xla_model as xm

                xm.mark_step()

    per_class_recall = []
    for c in range(num_classes):
        denom = int(confusion[c, :].sum())
        per_class_recall.append(float(confusion[c, c]) / max(denom, 1))

    total = int(confusion.sum())
    overall_accuracy = float(np.trace(confusion)) / max(total, 1)

    return {
        "confusion_matrix": confusion.tolist(),
        "per_class_recall": per_class_recall,
        "overall_accuracy": overall_accuracy,
    }


def build_obstacle_logit_bias_sweep(
    min_bias: float,
    max_bias: float,
    step: float,
) -> list[float]:
    min_v = float(min_bias)
    max_v = float(max_bias)
    if max_v < min_v:
        min_v, max_v = max_v, min_v
    step_v = abs(float(step))
    if step_v <= 0.0:
        return [0.0]

    values: list[float] = []
    current = min_v
    while current <= max_v + (step_v * 0.5):
        values.append(round(current, 6))
        current += step_v
    if not values:
        values = [0.0]
    return values


def run_obstacle_logit_bias_sweep(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    device_kind: str,
    num_classes: int,
    biases: list[float],
    binary_obstacle_only: bool = False,
) -> dict:
    entries: list[dict] = []
    best_accuracy: dict | None = None
    best_obstacle_recall: dict | None = None

    for bias in biases:
        report = evaluate_detailed(
            model,
            loader,
            device,
            device_kind,
            num_classes,
            obstacle_logit_bias=float(bias),
            binary_obstacle_only=binary_obstacle_only,
        )
        entry = {
            "obstacle_logit_bias": float(bias),
            "overall_accuracy": float(report["overall_accuracy"]),
            "per_class_recall": report["per_class_recall"],
            "confusion_matrix": report["confusion_matrix"],
        }
        entries.append(entry)
        if best_accuracy is None or entry["overall_accuracy"] > best_accuracy["overall_accuracy"]:
            best_accuracy = entry
        if (
            best_obstacle_recall is None
            or float(entry["per_class_recall"][1]) > float(best_obstacle_recall["per_class_recall"][1])
        ):
            best_obstacle_recall = entry

    return {
        "entries": entries,
        "best_by_accuracy": best_accuracy,
        "best_by_obstacle_recall": best_obstacle_recall,
    }


def compute_class_weights(class_counts: np.ndarray) -> np.ndarray:
    counts = np.clip(class_counts.astype(np.float64), 1.0, None)
    # Tempered class balancing:
    # Use inverse-sqrt frequency (less aggressive than inverse frequency) and
    # cap the maximum weight so rare classes do not dominate loss.
    weights = 1.0 / np.sqrt(counts)
    weights /= np.mean(weights)
    max_weight = 1.8
    weights = np.clip(weights, 1e-8, max_weight)
    weights /= np.mean(weights)
    return weights.astype(np.float32)


def compute_obstacle_ground_pos_weight(class_counts: np.ndarray, max_weight: float = 12.0) -> float:
    counts = np.asarray(class_counts, dtype=np.float64)
    obstacle = max(float(counts[1]), 1.0)
    non_obstacle = max(float(counts[0]) + float(counts[2]), 1.0)
    weight = non_obstacle / obstacle
    return float(np.clip(weight, 1.0, max_weight))


def _topk_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    frac = float(np.clip(fraction, 0.0, 1.0))
    if frac <= 0.0 or frac >= 1.0:
        return values.mean()
    k = max(1, int(np.ceil(values.numel() * frac)))
    return torch.topk(values.reshape(-1), k=k).values.mean()


def balanced_hard_obstacle_bce_loss(
    obstacle_logits: torch.Tensor,
    obstacle_targets: torch.Tensor,
    pos_weight: torch.Tensor | None,
    hard_fraction: float,
) -> torch.Tensor:
    bce_kwargs = {"reduction": "none"}
    if pos_weight is not None:
        bce_kwargs["pos_weight"] = pos_weight
    per_ray = F.binary_cross_entropy_with_logits(obstacle_logits, obstacle_targets, **bce_kwargs)
    pos_mask = obstacle_targets > 0.5
    neg_mask = ~pos_mask
    pos_losses = per_ray[pos_mask]
    neg_losses = per_ray[neg_mask]
    if pos_losses.numel() == 0:
        return _topk_mean(neg_losses, hard_fraction)

    pos_term = _topk_mean(pos_losses, hard_fraction)
    if neg_losses.numel() == 0:
        return pos_term

    pos_k = max(1, int(np.ceil(pos_losses.numel() * max(float(hard_fraction), 1e-6))))
    neg_k = min(neg_losses.numel(), pos_k)
    neg_term = torch.topk(neg_losses.reshape(-1), k=neg_k).values.mean()
    return 0.5 * (pos_term + neg_term)


def compute_obstacle_oversample_weights(
    dataset: SequencePieceDataset,
    target_fraction: float,
) -> tuple[np.ndarray, dict]:
    total = len(dataset)
    if total <= 0:
        return np.zeros((0,), dtype=np.float64), {
            "enabled": False,
            "num_samples": 0,
            "obstacle_samples": 0,
            "non_obstacle_samples": 0,
            "natural_obstacle_fraction": 0.0,
            "sampled_obstacle_fraction": 0.0,
        }

    mask = dataset.obstacle_target_mask()
    obstacle_count = int(mask.sum())
    non_obstacle_count = int(total - obstacle_count)
    natural_fraction = float(obstacle_count) / float(total)

    if obstacle_count == 0 or non_obstacle_count == 0 or target_fraction <= 0.0:
        return np.ones((total,), dtype=np.float64), {
            "enabled": False,
            "num_samples": total,
            "obstacle_samples": obstacle_count,
            "non_obstacle_samples": non_obstacle_count,
            "natural_obstacle_fraction": natural_fraction,
            "sampled_obstacle_fraction": natural_fraction,
        }

    target_fraction = float(np.clip(target_fraction, 1e-3, 1.0 - 1e-3))
    obstacle_weight = target_fraction / float(obstacle_count)
    non_obstacle_weight = (1.0 - target_fraction) / float(non_obstacle_count)
    weights = np.where(mask, obstacle_weight, non_obstacle_weight).astype(np.float64)
    weights /= max(weights.mean(), 1e-12)
    return weights, {
        "enabled": True,
        "num_samples": total,
        "obstacle_samples": obstacle_count,
        "non_obstacle_samples": non_obstacle_count,
        "natural_obstacle_fraction": natural_fraction,
        "sampled_obstacle_fraction": target_fraction,
    }


def build_loaders(
    data_dir: Path,
    val_fraction: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    max_history: int,
    min_history: int,
    histories_per_target: int,
    exclude_after_teleport_steps: int,
    history_step_min: int,
    history_step_max: int,
    obstacle_oversample_target_frac: float,
    feature_mode: str,
    no_hit_range_cm: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, dict]:
    log(f"Scanning data directory: {data_dir}")
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No data files found in {data_dir}")
    log(f"Found {len(files)} data files")

    rng = np.random.default_rng(seed)
    order = np.arange(len(files))
    rng.shuffle(order)
    files = [files[i] for i in order]
    log("Shuffled file order for train/val split")

    if len(files) == 1:
        log("Single data file detected, applying temporal split for validation")
        try:
            single_world = load_world_file(files[0])
        except ValueError as exc:
            raise SystemExit(f"No valid data files in {data_dir}: {exc}") from exc
        train_world, val_world = _split_single_world_for_validation(single_world, val_fraction)
        train_worlds = [train_world]
        val_worlds = [val_world]
        train_files = [Path(f"{files[0].name}::train_split")]
        val_files = [Path(f"{files[0].name}::val_split")]
    else:
        val_count = max(1, int(round(len(files) * val_fraction)))
        val_files = files[:val_count]
        train_files = files[val_count:]
        if not train_files:
            train_files, val_files = val_files, train_files
        log(f"Split worlds -> train={len(train_files)} val={len(val_files)}")
        train_worlds = []
        loaded_train_files: list[Path] = []
        for i, p in enumerate(train_files, start=1):
            log(f"Loading train world {i}/{len(train_files)}: {p.name}")
            try:
                train_worlds.append(load_world_file(p))
                loaded_train_files.append(p)
            except ValueError as exc:
                log(f"Skipping invalid train world {p.name}: {exc}")
        train_files = loaded_train_files
        val_worlds = []
        loaded_val_files: list[Path] = []
        for i, p in enumerate(val_files, start=1):
            log(f"Loading val world {i}/{len(val_files)}: {p.name}")
            try:
                val_worlds.append(load_world_file(p))
                loaded_val_files.append(p)
            except ValueError as exc:
                log(f"Skipping invalid val world {p.name}: {exc}")
        val_files = loaded_val_files
        if not train_worlds:
            raise SystemExit(
                "No valid training worlds were found. "
                "Remove malformed files or regenerate data."
            )
        if not val_worlds:
            log("No valid validation worlds found; using first training world for validation.")
            val_worlds = [train_worlds[0]]
            val_files = [Path(f"{train_files[0].name}::val_fallback")]
    log("Computing normalization stats from training worlds")
    stats = compute_norm_stats(train_worlds, feature_mode)

    log("Converting worlds to model features")
    train_feats, train_targets = zip(
        *(world_to_features(w, stats, feature_mode, no_hit_range_cm=no_hit_range_cm) for w in train_worlds)
    )
    val_feats, val_targets = zip(
        *(world_to_features(w, stats, feature_mode, no_hit_range_cm=no_hit_range_cm) for w in val_worlds)
    )
    train_teleports = [w.teleport_flag for w in train_worlds]
    val_teleports = [w.teleport_flag for w in val_worlds]

    history_min_label = int(max(min_history, 1))
    history_max_label = "unbounded" if int(max_history) <= 0 else str(int(max_history))
    log(
        "Constructing sequence-piece datasets "
        f"(history_len={history_min_label}..{history_max_label} "
        f"sampled histories_per_target={histories_per_target} "
        f"history_step_range={max(1, int(history_step_min))}..{max(int(history_step_min), int(history_step_max))})"
    )
    train_ds = SequencePieceDataset(
        list(train_feats),
        list(train_targets),
        train_teleports,
        max_history=max_history,
        min_history=min_history,
        histories_per_target=histories_per_target,
        exclude_after_teleport_steps=exclude_after_teleport_steps,
        history_step_min=history_step_min,
        history_step_max=history_step_max,
        seed=seed,
    )
    val_ds = SequencePieceDataset(
        list(val_feats),
        list(val_targets),
        val_teleports,
        max_history=max_history,
        min_history=min_history,
        histories_per_target=histories_per_target,
        exclude_after_teleport_steps=exclude_after_teleport_steps,
        history_step_min=history_step_min,
        history_step_max=history_step_max,
        seed=seed + 1,
    )
    log(f"Dataset samples -> train={len(train_ds)} val={len(val_ds)}")

    train_sample_weights, oversample_meta = compute_obstacle_oversample_weights(
        train_ds,
        obstacle_oversample_target_frac,
    )
    train_sampler = None
    if oversample_meta["enabled"]:
        train_sampler = WeightedRandomSampler(
            torch.from_numpy(train_sample_weights),
            num_samples=len(train_ds),
            replacement=True,
        )
        log(
            "Using obstacle-target oversampling "
            f"(natural_frac={oversample_meta['natural_obstacle_fraction']:.3f} "
            f"sampled_frac={oversample_meta['sampled_obstacle_fraction']:.3f} "
            f"obstacle_samples={oversample_meta['obstacle_samples']} "
            f"non_obstacle_samples={oversample_meta['non_obstacle_samples']})"
        )
    else:
        log(
            "Obstacle-target oversampling disabled "
            f"(obstacle_samples={oversample_meta['obstacle_samples']} "
            f"non_obstacle_samples={oversample_meta['non_obstacle_samples']})"
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": int(max(num_workers, 0)),
        "pin_memory": bool(pin_memory),
        "collate_fn": collate_padded,
    }
    if int(max(num_workers, 0)) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(max(prefetch_factor, 2))

    train_loader = DataLoader(
        train_ds,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_kwargs,
    )
    log(
        "DataLoaders ready -> "
        f"batch_size={batch_size} num_workers={int(max(num_workers, 0))} "
        f"pin_memory={bool(pin_memory)}"
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )

    meta = {
        "train_files": [p.name for p in train_files],
        "val_files": [p.name for p in val_files],
        "num_train_worlds": len(train_worlds),
        "num_val_worlds": len(val_worlds),
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
        "num_sensors": int(train_worlds[0].lidar_cm.shape[1]),
        "input_dim": int(train_feats[0].shape[1]),
        "num_classes": 3,
        "min_history": int(max(min_history, 1)),
        "max_history": int(max_history) if int(max_history) > 0 else 0,
        "feature_mode": feature_mode,
        "no_hit_range_cm": float(no_hit_range_cm),
        "histories_per_target": int(histories_per_target),
        "exclude_after_teleport_steps": int(exclude_after_teleport_steps),
        "history_step_min": int(max(history_step_min, 1)),
        "history_step_max": int(max(history_step_max, max(history_step_min, 1))),
        "sequence_sampling": "sampled_per_target_variable_step",
        "obstacle_oversample_target_frac": float(obstacle_oversample_target_frac),
        "train_obstacle_target_samples": int(oversample_meta["obstacle_samples"]),
        "train_non_obstacle_target_samples": int(oversample_meta["non_obstacle_samples"]),
        "train_obstacle_target_fraction": float(oversample_meta["natural_obstacle_fraction"]),
        "train_sampled_obstacle_target_fraction": float(oversample_meta["sampled_obstacle_fraction"]),
        "train_teleport_rows": int(np.sum([int(np.sum(w > 0)) for w in train_teleports])),
        "val_teleport_rows": int(np.sum([int(np.sum(w > 0)) for w in val_teleports])),
        "train_class_counts": np.bincount(
            np.concatenate([t.reshape(-1) for t in train_targets]),
            minlength=3,
        ).astype(np.int64).tolist(),
        "val_class_counts": np.bincount(
            np.concatenate([t.reshape(-1) for t in val_targets]),
            minlength=3,
        ).astype(np.int64).tolist(),
        "norm": {
            "pose_mean": stats.pose_mean.tolist(),
            "pose_std": stats.pose_std.tolist(),
            "dist_mean": stats.dist_mean,
            "dist_std": stats.dist_std,
        },
    }
    return train_loader, val_loader, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train obstacle-first CNN for current-step lidar class prediction.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/gru_lidar_classifier.pt"))
    parser.add_argument("--eval-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--model-arch",
        type=str,
        choices=(
            "obstacle_first_ego_map_cnn",
            "obstacle_first_tsunet",
            "obstacle_first_causal_tscnn",
        ),
        default="obstacle_first_ego_map_cnn",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--disable-pin-memory", action="store_true")
    parser.add_argument("--amp", type=str, choices=("auto", "on", "off"), default="off")
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--max-history", type=int, default=64)
    parser.add_argument("--min-history", type=int, default=8)
    parser.add_argument("--histories-per-target", type=int, default=3)
    parser.add_argument("--exclude-after-teleport-steps", type=int, default=1)
    parser.add_argument("--history-step-min", type=int, default=4)
    parser.add_argument("--history-step-max", type=int, default=16)
    parser.add_argument("--no-hit-range-cm", type=float, default=DEFAULT_LIDAR_MAX_RANGE_CM)
    parser.add_argument("--obstacle-oversample-target-frac", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every-batches", type=int, default=25)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--obstacle-class-focus-weight", type=float, default=0.75)
    parser.add_argument("--obstacle-aux-weight", type=float, default=0.35)
    parser.add_argument("--obstacle-pos-weight-cap", type=float, default=4.0)
    parser.add_argument("--hard-example-fraction", type=float, default=1.0)
    parser.add_argument("--obstacle-primary-loss-weight", type=float, default=2.0)
    parser.add_argument("--safe-type-loss-weight", type=float, default=0.2)
    parser.add_argument("--binary-obstacle-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-min-epochs", type=int, default=4)
    parser.add_argument("--sweep-obstacle-logit-bias-min", type=float, default=-1.0)
    parser.add_argument("--sweep-obstacle-logit-bias-max", type=float, default=1.0)
    parser.add_argument("--sweep-obstacle-logit-bias-step", type=float, default=0.1)
    args = parser.parse_args()
    if args.min_history < 1:
        raise SystemExit("--min-history must be >= 1")
    if args.max_history > 0 and args.min_history > args.max_history:
        raise SystemExit("--min-history cannot exceed --max-history")

    report_base = args.eval_checkpoint if args.eval_checkpoint is not None else args.output
    report_base.parent.mkdir(parents=True, exist_ok=True)
    log_path = (
        report_base.with_suffix(".sweep.log.txt")
        if args.eval_checkpoint is not None
        else args.output.with_suffix(".log.txt")
    )
    log_fh = log_path.open("w", encoding="utf-8")
    set_log_file(log_fh)

    log("Starting lidar training script")
    log("Parsed args: " + json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))
    log(f"Writing logs to: {log_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    log(f"Seeds set -> numpy={args.seed} torch={args.seed}")

    device, device_kind = select_runtime_device(args.device)
    configure_runtime_for_device(device_kind)
    loader_num_workers = resolve_data_loader_workers(args.num_workers)
    loader_pin_memory = bool(device_kind == "cuda" and not args.disable_pin_memory)
    amp_enabled = bool(device_kind == "cuda" and args.amp != "off")
    if args.amp == "on" and device_kind != "cuda":
        log("AMP requested, but mixed precision is only enabled for CUDA in this script; continuing without AMP.")
    log(
        "Runtime tuning: "
        f"device={device_kind} num_workers={loader_num_workers} "
        f"pin_memory={loader_pin_memory} amp={amp_enabled}"
    )
    binary_obstacle_only_mode = bool(args.binary_obstacle_only)
    preloaded_eval_ckpt: dict | None = None
    if args.eval_checkpoint is not None:
        preloaded_eval_ckpt = load_checkpoint_for_device(args.eval_checkpoint, device, device_kind)
        preloaded_model_config = dict(preloaded_eval_ckpt["model_config"])
        resolved_eval_model_type = infer_checkpoint_model_type(
            preloaded_model_config,
            preloaded_eval_ckpt.get("model_state_dict"),
        )
        default_eval_feature_mode = (
            EGO_MAP_FEATURE_MODE if resolved_eval_model_type == "obstacle_first_ego_map_cnn" else LEGACY_FEATURE_MODE
        )
        feature_mode = str(preloaded_eval_ckpt.get("meta", {}).get("feature_mode", default_eval_feature_mode))
        no_hit_range_cm = float(preloaded_eval_ckpt.get("meta", {}).get("no_hit_range_cm", args.no_hit_range_cm))
    else:
        feature_mode = EGO_MAP_FEATURE_MODE if args.model_arch == "obstacle_first_ego_map_cnn" else LEGACY_FEATURE_MODE
        no_hit_range_cm = float(args.no_hit_range_cm)

    effective_obstacle_oversample_target_frac = (
        0.0 if binary_obstacle_only_mode else args.obstacle_oversample_target_frac
    )
    train_loader, val_loader, meta = build_loaders(
        data_dir=args.data_dir,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        num_workers=loader_num_workers,
        pin_memory=loader_pin_memory,
        prefetch_factor=args.prefetch_factor,
        max_history=args.max_history,
        min_history=args.min_history,
        histories_per_target=args.histories_per_target,
        exclude_after_teleport_steps=args.exclude_after_teleport_steps,
        history_step_min=args.history_step_min,
        history_step_max=args.history_step_max,
        obstacle_oversample_target_frac=effective_obstacle_oversample_target_frac,
        feature_mode=feature_mode,
        no_hit_range_cm=no_hit_range_cm,
        seed=args.seed,
    )
    if len(train_loader.dataset) == 0:
        raise SystemExit("No training samples were constructed.")
    log(
        "Loader summary: "
        f"train_worlds={meta['num_train_worlds']} val_worlds={meta['num_val_worlds']} "
        f"train_samples={meta['num_train_samples']} val_samples={meta['num_val_samples']}"
    )
    log(
        "Teleport rows: "
        f"train={meta['train_teleport_rows']} val={meta['val_teleport_rows']} "
        f"(exclude_after_teleport_steps={meta['exclude_after_teleport_steps']})"
    )
    log(
        "History sampling: "
        f"length={meta['min_history']}..{('unbounded' if int(meta['max_history']) <= 0 else meta['max_history'])} "
        f"sampled={meta['histories_per_target']} "
        f"step_range={meta['history_step_min']}..{meta['history_step_max']}"
    )
    log(
        "Obstacle-target sample mix: "
        f"natural={meta['train_obstacle_target_fraction']:.3f} "
        f"sampled={meta['train_sampled_obstacle_target_fraction']:.3f} "
        f"counts={meta['train_obstacle_target_samples']}/{meta['train_non_obstacle_target_samples']}"
    )
    if binary_obstacle_only_mode:
        log("Binary obstacle-only mode disables obstacle-window oversampling; using natural sample mix.")
    log(
        "Class counts (ground, obstacle, none): "
        f"train={meta['train_class_counts']} val={meta['val_class_counts']}"
    )

    log(f"Using device: {device} ({device_kind})")
    if args.eval_checkpoint is not None:
        log(f"Eval-only mode: loading checkpoint {args.eval_checkpoint}")
        ckpt = preloaded_eval_ckpt if preloaded_eval_ckpt is not None else load_checkpoint_for_device(
            args.eval_checkpoint,
            device,
            device_kind,
        )
        model_config = dict(ckpt["model_config"])
        model_config["model_type"] = infer_checkpoint_model_type(model_config, ckpt.get("model_state_dict"))
        model = build_lidar_model_from_config(model_config, ckpt.get("model_state_dict")).to(device)
        load_model_state_with_compat(model, ckpt["model_state_dict"], log)
        binary_obstacle_only_mode = bool(model_config.get("binary_obstacle_only", binary_obstacle_only_mode))
    else:
        if args.model_arch == "obstacle_first_ego_map_cnn":
            model = EgocentricMapLidarClassifier(
                input_dim=meta["input_dim"],
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_sensors=meta["num_sensors"],
                num_classes=meta["num_classes"],
                dropout=args.dropout,
                max_range_cm=no_hit_range_cm,
            ).to(device)
        elif args.model_arch == "obstacle_first_causal_tscnn":
            model = LegacyCausalTSCNNClassifier(
                input_dim=meta["input_dim"],
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_sensors=meta["num_sensors"],
                num_classes=meta["num_classes"],
                dropout=args.dropout,
            ).to(device)
        else:
            model = GRULidarClassifier(
                input_dim=meta["input_dim"],
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_sensors=meta["num_sensors"],
                num_classes=meta["num_classes"],
                dropout=args.dropout,
            ).to(device)
        model_config = {
            "model_type": model.model_type,
            "architecture_revision": 3,
            "input_dim": meta["input_dim"],
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_sensors": meta["num_sensors"],
            "num_classes": meta["num_classes"],
            "dropout": args.dropout,
            "feature_mode": meta["feature_mode"],
            "no_hit_range_cm": float(meta["no_hit_range_cm"]),
            "region_sensor_groups": getattr(model, "region_sensor_groups", {}),
            "fusion_hidden_dim": int(getattr(model, "fusion_hidden_dim", args.hidden_dim)),
            "attention_ff_dim": int(getattr(model, "attention_ff_dim", args.hidden_dim)),
            "attention_heads": int(max(getattr(model, "region_attention_heads", {}).values(), default=1)),
            "region_attention_heads": getattr(model, "region_attention_heads", {}),
            "sensor_embed_dim": int(getattr(model, "sensor_embed_dim", max(4, min(16, args.hidden_dim // 4)))),
            "decoder_hidden_dim": int(getattr(model, "decoder_hidden_dim", max(48, args.hidden_dim))),
            "obstacle_context_dim": int(getattr(model, "obstacle_context_dim", 0)),
            "aux_decoder_levels": int(getattr(model, "aux_decoder_levels", 0)),
            "binary_obstacle_only": bool(binary_obstacle_only_mode),
        }
        if model.model_type == "obstacle_first_ego_map_cnn":
            model_config.update(
                {
                    "map_half_extent_cm": float(model.map_half_extent_cm),
                    "map_cell_size_cm": float(model.map_cell_size_cm),
                    "map_ray_write_samples": int(model.map_ray_write_samples),
                    "query_ray_samples": int(model.query_ray_samples),
                }
            )
    param_count = sum(p.numel() for p in model.parameters())
    log(f"Model initialized: type={model_config['model_type']} params={param_count}")
    log(
        "Feature pipeline: "
        f"mode={meta['feature_mode']} "
        f"no_hit_range_cm={meta['no_hit_range_cm']:.1f}"
    )
    if model_config["model_type"] == "obstacle_first_ego_map_cnn":
        log(
            "Egocentric map CNN layout: "
            f"map={getattr(model, 'map_size', 'n/a')}x{getattr(model, 'map_size', 'n/a')} "
            f"cell_cm={model_config.get('map_cell_size_cm', 'n/a')} "
            f"half_extent_cm={model_config.get('map_half_extent_cm', 'n/a')} "
            f"write_samples={model_config.get('map_ray_write_samples', 'n/a')} "
            f"query_samples={model_config.get('query_ray_samples', 'n/a')} "
            f"hidden={model_config['hidden_dim']} "
            f"fusion={model_config['fusion_hidden_dim']} "
            f"decoder_hidden={model_config['decoder_hidden_dim']}"
        )
    elif model_config["model_type"] == "obstacle_first_tsunet":
        log(
            "Time-Sensor U-Net layout: "
            f"groups={model_config['region_sensor_groups']} "
            f"region_attention_heads={model_config.get('region_attention_heads', {})} "
            f"sensor_hidden={model_config['hidden_dim']} "
            f"unet_depth={model.unet_depth} "
            f"global_context={model_config['fusion_hidden_dim']} "
            f"attention_ff={model_config['attention_ff_dim']} "
            f"sensor_embed={model_config['sensor_embed_dim']} "
            f"decoder_hidden={model_config['decoder_hidden_dim']} "
            f"obstacle_context={model_config.get('obstacle_context_dim', 'n/a')}"
        )
    else:
        log(
            "Causal TS-CNN layout: "
            f"groups={model_config['region_sensor_groups']} "
            f"region_attention_heads={model_config.get('region_attention_heads', {})} "
            f"sensor_hidden={model_config['hidden_dim']} "
            f"blocks={getattr(model, 'backbone_depth', model_config['num_layers'])} "
            f"global_context={model_config['fusion_hidden_dim']} "
            f"attention_ff={model_config['attention_ff_dim']} "
            f"sensor_embed={model_config['sensor_embed_dim']} "
            f"decoder_hidden={model_config['decoder_hidden_dim']} "
            f"obstacle_context={model_config.get('obstacle_context_dim', 'n/a')}"
        )

    eval_num_classes = 2 if binary_obstacle_only_mode else meta["num_classes"]

    if args.eval_checkpoint is not None:
        log("Running eval-only detailed validation evaluation")
        final_report = evaluate_detailed(
            model,
            val_loader,
            device,
            device_kind,
            eval_num_classes,
            binary_obstacle_only=binary_obstacle_only_mode,
        )
        final_report.update(
            {
                "num_train_samples": meta["num_train_samples"],
                "num_val_samples": meta["num_val_samples"],
                "source_checkpoint": str(args.eval_checkpoint),
                "binary_obstacle_only": bool(binary_obstacle_only_mode),
            }
        )
        val_report_path = args.eval_checkpoint.with_suffix(".eval_val_report.json")
        with val_report_path.open("w", encoding="utf-8") as fh:
            json.dump(final_report, fh, indent=2)

        sweep_biases = build_obstacle_logit_bias_sweep(
            args.sweep_obstacle_logit_bias_min,
            args.sweep_obstacle_logit_bias_max,
            args.sweep_obstacle_logit_bias_step,
        )
        log(
            "Running obstacle-logit bias sweep: "
            f"{sweep_biases[0]:.3f}..{sweep_biases[-1]:.3f} "
            f"step={args.sweep_obstacle_logit_bias_step:.3f} "
            f"count={len(sweep_biases)}"
        )
        sweep_report = run_obstacle_logit_bias_sweep(
            model,
            val_loader,
            device,
            device_kind,
            eval_num_classes,
            sweep_biases,
            binary_obstacle_only=binary_obstacle_only_mode,
        )
        sweep_report.update(
            {
                "source_checkpoint": str(args.eval_checkpoint),
                "num_train_samples": meta["num_train_samples"],
                "num_val_samples": meta["num_val_samples"],
                "binary_obstacle_only": bool(binary_obstacle_only_mode),
            }
        )
        sweep_report_path = args.eval_checkpoint.with_suffix(".threshold_sweep.json")
        with sweep_report_path.open("w", encoding="utf-8") as fh:
            json.dump(sweep_report, fh, indent=2)
        log("Eval-only sweep complete")
        log_plain(f"saved val report: {val_report_path}")
        log_plain(f"saved threshold sweep: {sweep_report_path}")
        log_plain(f"saved log: {log_path}")
        log_fh.close()
        set_log_file(None)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    log(f"Optimizer initialized: AdamW lr={args.lr} weight_decay={args.weight_decay}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        min_lr=args.min_lr,
    )
    log(
        "Scheduler initialized: ReduceLROnPlateau "
        f"factor={args.plateau_factor} patience={args.plateau_patience} min_lr={args.min_lr}"
    )
    class_weights_np = compute_class_weights(np.asarray(meta["train_class_counts"], dtype=np.float32))
    obstacle_pos_weight = compute_obstacle_ground_pos_weight(
        np.asarray(meta["train_class_counts"], dtype=np.float32),
        max_weight=args.obstacle_pos_weight_cap,
    )
    if binary_obstacle_only_mode:
        log(
            "Using binary obstacle-only training: obstacle-vs-non-obstacle only "
            "with plain BCE on all rays "
            f"(hard_example_fraction and obstacle_primary_loss_weight ignored: "
            f"{args.hard_example_fraction:.3f}, {args.obstacle_primary_loss_weight:.3f})"
        )
        log(
            "Ground-vs-none refinement is disabled for this experiment; "
            "the run now directly tests whether the backbone can learn obstacle detection."
        )
    else:
        log(
            "Using obstacle-first heads: obstacle-vs-non-obstacle primary plus ground-vs-none on non-obstacle rays "
            f"hard_example_fraction={args.hard_example_fraction:.3f} "
            f"obstacle_primary_loss_weight={args.obstacle_primary_loss_weight:.3f} "
            f"safe_type_loss_weight={args.safe_type_loss_weight:.3f}"
        )
        log(
            "Legacy class-weight/focal settings are ignored by the obstacle-first loss; "
            "the model now optimizes obstacle detection first and refines safe rays second."
        )
    best = {"val_loss": float("inf"), "epoch": -1}
    history: list[dict] = []
    metrics_csv_path = args.output.with_suffix(".metrics.csv")
    split_manifest_path = args.output.with_suffix(".split.json")
    log(f"Output directory ready: {args.output.parent}")
    with split_manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "train_files": meta["train_files"],
                "val_files": meta["val_files"],
                "num_train_worlds": meta["num_train_worlds"],
                "num_val_worlds": meta["num_val_worlds"],
                "num_train_samples": meta["num_train_samples"],
                "num_val_samples": meta["num_val_samples"],
                "min_history": meta["min_history"],
                "max_history": meta["max_history"],
                "histories_per_target": args.histories_per_target,
                "exclude_after_teleport_steps": args.exclude_after_teleport_steps,
                "history_step_min": args.history_step_min,
                "history_step_max": args.history_step_max,
                "feature_mode": meta["feature_mode"],
                "no_hit_range_cm": float(meta["no_hit_range_cm"]),
                "obstacle_oversample_target_frac": effective_obstacle_oversample_target_frac,
                "sequence_sampling": meta["sequence_sampling"],
                "train_teleport_rows": meta["train_teleport_rows"],
                "val_teleport_rows": meta["val_teleport_rows"],
                "train_obstacle_target_samples": meta["train_obstacle_target_samples"],
                "train_non_obstacle_target_samples": meta["train_non_obstacle_target_samples"],
                "train_obstacle_target_fraction": meta["train_obstacle_target_fraction"],
                "train_sampled_obstacle_target_fraction": meta["train_sampled_obstacle_target_fraction"],
                "train_class_counts": meta["train_class_counts"],
                "val_class_counts": meta["val_class_counts"],
                "obstacle_aux_weight": args.obstacle_aux_weight,
                "obstacle_class_focus_weight": args.obstacle_class_focus_weight,
                "obstacle_pos_weight_reference": obstacle_pos_weight,
                "obstacle_pos_weight_cap": args.obstacle_pos_weight_cap,
                "hard_example_fraction": args.hard_example_fraction,
            },
            fh,
            indent=2,
        )
    log(f"Wrote split manifest: {split_manifest_path}")
    with metrics_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_loss"])
    log(f"Initialized metrics CSV: {metrics_csv_path}")

    epochs_without_improve = 0
    for epoch in range(1, args.epochs + 1):
        log(f"Epoch {epoch:03d}/{args.epochs:03d} started")
        t0 = perf_counter()
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            device_kind,
            eval_num_classes,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            phase_name="train",
            log_every_batches=args.log_every_batches,
            class_weights=None,
            label_smoothing=args.label_smoothing,
            grad_clip_norm=args.grad_clip_norm,
            focal_gamma=args.focal_gamma,
            obstacle_class_focus_weight=args.obstacle_class_focus_weight,
            obstacle_aux_weight=args.obstacle_aux_weight,
            obstacle_pos_weight=obstacle_pos_weight,
            hard_example_fraction=args.hard_example_fraction,
            obstacle_primary_loss_weight=args.obstacle_primary_loss_weight,
            safe_type_loss_weight=args.safe_type_loss_weight,
            binary_obstacle_only=binary_obstacle_only_mode,
            use_amp=amp_enabled,
            grad_scaler=grad_scaler,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            None,
            device,
            device_kind,
            eval_num_classes,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            phase_name="val",
            log_every_batches=args.log_every_batches,
            class_weights=None,
            label_smoothing=args.label_smoothing,
            focal_gamma=args.focal_gamma,
            obstacle_class_focus_weight=args.obstacle_class_focus_weight,
            obstacle_aux_weight=args.obstacle_aux_weight,
            obstacle_pos_weight=obstacle_pos_weight,
            hard_example_fraction=args.hard_example_fraction,
            obstacle_primary_loss_weight=args.obstacle_primary_loss_weight,
            safe_type_loss_weight=args.safe_type_loss_weight,
            binary_obstacle_only=binary_obstacle_only_mode,
            use_amp=amp_enabled,
        )
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        log(f"Scheduler step complete -> lr={current_lr:.8f}")
        epoch_s = perf_counter() - t0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_seconds": epoch_s,
            }
        )
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch}
            epochs_without_improve = 0
            log(
                f"New best validation loss at epoch {epoch:03d}: "
                f"{best['val_loss']:.6f}. Saving checkpoint -> {args.output}"
            )
            save_checkpoint_for_device(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "norm": meta["norm"],
                    "history": history,
                    "meta": {
                        "train_files": meta["train_files"],
                        "val_files": meta["val_files"],
                        "min_history": meta["min_history"],
                        "max_history": meta["max_history"],
                        "histories_per_target": args.histories_per_target,
                        "exclude_after_teleport_steps": args.exclude_after_teleport_steps,
                        "history_step_min": args.history_step_min,
                        "history_step_max": args.history_step_max,
                        "feature_mode": meta["feature_mode"],
                        "no_hit_range_cm": float(meta["no_hit_range_cm"]),
                        "obstacle_oversample_target_frac": effective_obstacle_oversample_target_frac,
                        "sequence_sampling": meta["sequence_sampling"],
                        "class_weighting": "inactive_factorized_reference_only",
                        "class_weights_reference": class_weights_np.tolist(),
                        "obstacle_class_focus_weight": args.obstacle_class_focus_weight,
                        "obstacle_aux_weight": args.obstacle_aux_weight,
                        "obstacle_pos_weight_reference": obstacle_pos_weight,
                        "obstacle_pos_weight_cap": args.obstacle_pos_weight_cap,
                        "hard_example_fraction": args.hard_example_fraction,
                        "obstacle_primary_loss_weight": args.obstacle_primary_loss_weight,
                        "safe_type_loss_weight": args.safe_type_loss_weight,
                        "binary_obstacle_only": bool(binary_obstacle_only_mode),
                        "loss_type": "binary_obstacle_only" if binary_obstacle_only_mode else "obstacle_first_binary",
                        "prediction_target": "current_step_lidar_class",
                        "focal_gamma": args.focal_gamma,
                        "label_smoothing": args.label_smoothing,
                        "weight_decay": args.weight_decay,
                        "grad_clip_norm": args.grad_clip_norm,
                        "train_class_counts": meta["train_class_counts"],
                        "val_class_counts": meta["val_class_counts"],
                        "best_epoch": best["epoch"],
                        "best_val_loss": best["val_loss"],
                    },
                },
                args.output,
                device_kind,
            )
        else:
            epochs_without_improve += 1
        with metrics_csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, best["val_loss"]])
        log_plain(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.5f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.5f} val_acc={val_acc:.4f} "
            f"best_val={best['val_loss']:.5f}@{best['epoch']:03d} "
            f"lr={current_lr:.7f} "
            f"time={epoch_s:.2f}s"
        )
        log(f"Epoch {epoch:03d}/{args.epochs:03d} finished")
        if epoch >= args.early_stop_min_epochs and epochs_without_improve >= args.early_stop_patience:
            log(
                f"Early stopping triggered after epoch {epoch:03d}; "
                f"no val improvement for {epochs_without_improve} epochs."
            )
            break

    history_path = args.output.with_suffix(".history.json")
    log("Writing final history JSON")
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    if args.output.exists():
        log("Loading best checkpoint for final validation report")
        best_ckpt = load_checkpoint_for_device(args.output, device, device_kind)
        model.load_state_dict(best_ckpt["model_state_dict"])
    log("Running detailed validation evaluation")
    final_report = evaluate_detailed(
        model,
        val_loader,
        device,
        device_kind,
        eval_num_classes,
        binary_obstacle_only=binary_obstacle_only_mode,
    )
    final_report.update(
        {
            "best_epoch": best["epoch"],
            "best_val_loss": best["val_loss"],
            "num_train_samples": meta["num_train_samples"],
            "num_val_samples": meta["num_val_samples"],
            "binary_obstacle_only": bool(binary_obstacle_only_mode),
        }
    )
    val_report_path = args.output.with_suffix(".val_report.json")
    with val_report_path.open("w", encoding="utf-8") as fh:
        json.dump(final_report, fh, indent=2)
    sweep_biases = build_obstacle_logit_bias_sweep(
        args.sweep_obstacle_logit_bias_min,
        args.sweep_obstacle_logit_bias_max,
        args.sweep_obstacle_logit_bias_step,
    )
    log(
        "Running post-train obstacle-logit bias sweep: "
        f"{sweep_biases[0]:.3f}..{sweep_biases[-1]:.3f} "
        f"step={args.sweep_obstacle_logit_bias_step:.3f} "
        f"count={len(sweep_biases)}"
    )
    sweep_report = run_obstacle_logit_bias_sweep(
        model,
        val_loader,
        device,
        device_kind,
        eval_num_classes,
        sweep_biases,
        binary_obstacle_only=binary_obstacle_only_mode,
    )
    sweep_report.update(
        {
            "best_epoch": best["epoch"],
            "best_val_loss": best["val_loss"],
            "num_train_samples": meta["num_train_samples"],
            "num_val_samples": meta["num_val_samples"],
            "binary_obstacle_only": bool(binary_obstacle_only_mode),
        }
    )
    sweep_report_path = args.output.with_suffix(".threshold_sweep.json")
    with sweep_report_path.open("w", encoding="utf-8") as fh:
        json.dump(sweep_report, fh, indent=2)
    log("Training run complete")
    log_plain(f"saved model: {args.output}")
    log_plain(f"saved history: {history_path}")
    log_plain(f"saved metrics csv: {metrics_csv_path}")
    log_plain(f"saved split manifest: {split_manifest_path}")
    log_plain(f"saved val report: {val_report_path}")
    log_plain(f"saved threshold sweep: {sweep_report_path}")
    log_plain(f"saved log: {log_path}")
    log_fh.close()
    set_log_file(None)


if __name__ == "__main__":
    main()
