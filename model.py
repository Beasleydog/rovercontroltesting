from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

try:
    from train import (
        DEFAULT_LIDAR_MAX_RANGE_CM,
        GRULidarInferencer,
        NormStats,
        load_checkpoint_for_device,
        load_gru_lidar_inferencer,
        load_model_state_with_compat,
        select_runtime_device,
    )
except Exception:
    load_gru_lidar_inferencer = None
    DEFAULT_LIDAR_MAX_RANGE_CM = 1000.0
    GRULidarInferencer = None
    NormStats = None
    load_checkpoint_for_device = None
    load_model_state_with_compat = None
    select_runtime_device = None


RUNS_DIR = Path(__file__).resolve().parent / "runs"
SIMPLE_RUNS_DIR = Path(__file__).resolve().parent / "simpleruns"
LEGACY_CHECKPOINT_PATH = RUNS_DIR / "gru_lidar_classifier.pt"
SIMPLE_CHECKPOINT_PATH = SIMPLE_RUNS_DIR / "pose_aligned_beam_transformer.pt"
EXTERNAL_SIMPLETRAIN_PATH = Path(__file__).resolve().parent.parent / "lidarmodeltesting" / "simpletrain.py"
MAX_HISTORY = 64
DEFAULT_OBSTACLE_LOGIT_BIAS = 0.0
FALLBACK_OBSTACLE_MAX_CM = 1000.0
USE_SIMPLE_MODEL = True

_INFERENCER = None

_FEATURE_HISTORY: list[np.ndarray] = []
_SIGNATURE_HISTORY: list[np.ndarray] = []


def _active_checkpoint_path() -> Path:
    return SIMPLE_CHECKPOINT_PATH if USE_SIMPLE_MODEL else LEGACY_CHECKPOINT_PATH


def _load_external_simpletrain_module():
    if not EXTERNAL_SIMPLETRAIN_PATH.exists():
        raise FileNotFoundError(f"Simple model loader not found: {EXTERNAL_SIMPLETRAIN_PATH}")
    spec = importlib.util.spec_from_file_location("_external_simpletrain", EXTERNAL_SIMPLETRAIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {EXTERNAL_SIMPLETRAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pose_aligned_transformer_inferencer(checkpoint_path: Path) -> GRULidarInferencer:
    if any(value is None for value in (GRULidarInferencer, NormStats, load_checkpoint_for_device, load_model_state_with_compat, select_runtime_device)):
        raise RuntimeError("Local train.py helpers are unavailable for simple-model inference.")

    simpletrain = _load_external_simpletrain_module()
    PoseAlignedBeamTransformerClassifier = simpletrain.PoseAlignedBeamTransformerClassifier
    device_t, device_kind = select_runtime_device("auto")
    ckpt = load_checkpoint_for_device(checkpoint_path, device_t, device_kind)
    cfg = dict(ckpt["model_config"])
    model = PoseAlignedBeamTransformerClassifier(
        input_dim=int(cfg["input_dim"]),
        num_sensors=int(cfg["num_sensors"]),
        num_classes=int(cfg["num_classes"]),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
        max_range_cm=float(cfg.get("no_hit_range_cm", DEFAULT_LIDAR_MAX_RANGE_CM)),
        transformer_layers=int(cfg.get("transformer_layers", 4)),
        attention_heads=int(cfg.get("attention_heads", 4)),
        ff_mult=int(cfg.get("ff_mult", 4)),
        decoder_hidden_dim=int(cfg["decoder_hidden_dim"]),
    ).to(device_t)
    load_model_state_with_compat(model, ckpt["model_state_dict"])
    model.eval()

    norm = ckpt["norm"]
    stats = NormStats(
        pose_mean=np.asarray(norm["pose_mean"], dtype=np.float32),
        pose_std=np.asarray(norm["pose_std"], dtype=np.float32),
        dist_mean=float(norm["dist_mean"]),
        dist_std=float(norm["dist_std"]),
    )
    max_history = int(ckpt.get("meta", {}).get("max_history", MAX_HISTORY))
    return GRULidarInferencer(
        model=model,
        stats=stats,
        device=device_t,
        num_sensors=int(cfg["num_sensors"]),
        input_dim=int(cfg["input_dim"]),
        max_history=max_history,
        binary_obstacle_only=bool(ckpt.get("meta", {}).get("binary_obstacle_only", False)),
        no_hit_range_cm=float(ckpt.get("meta", {}).get("no_hit_range_cm", DEFAULT_LIDAR_MAX_RANGE_CM)),
    )


def _load_inferencer():
    checkpoint_path = _active_checkpoint_path()
    if USE_SIMPLE_MODEL:
        return _load_pose_aligned_transformer_inferencer(checkpoint_path)
    if load_gru_lidar_inferencer is None:
        raise RuntimeError("Local GRU loader is unavailable.")
    return load_gru_lidar_inferencer(checkpoint_path, max_history=MAX_HISTORY)


def configure_inference(use_simple_model: bool = False) -> None:
    global USE_SIMPLE_MODEL, _INFERENCER
    requested = bool(use_simple_model)
    if USE_SIMPLE_MODEL == requested and (_INFERENCER is not None or not _FEATURE_HISTORY):
        return
    USE_SIMPLE_MODEL = requested
    _INFERENCER = None
    reset_history()


def _ensure_inferencer_loaded() -> None:
    global _INFERENCER
    if _INFERENCER is not None:
        return
    checkpoint_path = _active_checkpoint_path()
    if not checkpoint_path.exists():
        return
    try:
        _INFERENCER = _load_inferencer()
    except Exception:
        _INFERENCER = None


def inferencer_backend() -> str:
    _ensure_inferencer_loaded()
    checkpoint_path = _active_checkpoint_path()
    if _INFERENCER is not None:
        if USE_SIMPLE_MODEL:
            return f"simple_pt_loaded:{checkpoint_path.name}"
        return "gru_pt_loaded:local_train_py"
    if checkpoint_path.exists() and not USE_SIMPLE_MODEL and load_gru_lidar_inferencer is None:
        return "fallback_no_local_train_loader"
    if checkpoint_path.exists():
        if USE_SIMPLE_MODEL:
            return "fallback_simple_inferencer_unavailable"
        return "fallback_inferencer_unavailable"
    return "fallback_no_checkpoint"


def reset_history() -> None:
    _FEATURE_HISTORY.clear()
    _SIGNATURE_HISTORY.clear()


def _build_history_signature(
    lidar_arr: np.ndarray,
    pose_arr: np.ndarray,
    basis_arr: np.ndarray | None,
) -> np.ndarray:
    parts = [pose_arr.reshape(-1), lidar_arr.reshape(-1)]
    if basis_arr is not None:
        parts.append(basis_arr.reshape(-1))
    return np.concatenate(parts).astype(np.float32, copy=False)


def _history_delta(signature_t: np.ndarray) -> float:
    if not _SIGNATURE_HISTORY:
        return float("inf")
    return float(np.max(np.abs(signature_t - _SIGNATURE_HISTORY[-1])))


def ingest_lidar(
    lidar_cm: np.ndarray,
    pose_xyz_cm: np.ndarray,
    basis: np.ndarray | None = None,
    obstacle_logit_bias: float = DEFAULT_OBSTACLE_LOGIT_BIAS,
    min_history_delta: float = 0.0,
    min_history_frames: int = 0,
) -> dict[str, np.ndarray | bool | int | float]:
    _ensure_inferencer_loaded()
    lidar_arr = np.asarray(lidar_cm, dtype=np.float32).reshape(-1)
    pose_arr = np.asarray(pose_xyz_cm, dtype=np.float32).reshape(-1)
    basis_arr = None if basis is None else np.asarray(basis, dtype=np.float32).reshape(3, 3)
    signature_t = _build_history_signature(lidar_arr, pose_arr, basis_arr)
    history_delta = _history_delta(signature_t)
    accept_current = history_delta >= max(0.0, float(min_history_delta))
    if not _SIGNATURE_HISTORY:
        accept_current = True
    effective_history_len = len(_FEATURE_HISTORY) + (1 if accept_current else 0)
    history_ready = effective_history_len >= max(0, int(min_history_frames))

    if _INFERENCER is None:
        obstacle_mask = (lidar_arr >= 0.0) & (lidar_arr <= FALLBACK_OBSTACLE_MAX_CM)
        class_ids = np.where(obstacle_mask, 1, 0).astype(np.int64)
        output: dict[str, np.ndarray | bool | int | float] = {
            "class_ids": class_ids,
            "obstacle_mask": obstacle_mask,
            "obstacle_logits": obstacle_mask.astype(np.float32),
            "history_len": effective_history_len,
            "history_ready": history_ready,
            "history_accepted": accept_current,
            "history_delta": history_delta,
        }
        if accept_current:
            _FEATURE_HISTORY.append(signature_t)
            _SIGNATURE_HISTORY.append(signature_t)
            if MAX_HISTORY > 0 and len(_FEATURE_HISTORY) > MAX_HISTORY:
                del _FEATURE_HISTORY[:-MAX_HISTORY]
                del _SIGNATURE_HISTORY[:-MAX_HISTORY]
        if not history_ready:
            output["class_ids"] = np.zeros_like(class_ids)
            output["obstacle_mask"] = np.zeros_like(obstacle_mask, dtype=bool)
            output["obstacle_logits"] = np.zeros_like(output["obstacle_logits"], dtype=np.float32)
        return output

    feature_t = _INFERENCER.featurize_timestep(
        pose_arr,
        lidar_arr,
        basis=basis_arr,
    )
    history = np.asarray([*_FEATURE_HISTORY, feature_t], dtype=np.float32)

    if _INFERENCER.binary_obstacle_only:
        obstacle_logits = _INFERENCER.predict_current_obstacle_logits_from_feature_history(history)
        obstacle_mask = _INFERENCER.predict_current_obstacle_mask_from_feature_history_with_bias(
            history,
            obstacle_logit_bias=float(obstacle_logit_bias),
        )
        class_ids = np.where(obstacle_mask, 1, 0).astype(np.int64)
        output = {
            "class_ids": class_ids,
            "obstacle_mask": obstacle_mask,
            "obstacle_logits": obstacle_logits,
        }
    else:
        class_ids = _INFERENCER.predict_current_from_feature_history(history).astype(np.int64)
        output = {
            "class_ids": class_ids,
            "obstacle_mask": (class_ids == 1),
        }

    output["history_len"] = effective_history_len
    output["history_ready"] = history_ready
    output["history_accepted"] = accept_current
    output["history_delta"] = history_delta

    if accept_current:
        _FEATURE_HISTORY.append(feature_t)
        _SIGNATURE_HISTORY.append(signature_t)
    if _INFERENCER.max_history > 0 and len(_FEATURE_HISTORY) > _INFERENCER.max_history:
        del _FEATURE_HISTORY[:-_INFERENCER.max_history]
        del _SIGNATURE_HISTORY[:-_INFERENCER.max_history]
    if not history_ready:
        output["class_ids"] = np.zeros_like(np.asarray(output["class_ids"]))
        output["obstacle_mask"] = np.zeros_like(np.asarray(output["obstacle_mask"]), dtype=bool)
        if "obstacle_logits" in output:
            output["obstacle_logits"] = np.zeros_like(np.asarray(output["obstacle_logits"]), dtype=np.float32)

    return output
