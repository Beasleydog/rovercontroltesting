from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PingSample:
    rover_x_m: float
    rover_y_m: float
    ping_value: float


CURRENT_SCALE = 14.034819
CURRENT_DECAY = -0.066163
CURRENT_ESTIMATE_X_M = -6091.469360801471
CURRENT_ESTIMATE_Y_M = -10748.04475850622

# Round-1 logged pings from the run in question.
SAMPLES = (
    PingSample(-5841.8, -10420.101, -50.092),
    PingSample(-5882.8, -10476.4, -46.805),
    PingSample(-5805.8, -10486.3, -49.016),
)


def implied_formula(x_m: float, y_m: float) -> tuple[float, float, float]:
    distances = [
        math.hypot(x_m - sample.rover_x_m, y_m - sample.rover_y_m)
        for sample in SAMPLES
    ]
    if min(distances) <= 0.0:
        raise ValueError("Candidate target is exactly on top of a ping sample.")

    decay_12 = math.log(distances[1] / distances[0]) / (
        SAMPLES[1].ping_value - SAMPLES[0].ping_value
    )
    decay_13 = math.log(distances[2] / distances[0]) / (
        SAMPLES[2].ping_value - SAMPLES[0].ping_value
    )
    decay = 0.5 * (decay_12 + decay_13)
    scale_values = [
        distances[idx] / math.exp(decay * sample.ping_value)
        for idx, sample in enumerate(SAMPLES)
    ]
    scale = sum(scale_values) / len(scale_values)
    log_fit_error = sum(
        (
            math.log(distances[idx])
            - (math.log(scale) + decay * sample.ping_value)
        )
        ** 2
        for idx, sample in enumerate(SAMPLES)
    )
    return (scale, decay, log_fit_error)


def score_candidate(x_m: float, y_m: float) -> tuple[float, float, float, float]:
    scale, decay, log_fit_error = implied_formula(x_m, y_m)
    # Three pings alone do not identify a unique exponential. This biases the search
    # toward the current formula and current trilaterated target instead of selecting
    # an arbitrary exact-fit member from the solution family.
    regularization = (
        ((math.log(scale) - math.log(CURRENT_SCALE)) / 0.05) ** 2
        + ((decay - CURRENT_DECAY) / 0.01) ** 2
        + (
            math.hypot(
                x_m - CURRENT_ESTIMATE_X_M,
                y_m - CURRENT_ESTIMATE_Y_M,
            )
            / 50.0
        )
        ** 2
    )
    return (log_fit_error * 1_000_000.0 + regularization, scale, decay, log_fit_error)


def main() -> None:
    best: tuple[float, float, float, float, float, float] | None = None

    for radius_m in range(0, 801, 2):
        for angle_idx in range(720):
            angle = 2.0 * math.pi * angle_idx / 720.0
            x_m = CURRENT_ESTIMATE_X_M + radius_m * math.cos(angle)
            y_m = CURRENT_ESTIMATE_Y_M + radius_m * math.sin(angle)
            score, scale, decay, log_fit_error = score_candidate(x_m, y_m)
            if best is None or score < best[0]:
                best = (score, x_m, y_m, scale, decay, log_fit_error)

    assert best is not None
    best_score, best_x_m, best_y_m, best_scale, best_decay, best_log_fit_error = best
    refine_step_m = 50.0

    for _ in range(50_000):
        x_m = best_x_m + random.uniform(-refine_step_m, refine_step_m)
        y_m = best_y_m + random.uniform(-refine_step_m, refine_step_m)
        score, scale, decay, log_fit_error = score_candidate(x_m, y_m)
        if score < best_score:
            best_score = score
            best_x_m = x_m
            best_y_m = y_m
            best_scale = scale
            best_decay = decay
            best_log_fit_error = log_fit_error
            refine_step_m = max(refine_step_m * 0.999, 0.01)

    print("Best overlap search result")
    print(f"  target = ({best_x_m:.6f}, {best_y_m:.6f}) m")
    print(
        f"  formula = {best_scale:.9f} * exp({best_decay:.9f} * ping)"
    )
    print(f"  log-fit error = {best_log_fit_error:.12e}")
    print()
    print("Per-sample residuals")
    for idx, sample in enumerate(SAMPLES, start=1):
        actual_distance = math.hypot(
            best_x_m - sample.rover_x_m,
            best_y_m - sample.rover_y_m,
        )
        predicted_distance = best_scale * math.exp(best_decay * sample.ping_value)
        print(
            f"  ping {idx}: distance={actual_distance:.6f} m | "
            f"predicted={predicted_distance:.6f} m | "
            f"residual={actual_distance - predicted_distance:.6f} m"
        )


if __name__ == "__main__":
    main()
