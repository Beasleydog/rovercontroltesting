# Project Overview

## Goal

Build a binary obstacle classifier for a rover equipped with 17 LiDAR distance
sensors. The classifier takes in raw distance readings (in cm) from all sensors
plus IMU data (heading, pitch, roll) and outputs whether the rover is currently
near an obstacle.

The key requirement is **low false positive rate** — the rover should not stop
for phantom obstacles. Missing a real obstacle is less costly than constantly
halting for false alarms, so the classifier is tuned with a higher decision
threshold to minimize FPR at the expense of slightly higher FNR.

## Data Source

The data comes from the SUITS DUST 2026 rover simulation. Driving sessions were
recorded as CSV files with timestamped sensor readings at ~4-5 Hz. A separate
labeling pipeline produced ground-truth binary obstacle flags for a subset of
sessions.

## What the Model Does

Given a sliding window of the last 5 timesteps (about 1 second of driving), the
model outputs a probability (0.0 to 1.0) that the rover is currently near an
obstacle. This probability is compared to a configurable threshold to produce a
binary obstacle/clear decision.

The model does NOT:
- Identify which sensor sees the obstacle
- Estimate obstacle distance or size
- Predict future obstacles
- Classify obstacle type

## Why This Architecture

After testing 60+ model configurations across 16 architecture families, a 1D CNN
with residual connections and a window of 5 timesteps emerged as the best model.
CNNs outperformed recurrent models (LSTM, GRU) because:

1. The obstacle signal is primarily in the *current* distance readings — short
   distances mean obstacle. Temporal patterns add only marginal value.
2. With only ~2500 labeled samples across 6 sessions, recurrent models overfit.
3. The 1D convolution naturally captures correlations between neighboring
   timesteps without the vanishing gradient issues of RNNs.
4. The residual connection helps gradient flow through the 3-layer architecture.

## Key Design Decisions

1. **Position augmentation**: X/Y positions are randomly offset by +/-10,000 per
   session during training so the model cannot memorize "obstacles are at
   location (X, Y)". This was critical — without it, models appeared to perform
   well but were actually just learning a map of obstacle locations.

2. **Session-based splits**: Train/val/test splits are done by driving session,
   not by random row sampling. This prevents data leakage from temporally
   correlated readings within the same session.

3. **Distance features only**: The binary obstacle labels are NEVER used as
   input features. Early experiments accidentally included them, which inflated
   accuracy to ~97% but was pure label leakage. The model only sees distances.

4. **Threshold tuning**: The default threshold (0.75) was chosen to minimize
   false positives while keeping false negatives reasonable. See the threshold
   guide in the README for other options.
