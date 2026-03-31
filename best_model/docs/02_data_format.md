# Data Format

## Raw CSV Files

Each driving session produces a CSV with one row per timestep (~4-5 Hz).

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `iso_time_utc` | string | UTC timestamp (ISO 8601) |
| `elapsed_s` | float | Seconds since session start |
| `step_idx` | int | Sequential step counter |
| `rover_pos_x` | float | Rover X position (simulation units) |
| `rover_pos_y` | float | Rover Y position (simulation units) |
| `rover_pos_z` | float | Rover Z position (simulation units) |
| `heading` | float | Rover heading in degrees |
| `pitch` | float | Rover pitch in degrees |
| `roll` | float | Rover roll in degrees |
| `lidar_00_..._cm` | float | Distance reading from sensor 00 in cm |
| ... | ... | (17 sensor columns total) |
| `lidar_16_..._cm` | float | Distance reading from sensor 16 in cm |

### Distance Values

| Value | Meaning |
|-------|---------|
| Positive float (e.g. `523.4`) | Valid distance reading in centimeters |
| `-1` | No return — beam went to max range without hitting anything |
| Very large magnitude (e.g. `-5.7e28`) | Sensor error / invalid reading |

The classifier automatically handles `-1` and invalid values by replacing them
with a sentinel value of `-1.0`.

### Labeled Files

A separate set of CSVs in `labeled_obstacles_liveexport/` contains the same
structure but with `_is_obstacle` suffix columns instead of `_cm`:

| Column | Type | Description |
|--------|------|-------------|
| `lidar_00_..._cm_is_obstacle` | int (0/1) | Whether sensor 00 detected an obstacle |
| ... | ... | (17 binary label columns) |

These are the ground-truth labels used for training. Only 6 sessions have both
distance readings AND labels.

## Sensor Layout

The 17 LiDAR sensors are mounted at fixed positions on the rover frame:

```
          FRONT
   [15] [00]   [02]   [04] [16]
     [01] [13]   [14] [03]
        [05]       [06]

   [07] ---- ROVER ---- [08]

        [09]       [12]
          [10] [11]
          REAR
```

### Sensor Details

| # | Name | Position | Yaw | Pitch | Purpose |
|---|------|----------|-----|-------|---------|
| 00 | front left wheel hub | front-left | +30 | 0 | Wide-angle left forward |
| 01 | front left frame | front-left | +20 | -20 | Downward-angled left |
| 02 | front center frame | front-center | 0 | 0 | Straight ahead (most important) |
| 03 | front right frame | front-right | -20 | -20 | Downward-angled right |
| 04 | front right wheel hub | front-right | -30 | 0 | Wide-angle right forward |
| 05 | front left frame | front-left | 0 | -25 | Ground-facing left |
| 06 | front right frame | front-right | 0 | -25 | Ground-facing right |
| 07 | left mid frame | mid-left | +90 | -20 | Side-facing left |
| 08 | right mid frame | mid-right | -90 | -20 | Side-facing right |
| 09 | rear left wheel hub | rear-left | +140 | 0 | Rear-left diagonal |
| 10 | rear left frame | rear | 180 | 0 | Straight back left |
| 11 | rear right frame | rear | 180 | 0 | Straight back right |
| 12 | rear right wheel hub | rear-right | -140 | 0 | Rear-right diagonal |
| 13 | front left frame | front-left | +20 | -10 | Shallow-angled left |
| 14 | front right frame | front-right | -20 | -10 | Shallow-angled right |
| 15 | front left wheel hub | front-left | +15 | 0 | Narrow-angle left forward |
| 16 | front right wheel hub | front-right | -15 | 0 | Narrow-angle right forward |

Sensor origin offsets (in cm from rover center) and exact angles are stored in
the `manifest.json` from the labeling pipeline.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total labeled sessions | 6 |
| Total labeled samples | 2,569 |
| Obstacle rate (any sensor) | 40.1% |
| Obstacle rate (front sensors) | 32.3% |
| Sampling rate | ~4-5 Hz |
| Valid distance range | 0 - 2000 cm |
| Most active sensor | front center (02) |
| Least active sensor | front left wheel hub (00) |

## Unlabeled Data

An additional 48 CSV files contain distance readings WITHOUT obstacle labels.
These were recorded during other driving sessions and could be used for:
- Unsupervised pre-training
- Semi-supervised learning
- Data augmentation
- Testing the classifier on new data (without ground truth)
