# Feature Engineering

## Overview

The model uses 55 features derived from the raw sensor data. No LiDAR obstacle
labels are ever used as input features — they are targets only.

## Feature Categories

### 1. LiDAR Distance Features (34 features)

For each of the 17 sensors:
- **Raw distance** (`dist_XX_...`): The cleaned distance value in cm. Invalid
  readings (sensor errors, values > 2000 cm) are replaced with `-1.0` sentinel.
- **Valid flag** (`dist_XX_..._valid`): Binary 1.0 if the reading is valid
  (positive distance), 0.0 if no return or sensor error.

These are by far the most important features. The front-center distance
(`dist_02`) alone accounts for ~27% of the model's feature importance.

### 2. IMU Features (5 features)

| Feature | Description |
|---------|-------------|
| `heading` | Raw heading in degrees |
| `pitch` | Raw pitch in degrees |
| `roll` | Raw roll in degrees |
| `heading_sin` | sin(heading) — circular encoding |
| `heading_cos` | cos(heading) — circular encoding |

Heading is encoded as sin/cos to handle the wraparound at 0/360 degrees.

### 3. Position Features (3 features)

| Feature | Description |
|---------|-------------|
| `pos_x` | Rover X (augmented with random offset during training) |
| `pos_y` | Rover Y (augmented with random offset during training) |
| `pos_z` | Rover Z (augmented with random offset during training) |

Position is augmented with per-session random offsets (X/Y: +/-10000, Z: +/-50)
to prevent the model from memorizing obstacle locations on the map.

### 4. Kinematic Features (8 features)

Computed as differences between consecutive timesteps within each session:

| Feature | Description |
|---------|-------------|
| `d_x`, `d_y`, `d_z` | Position deltas (velocity proxy) |
| `d_heading` | Heading change (turn rate with sign) |
| `d_pitch`, `d_roll` | IMU rate of change |
| `speed_xy` | sqrt(d_x^2 + d_y^2) — ground speed magnitude |
| `turn_rate` | abs(d_heading) — unsigned turn rate |

### 5. Aggregate Distance Features (5 features)

| Feature | Description |
|---------|-------------|
| `min_dist` | Minimum valid distance across all 17 sensors |
| `mean_dist` | Mean valid distance across all sensors |
| `n_valid_returns` | Count of sensors with valid readings (0-17) |
| `front_min_dist` | Minimum valid distance across front-facing sensors |
| `rear_min_dist` | Minimum valid distance across rear-facing sensors |

## Feature Importance (XGBoost, Single-Timestep Model)

Top 10 most important features:

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `dist_02_front_center_frame_forward` | 0.274 |
| 2 | `dist_14_front_right_frame_yaw_n20_pitch_n10` | 0.156 |
| 3 | `roll` | 0.088 |
| 4 | `dist_07_left_mid_frame_left_pitch_n20` | 0.064 |
| 5 | `dist_02_front_center_frame_forward_valid` | 0.059 |
| 6 | `dist_08_right_mid_frame_right_pitch_n20` | 0.044 |
| 7 | `dist_04_front_right_wheel_hub_yaw_n30_valid` | 0.031 |
| 8 | `dist_09_rear_left_wheel_hub_back` | 0.027 |
| 9 | `dist_04_front_right_wheel_hub_yaw_n30` | 0.026 |
| 10 | `dist_07_left_mid_frame_left_pitch_n20` | 0.023 |

The front-center LiDAR distance dominates — which makes physical sense, as it
points straight ahead in the direction of travel.

## Preprocessing

All features are standardized using `sklearn.StandardScaler` fitted on the
training set. The scaler is saved in `scaler.pkl` and must be loaded at inference
time. The `ObstacleClassifier` class handles this automatically.
