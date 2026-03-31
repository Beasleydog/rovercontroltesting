# LiDAR Obstacle Classifier

## Best Model: CNN Deep+Residual (Window=5)

A 1D CNN with residual connections that classifies whether the rover is near an
obstacle based on the last 5 LiDAR distance readings + IMU data.

### Performance (test set, session-based split)

| Threshold | FPR   | FNR   | F1    | Precision | Recall |
|-----------|-------|-------|-------|-----------|--------|
| 0.50      | 4.5%  | 4.6%  | 0.936 | 0.919     | 0.954  |
| 0.65      | 2.9%  | 5.4%  | 0.946 | 0.946     | 0.946  |
| **0.75**  | **2.5%** | **6.2%** | **0.946** | **0.953** | **0.939** |
| 0.80      | 1.6%  | 6.9%  | 0.949 | 0.968     | 0.931  |
| 0.85      | 1.2%  | 7.7%  | 0.949 | 0.976     | 0.923  |
| 0.90      | 1.2%  | 8.5%  | 0.944 | 0.975     | 0.915  |

**Default threshold: 0.75** (best balance of low FPR with high recall)

## Quick Start

```python
from predict import ObstacleClassifier

# Load model
clf = ObstacleClassifier('best_model.pt', 'config.json', 'scaler.pkl')

# Adjust threshold (lower = more sensitive, higher = fewer false alarms)
clf.set_threshold(0.75)

# Stream predictions (feed one reading at a time)
result = clf.predict(
    lidar_distances=[500.0, 300.0, ...],  # 17 distance values in cm
    heading=180.0, pitch=3.0, roll=-1.5,  # IMU degrees
    pos_x=0.0, pos_y=0.0, pos_z=0.0,     # rover position
)
# Returns None for first 4 calls (filling window buffer)
# Then: {'obstacle': True/False, 'confidence': 0.82}

# Or batch-process a CSV
results_df = clf.predict_csv('my_lidar_log.csv', threshold=0.80)
```

## Changing the Threshold

```python
clf.set_threshold(0.50)  # balanced, catches everything, some false alarms
clf.set_threshold(0.75)  # recommended, very few false alarms
clf.set_threshold(0.85)  # paranoid, almost zero false alarms, misses ~8%
clf.set_threshold(0.90)  # extreme, zero false alarms on test set
```

Call `clf.set_threshold()` at any time -- it takes effect immediately on the next
`predict()` call. The confidence score is always returned so you can also threshold
externally.

## Files

| File | Description |
|------|-------------|
| `best_model.pt` | PyTorch model weights |
| `config.json` | Model config, feature names, threshold sweep data |
| `scaler.pkl` | StandardScaler fitted on training data |
| `predict.py` | Inference code with `ObstacleClassifier` class |
| `experiment_results.csv` | Full sweep results (60+ architectures tested) |
| `README.md` | This file |

## Input Format

The model expects 17 LiDAR distance readings in cm, in this sensor order:

| # | Sensor | Yaw | Pitch |
|---|--------|-----|-------|
| 00 | front left wheel hub | +30 | 0 |
| 01 | front left frame | +20 | -20 |
| 02 | front center frame | 0 | 0 |
| 03 | front right frame | -20 | -20 |
| 04 | front right wheel hub | -30 | 0 |
| 05 | front left frame | 0 | -25 |
| 06 | front right frame | 0 | -25 |
| 07 | left mid frame | +90 | -20 |
| 08 | right mid frame | -90 | -20 |
| 09 | rear left wheel hub | +140 | 0 |
| 10 | rear left frame | 180 | 0 |
| 11 | rear right frame | 180 | 0 |
| 12 | rear right wheel hub | -140 | 0 |
| 13 | front left frame | +20 | -10 |
| 14 | front right frame | -20 | -10 |
| 15 | front left wheel hub | +15 | 0 |
| 16 | front right wheel hub | -15 | 0 |

Special values: `-1` = no return (max range exceeded), large values = sensor error
(both are automatically handled).

## Dependencies

- Python 3.8+
- PyTorch
- numpy
- scikit-learn (for scaler)
- pandas (only for `predict_csv`)

## Architecture Details

```
CNN_DeepRes(window=5, features=55)
  Conv1d(55 -> 64, kernel=3) + ReLU + BatchNorm
  Conv1d(64 -> 64, kernel=3) + ReLU + BatchNorm + RESIDUAL
  Conv1d(64 -> 128, kernel=3) + ReLU + BatchNorm
  AdaptiveAvgPool1d(1)
  Linear(128 -> 64) + ReLU + Dropout(0.3)
  Linear(64 -> 1)
```

Trained on 4 driving sessions (~1660 windows), validated on 1, tested on 1.
Position data augmented with per-session random offsets to prevent location
memorization.
