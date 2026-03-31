"""
Obstacle Classifier — Inference Script
=======================================
Loads the trained CNN model and classifies LiDAR readings as obstacle/clear.

Usage:
    from predict import ObstacleClassifier

    clf = ObstacleClassifier('best_model.pt', 'config.json', 'scaler.pkl')
    clf.set_threshold(0.75)  # adjust as needed

    # Feed readings one at a time (it buffers the window internally)
    result = clf.predict(lidar_distances, heading, pitch, roll, pos_x, pos_y, pos_z)
    # result = {'obstacle': True/False, 'confidence': 0.0-1.0}

    # Or classify a full CSV file
    results_df = clf.predict_csv('my_lidar_log.csv')
"""

import json, pickle, warnings
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore', message='X does not have valid feature names')


class CNN_DeepRes(nn.Module):
    """3-layer CNN with residual connection (best architecture from sweep)."""
    def __init__(self, nf, w, ch=64):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv1d(nf, ch, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(ch))
        self.c2 = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(ch))
        self.c3 = nn.Sequential(nn.Conv1d(ch, ch*2, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(ch*2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(ch*2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.c1(x)
        x = self.c2(x) + x  # residual
        x = self.c3(x)
        return self.fc(self.pool(x).squeeze(-1)).squeeze(-1)


class ObstacleClassifier:
    def __init__(self, model_path='best_model.pt', config_path='config.json',
                 scaler_path='scaler.pkl', device=None):
        # Load config
        with open(config_path) as f:
            self.config = json.load(f)

        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        self.window_size = self.config['window_size']
        n_feat = self.config['n_features']
        self.model = CNN_DeepRes(n_feat, self.window_size)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        # Threshold
        self.threshold = self.config.get('default_threshold', 0.75)

        # Feature config
        self.dist_cols = self.config['dist_cols']
        self.feature_names = self.config['feature_names']
        self.sentinel = self.config['sentinel_value']
        self.max_range = self.config['max_valid_range_cm']

        # Rolling buffer for streaming predictions
        self.buffer = []
        self._prev_row = None

    def set_threshold(self, threshold):
        """
        Set the classification threshold (0.0 - 1.0).

        Lower  = more sensitive, catches more obstacles but more false alarms.
        Higher = more conservative, fewer false alarms but may miss some obstacles.

        Guide (from validation):
            0.50 — FPR ~4.5%, FNR ~4.6%  (balanced)
            0.65 — FPR ~2.9%, FNR ~5.4%  (good balance)
            0.75 — FPR ~2.5%, FNR ~6.2%  (recommended default)
            0.80 — FPR ~1.6%, FNR ~6.9%  (low false positives)
            0.85 — FPR ~1.2%, FNR ~7.7%  (very few false positives)
            0.90 — FPR ~1.2%, FNR ~8.5%  (almost no false positives)
        """
        self.threshold = threshold

    def _clean_distances(self, distances):
        """Replace invalid distance readings with sentinel value."""
        cleaned = np.array(distances, dtype=float)
        invalid = (np.abs(cleaned) > self.max_range) | (cleaned == -1)
        cleaned[invalid] = self.sentinel
        return cleaned

    def _make_feature_row(self, lidar_dists, heading, pitch, roll, pos_x, pos_y, pos_z):
        """Build a single feature vector from raw sensor readings."""
        dists = self._clean_distances(lidar_dists)
        feat = {}

        # Distance features
        for i, c in enumerate(self.dist_cols):
            short = c.replace('lidar_', 'dist_').replace('_cm', '')
            feat[short] = dists[i]
            feat[short + '_valid'] = float(dists[i] > 0)

        # IMU
        feat['heading'] = heading
        feat['pitch'] = pitch
        feat['roll'] = roll
        feat['heading_sin'] = np.sin(np.radians(heading))
        feat['heading_cos'] = np.cos(np.radians(heading))

        # Position
        feat['pos_x'] = pos_x
        feat['pos_y'] = pos_y
        feat['pos_z'] = pos_z

        # Kinematics (diff from previous)
        if self._prev_row is not None:
            feat['d_x'] = pos_x - self._prev_row['pos_x']
            feat['d_y'] = pos_y - self._prev_row['pos_y']
            feat['d_z'] = pos_z - self._prev_row['pos_z']
            feat['d_heading'] = heading - self._prev_row['heading']
            feat['d_pitch'] = pitch - self._prev_row['pitch']
            feat['d_roll'] = roll - self._prev_row['roll']
        else:
            feat['d_x'] = feat['d_y'] = feat['d_z'] = 0.0
            feat['d_heading'] = feat['d_pitch'] = feat['d_roll'] = 0.0

        feat['speed_xy'] = np.sqrt(feat['d_x']**2 + feat['d_y']**2)
        feat['turn_rate'] = abs(feat['d_heading'])

        # Distance-derived
        valid_d = [d for d in dists if d > 0]
        feat['min_dist'] = min(valid_d) if valid_d else -1
        feat['mean_dist'] = np.mean(valid_d) if valid_d else -1
        feat['n_valid_returns'] = len(valid_d)

        front_d = [dists[i] for i, c in enumerate(self.dist_cols) if 'front' in c and dists[i] > 0]
        rear_d = [dists[i] for i, c in enumerate(self.dist_cols) if 'rear' in c and dists[i] > 0]
        feat['front_min_dist'] = min(front_d) if front_d else -1
        feat['rear_min_dist'] = min(rear_d) if rear_d else -1

        self._prev_row = feat.copy()

        # Build vector in correct order
        return np.array([feat[fn] for fn in self.feature_names], dtype=float)

    def predict(self, lidar_distances, heading, pitch, roll, pos_x, pos_y, pos_z):
        """
        Classify a single timestep.

        Args:
            lidar_distances: list/array of 17 distance readings in cm
            heading, pitch, roll: IMU readings in degrees
            pos_x, pos_y, pos_z: rover position

        Returns:
            dict with 'obstacle' (bool), 'confidence' (float 0-1)
            Returns None if buffer not yet full (need window_size readings first)
        """
        row = self._make_feature_row(lidar_distances, heading, pitch, roll, pos_x, pos_y, pos_z)
        scaled = self.scaler.transform(row.reshape(1, -1)).flatten()
        self.buffer.append(scaled)

        # Keep buffer at window size
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]

        if len(self.buffer) < self.window_size:
            return None  # not enough history yet

        window = np.array(self.buffer)  # (W, features)
        with torch.no_grad():
            x = torch.FloatTensor(window).unsqueeze(0).to(self.device)
            confidence = torch.sigmoid(self.model(x)).cpu().item()

        return {
            'obstacle': confidence >= self.threshold,
            'confidence': confidence,
        }

    def predict_csv(self, csv_path, threshold=None):
        """
        Run predictions on a full CSV log file.

        Args:
            csv_path: path to CSV with distance columns + IMU + position
            threshold: override threshold (or use self.threshold)

        Returns:
            pandas DataFrame with original data + 'obstacle_prob' and 'obstacle_pred' columns
        """
        import pandas as pd

        if threshold is not None:
            old_thresh = self.threshold
            self.threshold = threshold

        df = pd.read_csv(csv_path)

        # Find distance columns
        csv_dist_cols = sorted([c for c in df.columns if 'lidar' in c and c.endswith('_cm')])
        if not csv_dist_cols:
            raise ValueError("CSV must have lidar distance columns ending in '_cm'")

        self.reset()

        predictions = []
        for _, row in df.iterrows():
            dists = [row[c] for c in csv_dist_cols]
            result = self.predict(
                dists, row['heading'], row['pitch'], row['roll'],
                row['rover_pos_x'], row['rover_pos_y'], row['rover_pos_z']
            )
            if result is None:
                predictions.append({'obstacle_prob': None, 'obstacle_pred': None})
            else:
                predictions.append({
                    'obstacle_prob': result['confidence'],
                    'obstacle_pred': result['obstacle'],
                })

        pred_df = pd.DataFrame(predictions)
        result = pd.concat([df, pred_df], axis=1)

        if threshold is not None:
            self.threshold = old_thresh

        return result

    def reset(self):
        """Clear the rolling buffer (call between sessions)."""
        self.buffer = []
        self._prev_row = None


if __name__ == '__main__':
    import sys, os

    # Quick self-test
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clf = ObstacleClassifier(
        os.path.join(script_dir, 'best_model.pt'),
        os.path.join(script_dir, 'config.json'),
        os.path.join(script_dir, 'scaler.pkl'),
    )

    print(f"Model loaded: {clf.config['model_name']}")
    print(f"Window size: {clf.window_size}")
    print(f"Threshold: {clf.threshold}")
    print(f"Features: {clf.config['n_features']}")
    print(f"Device: {clf.device}")
    print()

    # Test with dummy data
    for i in range(clf.window_size + 3):
        dists = [500.0] * 17  # all 5 meters away
        result = clf.predict(dists, 180.0, 3.0, -1.5, 0.0, 0.0, 0.0)
        if result:
            print(f"  Step {i}: obstacle={result['obstacle']}, confidence={result['confidence']:.3f}")

    print("\nTest with close obstacle:")
    clf.reset()
    for i in range(clf.window_size + 3):
        dists = [50.0] * 17  # all very close
        result = clf.predict(dists, 180.0, 3.0, -1.5, 0.0, 0.0, 0.0)
        if result:
            print(f"  Step {i}: obstacle={result['obstacle']}, confidence={result['confidence']:.3f}")

    print("\n✓ Self-test passed")
