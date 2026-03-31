# Lidar Obstacle Beam Classifier Bundle

## Problem Statement
This bundle packages a convolutional neural network for per-beam obstacle-hit classification from rover lidar history.
The task is:
Given a chunk of historical lidar timesteps ending at the current timestep, classify each beam in the current timestep as either hitting an obstacle or not hitting an obstacle.

Key constraint enforced during training:
Historical input chunks never contain the ground-truth obstacle labels. Labels are only targets for the current timestep.

The training pipeline also applies mandatory spatial augmentation to prevent memorizing fixed obstacle locations:
- x offset sampled uniformly from [-10000, 10000]
- y offset sampled uniformly from [-10000, 10000]
- z offset sampled uniformly from [-50, 50]
The same random offset is applied consistently across the full history chunk and current step.

## Data Layout
The raw CSV logs contain rover pose/orientation fields and 17 lidar beam distance readings per timestep.
The labeled CSV exports mirror those files and replace the 17 beam readings with 17 binary `_is_obstacle` columns.

Input features used by the model:
- Lidar beam distances after sanitization/log scaling
- Validity mask for each beam
- Absolute rover pose features (x, y, z) after scaling
- Heading represented as sin/cos
- Pitch and roll
- Pose deltas between consecutive steps
- Static beam index feature

Target:
- 17 binary outputs for the current timestep, one per beam

## Sanitization
Raw lidar values are treated as valid only when they are finite, > 0, and <= 5000 cm. Larger pathological values present in the source logs are zeroed through the validity/sanitization path rather than passed through as real distances.

## Models Explored
The project evaluated multiple model families and training settings:
- Plain temporal-beam CNNs
- Residual CNNs with temporal dilation
- Binary cross-entropy and focal loss
- Context lengths from 1 to 16 timesteps
- Positive-class weighting sweeps
- Threshold sweeps after training
- Additional regularization ideas: lidar noise, beam dropout, timestep dropout, deeper residual stacks

## Overfitting Control
A fresh file-level lockbox split was created and never used during candidate selection.
Candidate models were ranked by 3-fold cross-validation on development files only.
Only the top candidates were evaluated once on the untouched lockbox set.

Files documenting that protocol:
- `reports/lockbox_search/protocol.json`
- `reports/lockbox_search/cv_results.json`
- `reports/lockbox_search/lockbox_results.json`
- `reports/lockbox_search/report.md`

## Final Recommended Model
Selected model: `base_ctx4_focal_pw15`

Architecture/config:
- Model type: residual
- Context length: 4
- Widths/depth descriptor: [64, 64, 64, 64]
- Kernel: (5, 3)
- Dropout: 0.1
- Loss: focal
- Focal gamma: 1.5
- Positive weight scale: 1.5
- Recommended threshold for the exported checkpoint: 0.750
- Threshold used in the earlier untouched lockbox evaluation report: 0.750

Lockbox metrics for the final recommended model:
- Accuracy: 0.9608
- Obstacle precision: 0.6429
- Obstacle recall: 0.5726
- Obstacle F1: 0.6057
- Non-obstacle false positive rate: 0.0177
- Obstacle false negative rate: 0.4274
- TP: 4419
- FP: 2455
- FN: 3298
- TN: 136606

## Important Alternate Model
A more conservative alternate candidate also performed well on the untouched lockbox:
- `ctx4_focal_gamma2`
- Lockbox obstacle F1: 0.6014
- Lockbox obstacle precision: 0.6921
- Lockbox obstacle recall: 0.5317
- Lockbox non-obstacle FPR: 0.0131

## Cross-Validation Ranking Summary
- `ctx4_focal_noise`: cv_mean_obstacle_f1=0.6207, cv_mean_precision=0.6956, cv_mean_recall=0.5623, cv_mean_threshold=0.758
- `base_ctx4_focal_pw15`: cv_mean_obstacle_f1=0.6198, cv_mean_precision=0.7057, cv_mean_recall=0.5538, cv_mean_threshold=0.792
- `ctx4_focal_gamma2`: cv_mean_obstacle_f1=0.6135, cv_mean_precision=0.6668, cv_mean_recall=0.5704, cv_mean_threshold=0.742
- `ctx4_focal_timestepdrop`: cv_mean_obstacle_f1=0.6129, cv_mean_precision=0.6520, cv_mean_recall=0.5796, cv_mean_threshold=0.742
- `ctx4_focal_deeper`: cv_mean_obstacle_f1=0.6033, cv_mean_precision=0.6550, cv_mean_recall=0.5603, cv_mean_threshold=0.767
- `ctx4_focal_noise_beamdrop`: cv_mean_obstacle_f1=0.6005, cv_mean_precision=0.6526, cv_mean_recall=0.5577, cv_mean_threshold=0.767
- `ctx8_focal_pw10`: cv_mean_obstacle_f1=0.5994, cv_mean_precision=0.6955, cv_mean_recall=0.5268, cv_mean_threshold=0.733
- `ctx6_focal_pw15`: cv_mean_obstacle_f1=0.5970, cv_mean_precision=0.6354, cv_mean_recall=0.5630, cv_mean_threshold=0.758
- `ctx4_focal_beamdrop`: cv_mean_obstacle_f1=0.5960, cv_mean_precision=0.6839, cv_mean_recall=0.5299, cv_mean_threshold=0.775

## Threshold Behavior
A threshold sweep was generated for the previously selected best model family to show the precision/recall tradeoff.
Representative points from the sweep:
- threshold=0.500: obs_precision=0.2179, obs_recall=0.7591, obs_f1=0.3386, non_obs_FPR=0.1430, obs_FNR=0.2409
- threshold=0.600: obs_precision=0.3222, obs_recall=0.6595, obs_f1=0.4329, non_obs_FPR=0.0728, obs_FNR=0.3405
- threshold=0.700: obs_precision=0.5030, obs_recall=0.5534, obs_f1=0.5270, non_obs_FPR=0.0287, obs_FNR=0.4466
- threshold=0.750: obs_precision=0.6006, obs_recall=0.5021, obs_f1=0.5469, non_obs_FPR=0.0175, obs_FNR=0.4979
- threshold=0.800: obs_precision=0.7196, obs_recall=0.4358, obs_f1=0.5428, non_obs_FPR=0.0089, obs_FNR=0.5642
- threshold=0.850: obs_precision=0.8364, obs_recall=0.3564, obs_f1=0.4998, non_obs_FPR=0.0037, obs_FNR=0.6436
- threshold=0.900: obs_precision=0.9317, obs_recall=0.2607, obs_f1=0.4074, non_obs_FPR=0.0010, obs_FNR=0.7393

## What Is In This Bundle
- `best_model_checkpoint.pt`: deployable PyTorch checkpoint
- `model_metadata.json`: model config, threshold, feature assumptions, and performance summary
- `run_inference.py`: command-line inference script for raw CSV logs
- `train_lidar_cnn.py`: original broad experiment harness
- `lockbox_ablation_search.py`: stricter lockbox-safe ablation search
- `threshold_sweep_best_model.py`: threshold sweep script
- `reports/`: experiment reports and JSON outputs
- `README.md`: this document

## How To Run Inference
From the bundle directory:
```bash
python run_inference.py --input_csv path/to/raw_log.csv
```
This writes a CSV containing one row per predictable timestep (after enough history is available),
with per-beam obstacle probabilities and binary predictions.

## Notes On Deployment
- The model expects the same raw CSV schema used in the training logs.
- It does not require labeled obstacle columns for inference.
- Predictions begin only after enough context exists. With context length 4, the first 3 timesteps in a file do not produce outputs.
- The recommended threshold is chosen from development data without using the untouched lockbox for selection.

## Limitations
- Dataset class imbalance remains substantial; obstacle hits are relatively rare.
- The model is beam-wise multi-label, not a full geometric scene reconstruction system.
- Performance depends on the sensor layout and CSV schema matching the training data.
- Some source logs contain extreme invalid lidar values; sanitization handles them, but the underlying data quality issue remains worth auditing upstream.

## Reproducibility
Main scripts and result files included in this bundle are sufficient to rerun the experiments and regenerate the selected model, assuming the original dataset zip is available in the same environment.

