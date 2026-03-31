# Lidar CNN Experiment Report

## Dataset
- Total files: 103
- Total timesteps: 43793
- Beams per timestep: 17
- Overall positive rate: 0.0632
- Split summary: {
  "train": {
    "files": 73,
    "timesteps": 29146,
    "positive_labels": 37164,
    "positive_rate": 0.0750057519748447
  },
  "val": {
    "files": 16,
    "timesteps": 9479,
    "positive_labels": 5558,
    "positive_rate": 0.03449110417455303
  },
  "test": {
    "files": 14,
    "timesteps": 5168,
    "positive_labels": 4351,
    "positive_rate": 0.0495242214532872
  }
}

## Experiments
- `ctx8_plain_bce_pw1`: context=8, widths=(32, 64), k_t=3, k_b=3, dropout=0.1, model=plain, loss=bce, pos_w_scale=1.0, best_epoch=6, thr=0.800
  test_acc=0.9481, obstacle_f1=0.5206, obstacle_precision=0.4874, obstacle_recall=0.5586, fp=2552, fn=1918
- `ctx8_plain_bce_pw05`: context=8, widths=(32, 64), k_t=3, k_b=3, dropout=0.1, model=plain, loss=bce, pos_w_scale=0.5, best_epoch=4, thr=0.725
  test_acc=0.9562, obstacle_f1=0.5199, obstacle_precision=0.5803, obstacle_recall=0.4709, fp=1480, fn=2299
- `ctx8_plain_bce_pw2`: context=8, widths=(32, 64), k_t=3, k_b=3, dropout=0.1, model=plain, loss=bce, pos_w_scale=2.0, best_epoch=4, thr=0.800
  test_acc=0.9419, obstacle_f1=0.4708, obstacle_precision=0.4350, obstacle_recall=0.5130, fp=2895, fn=2116
- `ctx8_residual_bce`: context=8, widths=(64, 64, 64, 64), k_t=3, k_b=3, dropout=0.1, model=residual, loss=bce, pos_w_scale=1.0, best_epoch=1, thr=0.800
  test_acc=0.9460, obstacle_f1=0.4526, obstacle_precision=0.4626, obstacle_recall=0.4430, fp=2236, fn=2420
- `ctx8_residual_focal`: context=8, widths=(64, 64, 64, 64), k_t=3, k_b=3, dropout=0.1, model=residual, loss=focal, pos_w_scale=1.0, best_epoch=9, thr=0.725
  test_acc=0.9593, obstacle_f1=0.5715, obstacle_precision=0.6090, obstacle_recall=0.5383, fp=1502, fn=2006
- `ctx8_residual_focal_pw05`: context=8, widths=(64, 64, 64, 64), k_t=3, k_b=3, dropout=0.1, model=residual, loss=focal, pos_w_scale=0.5, best_epoch=6, thr=0.650
  test_acc=0.9603, obstacle_f1=0.5861, obstacle_precision=0.6182, obstacle_recall=0.5572, fp=1495, fn=1924
- `ctx8_residual_focal_pw2`: context=8, widths=(64, 64, 64, 64), k_t=3, k_b=3, dropout=0.1, model=residual, loss=focal, pos_w_scale=2.0, best_epoch=1, thr=0.675
  test_acc=0.9515, obstacle_f1=0.4912, obstacle_precision=0.5213, obstacle_recall=0.4644, fp=1853, fn=2327
- `ctx8_residual_bce_k5`: context=8, widths=(64, 64, 64, 64), k_t=5, k_b=5, dropout=0.15, model=residual, loss=bce, pos_w_scale=1.0, best_epoch=4, thr=0.800
  test_acc=0.9431, obstacle_f1=0.5349, obstacle_precision=0.4552, obstacle_recall=0.6486, fp=3373, fn=1527
- `ctx12_residual_bce`: context=12, widths=(64, 64, 64, 64), k_t=3, k_b=3, dropout=0.1, model=residual, loss=bce, pos_w_scale=1.0, best_epoch=1, thr=0.800
  test_acc=0.9447, obstacle_f1=0.4846, obstacle_precision=0.4611, obstacle_recall=0.5107, fp=2590, fn=2123
- `ctx12_residual_focal`: context=12, widths=(64, 64, 64, 64), k_t=3, k_b=3, dropout=0.1, model=residual, loss=focal, pos_w_scale=1.0, best_epoch=4, thr=0.650
  test_acc=0.9576, obstacle_f1=0.5624, obstacle_precision=0.5921, obstacle_recall=0.5356, fp=1601, fn=2015
- `ctx10_residual_focal`: context=10, widths=(64, 64, 64, 64), k_t=3, k_b=3, dropout=0.1, model=residual, loss=focal, pos_w_scale=1.0, best_epoch=6, thr=0.700
  test_acc=0.9562, obstacle_f1=0.5590, obstacle_precision=0.5708, obstacle_recall=0.5478, fp=1789, fn=1964
- `ctx12_residual_focal_g15`: context=12, widths=(64, 64, 64, 64), k_t=3, k_b=3, dropout=0.1, model=residual, loss=focal, pos_w_scale=1.0, best_epoch=4, thr=0.750
  test_acc=0.9590, obstacle_f1=0.5747, obstacle_precision=0.6088, obstacle_recall=0.5441, fp=1517, fn=1978
- `ctx12_residual_focal_g25`: context=12, widths=(64, 64, 64, 64), k_t=3, k_b=3, dropout=0.1, model=residual, loss=focal, pos_w_scale=1.0, best_epoch=4, thr=0.700
  test_acc=0.9592, obstacle_f1=0.5710, obstacle_precision=0.6144, obstacle_recall=0.5333, fp=1452, fn=2025
- `ctx12_residual_focal_pw15`: context=12, widths=(64, 64, 64, 64), k_t=3, k_b=3, dropout=0.1, model=residual, loss=focal, pos_w_scale=1.5, best_epoch=10, thr=0.750
  test_acc=0.9565, obstacle_f1=0.5596, obstacle_precision=0.5775, obstacle_recall=0.5428, fp=1723, fn=1984
- `ctx16_residual_focal`: context=16, widths=(64, 64, 64, 64), k_t=3, k_b=3, dropout=0.15, model=residual, loss=focal, pos_w_scale=1.0, best_epoch=2, thr=0.675
  test_acc=0.9541, obstacle_f1=0.5293, obstacle_precision=0.5588, obstacle_recall=0.5028, fp=1717, fn=2151
- `ctx4_residual_focal_pw15`: context=4, widths=(64, 64, 64, 64), k_t=5, k_b=3, dropout=0.1, model=residual, loss=focal, pos_w_scale=1.5, best_epoch=7, thr=0.800
  test_acc=0.9620, obstacle_f1=0.5990, obstacle_precision=0.6315, obstacle_recall=0.5697, fp=1445, fn=1870
