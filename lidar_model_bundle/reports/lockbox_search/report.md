# Lockbox Ablation Search

Fresh lockbox files were held out during candidate selection. Candidates were ranked using 3-fold CV on development files only.

## CV Ranking
- `ctx4_focal_noise`: cv_f1=0.6207, cv_prec=0.6956, cv_rec=0.5623, cv_thr=0.758
- `base_ctx4_focal_pw15`: cv_f1=0.6198, cv_prec=0.7057, cv_rec=0.5538, cv_thr=0.792
- `ctx4_focal_gamma2`: cv_f1=0.6135, cv_prec=0.6668, cv_rec=0.5704, cv_thr=0.742
- `ctx4_focal_timestepdrop`: cv_f1=0.6129, cv_prec=0.6520, cv_rec=0.5796, cv_thr=0.742
- `ctx4_focal_deeper`: cv_f1=0.6033, cv_prec=0.6550, cv_rec=0.5603, cv_thr=0.767
- `ctx4_focal_noise_beamdrop`: cv_f1=0.6005, cv_prec=0.6526, cv_rec=0.5577, cv_thr=0.767
- `ctx8_focal_pw10`: cv_f1=0.5994, cv_prec=0.6955, cv_rec=0.5268, cv_thr=0.733
- `ctx6_focal_pw15`: cv_f1=0.5970, cv_prec=0.6354, cv_rec=0.5630, cv_thr=0.758
- `ctx4_focal_beamdrop`: cv_f1=0.5960, cv_prec=0.6839, cv_rec=0.5299, cv_thr=0.775

## Lockbox Results
- `base_ctx4_focal_pw15`: lockbox_f1=0.6057, prec=0.6429, rec=0.5726, acc=0.9608, fpr_non=0.0177, fnr_obs=0.4274, fp=2455, fn=3298, threshold=0.750
- `ctx4_focal_gamma2`: lockbox_f1=0.6014, prec=0.6921, rec=0.5317, acc=0.9629, fpr_non=0.0131, fnr_obs=0.4683, fp=1825, fn=3614, threshold=0.775
- `ctx4_focal_noise`: lockbox_f1=0.5803, prec=0.6461, rec=0.5268, acc=0.9599, fpr_non=0.0160, fnr_obs=0.4732, fp=2227, fn=3652, threshold=0.750
