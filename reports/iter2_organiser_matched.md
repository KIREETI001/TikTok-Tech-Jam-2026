# Results on the organisers' composition

1200 authentic vs 1200 AI — 6 generators, 200 each.

| | |
|---|---|
| **Final Score** | **0.7170** |
| AUC_clean | 0.7421 |
| AUC_robust | 0.6919 |

| Generator | AUC clean | AUC robust |
|---|---|---|
| ADM | 0.537 | 0.542 |
| DDPM | 0.632 | 0.582 |
| Imagen | 0.663 | 0.647 |
| DDIM | 0.781 | 0.736 |
| VQDM | 0.916 | 0.793 |
| DALLE | 0.923 | 0.850 |

At the checkpoint's operating threshold (0.73): clean FPR 0.0258, clean FNR 0.8083.

| Condition | AUC | FPR | FNR |
|---|---|---|---|
| clean | 0.7421 | 0.0258 | 0.8083 |
| jpeg_q90 | 0.7116 | 0.0275 | 0.8633 |
| jpeg_q70 | 0.6920 | 0.0208 | 0.8933 |
| jpeg_q50 | 0.6505 | 0.0133 | 0.9383 |
| jpeg_q30 | 0.5849 | 0.0083 | 0.9717 |
| blur_sigma0.5 | 0.7786 | 0.0217 | 0.8142 |
| blur_sigma1 | 0.7751 | 0.0275 | 0.8117 |
| blur_sigma2 | 0.6281 | 0.1983 | 0.7108 |
| resize_0.5x | 0.7842 | 0.0392 | 0.7283 |
| resize_0.25x | 0.6775 | 0.0700 | 0.7783 |
| noise_sigma0.02 | 0.6641 | 0.0150 | 0.9308 |
| noise_sigma0.05 | 0.6350 | 0.0050 | 0.9783 |
| noise_sigma0.10 | 0.6459 | 0.0008 | 0.9975 |
| color_jitter_20pct | 0.7557 | 0.0200 | 0.8233 |
| center_crop_80pct | 0.7033 | 0.0233 | 0.8533 |
