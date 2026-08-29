# Results on the organisers' composition

1200 authentic vs 1200 AI — 6 generators, 200 each.

| | |
|---|---|
| **Final Score** | **0.8038** |
| AUC_clean | 0.8264 |
| AUC_robust | 0.7812 |

| Generator | AUC clean | AUC robust |
|---|---|---|
| ADM | 0.645 | 0.658 |
| DDPM | 0.752 | 0.711 |
| Imagen | 0.804 | 0.746 |
| DDIM | 0.859 | 0.814 |
| DALLE | 0.946 | 0.881 |
| VQDM | 0.952 | 0.876 |

At the checkpoint's operating threshold (0.1): clean FPR 0.1175, clean FNR 0.4017.

| Condition | AUC | FPR | FNR |
|---|---|---|---|
| clean | 0.8264 | 0.1175 | 0.4017 |
| jpeg_q90 | 0.8099 | 0.1433 | 0.4125 |
| jpeg_q70 | 0.7842 | 0.1317 | 0.4583 |
| jpeg_q50 | 0.7461 | 0.1217 | 0.5825 |
| jpeg_q30 | 0.7016 | 0.0833 | 0.7317 |
| blur_sigma0.5 | 0.8483 | 0.0858 | 0.4158 |
| blur_sigma1 | 0.8395 | 0.0933 | 0.4492 |
| blur_sigma2 | 0.7256 | 0.3050 | 0.3833 |
| resize_0.5x | 0.8477 | 0.1225 | 0.3775 |
| resize_0.25x | 0.7645 | 0.2950 | 0.3075 |
| noise_sigma0.02 | 0.7636 | 0.1058 | 0.5983 |
| noise_sigma0.05 | 0.7391 | 0.0658 | 0.7500 |
| noise_sigma0.10 | 0.7386 | 0.0350 | 0.8492 |
| color_jitter_20pct | 0.8339 | 0.1042 | 0.4133 |
| center_crop_80pct | 0.7941 | 0.1658 | 0.4042 |
