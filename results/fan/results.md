# Fan-background measurement: results

Gates used: front: hand [0.2, 0.65] m, fan [0.85, 1.25] m (fan at 1.0 m); side: hand [0.8, 1.25] m, fan [0.85, 1.25] m (fan at 1.0 m)

## Detection AUC (window level, hand gate)  [95% bootstrap CI]

| Method | front_off | front_on | side_off | side_on |
|---|---|---|---|---|
| sr_global | 0.974 [0.95, 1.00] | 0.994 [0.98, 1.00] | 0.986 [0.96, 1.00] | 0.943 [0.90, 0.98] |
| sr_cacfar | 0.964 [0.92, 0.99] | 0.989 [0.97, 1.00] | 0.988 [0.97, 1.00] | 0.791 [0.71, 0.86] |
| sr_clutmap | 0.977 [0.95, 1.00] | 0.999 [1.00, 1.00] | 0.993 [0.98, 1.00] | 0.955 [0.91, 0.99] |
| sr_pcen1 | 0.973 [0.94, 1.00] | 0.999 [1.00, 1.00] | 0.995 [0.99, 1.00] | 0.955 [0.91, 0.99] |
| sr_pcen013 | 0.966 [0.93, 0.99] | 1.000 [1.00, 1.00] | 0.993 [0.98, 1.00] | 0.955 [0.91, 0.99] |
| sr_pcen1_ca | 0.733 [0.64, 0.82] | 0.844 [0.78, 0.91] | 0.751 [0.67, 0.83] | 0.811 [0.74, 0.87] |

Windows per group: front_off: 60+/91-, front_on: 60+/91-, side_off: 60+/91-, side_on: 60+/91-

## Tests

| Test | ΔAUC | 95% CI | p (one-sided) | note |
|---|---|---|---|---|
| H1_front_on_pcen1_vs_sr_cacfar | +0.011 | [-0.000, +0.028] | 0.0530 | PRIMARY, n.s. (Bonferroni 0.0125) |
| H1_front_on_pcen1_vs_sr_global | +0.005 | [+0.000, +0.015] | 0.0195 | PRIMARY, n.s. (Bonferroni 0.0125) |
| H3_front_on_pcen1_vs_clutmap | +0.000 | [+0.000, +0.000] | 1.0000 | secondary |
| H4_front_on_pcen013_vs_pcen1 | +0.000 | [-0.001, +0.002] | 0.5067 | secondary |
| H1_side_on_pcen1_vs_sr_cacfar | +0.164 | [+0.087, +0.250] | 0.0005 | PRIMARY, significant (Bonferroni 0.0125) |
| H1_side_on_pcen1_vs_sr_global | +0.012 | [-0.017, +0.042] | 0.2159 | PRIMARY, n.s. (Bonferroni 0.0125) |
| H3_side_on_pcen1_vs_clutmap | +0.000 | [-0.002, +0.002] | 0.4703 | secondary |
| H4_side_on_pcen013_vs_pcen1 | -0.000 | [-0.006, +0.006] | 0.5552 | secondary |

## Secondary metrics

| Method | AUC loss front | AUC loss side | fan Pfa excess front [dB] | fan Pfa excess side [dB] | bg-ID front (↓) | bg-ID side (↓) | cross-bg acc front | cross-bg acc side |
|---|---|---|---|---|---|---|---|---|
| sr_global | -0.020 | +0.044 | +21.3 | +29.6 | 0.992 | 1.000 | 0.333 | 0.502 |
| sr_cacfar | -0.025 | +0.197 | +10.1 | +17.9 | 0.912 | 0.999 | 0.466 | 0.479 |
| sr_clutmap | -0.022 | +0.038 | -1.8 | -4.6 | 0.821 | 0.989 | 0.475 | 0.628 |
| sr_pcen1 | -0.027 | +0.040 | -1.0 | -1.7 | 0.812 | 0.962 | 0.657 | 0.572 |
| sr_pcen013 | -0.034 | +0.038 | -0.0 | +2.5 | 0.857 | 0.985 | 0.567 | 0.603 |
| sr_pcen1_ca | -0.111 | -0.060 | -5.6 | -3.4 | 0.716 | 0.927 | 0.589 | 0.651 |