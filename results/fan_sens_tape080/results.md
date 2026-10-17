# Fan-background measurement: results

Gates used: front: hand [0.2, 0.6] m, fan [0.65, 1.05] m (fan at 0.8 m); side: hand [0.6, 1.05] m, fan [0.65, 1.05] m (fan at 0.8 m)

## Detection AUC (window level, hand gate)  [95% bootstrap CI]

| Method | front_off | front_on | side_off | side_on |
|---|---|---|---|---|
| sr_global | 0.974 [0.95, 1.00] | 0.994 [0.98, 1.00] | 0.992 [0.98, 1.00] | 0.963 [0.93, 0.99] |
| sr_cacfar | 0.964 [0.92, 0.99] | 0.989 [0.97, 1.00] | 0.995 [0.99, 1.00] | 0.928 [0.88, 0.96] |
| sr_clutmap | 0.977 [0.95, 1.00] | 0.999 [1.00, 1.00] | 0.997 [0.99, 1.00] | 0.985 [0.96, 1.00] |
| sr_pcen1 | 0.973 [0.94, 1.00] | 0.999 [1.00, 1.00] | 0.998 [0.99, 1.00] | 0.985 [0.96, 1.00] |
| sr_pcen013 | 0.966 [0.93, 0.99] | 1.000 [1.00, 1.00] | 0.999 [0.99, 1.00] | 0.977 [0.95, 1.00] |
| sr_pcen1_ca | 0.733 [0.64, 0.82] | 0.844 [0.78, 0.91] | 0.758 [0.68, 0.83] | 0.864 [0.80, 0.92] |

Windows per group: front_off: 60+/91-, front_on: 60+/91-, side_off: 60+/91-, side_on: 60+/91-

## Tests

| Test | ΔAUC | 95% CI | p (one-sided) | note |
|---|---|---|---|---|
| H1_front_on_pcen1_vs_sr_cacfar | +0.011 | [-0.000, +0.028] | 0.0530 | PRIMARY, n.s. (Bonferroni 0.0125) |
| H1_front_on_pcen1_vs_sr_global | +0.005 | [+0.000, +0.015] | 0.0195 | PRIMARY, n.s. (Bonferroni 0.0125) |
| H3_front_on_pcen1_vs_clutmap | +0.000 | [+0.000, +0.000] | 1.0000 | secondary |
| H4_front_on_pcen013_vs_pcen1 | +0.000 | [-0.001, +0.002] | 0.5067 | secondary |
| H1_side_on_pcen1_vs_sr_cacfar | +0.057 | [+0.016, +0.101] | 0.0015 | PRIMARY, significant (Bonferroni 0.0125) |
| H1_side_on_pcen1_vs_sr_global | +0.022 | [-0.003, +0.055] | 0.0510 | PRIMARY, n.s. (Bonferroni 0.0125) |
| H3_side_on_pcen1_vs_clutmap | +0.000 | [-0.003, +0.003] | 0.5237 | secondary |
| H4_side_on_pcen013_vs_pcen1 | -0.008 | [-0.023, +0.003] | 0.9040 | secondary |

## Secondary metrics

| Method | AUC loss front | AUC loss side | fan Pfa excess front [dB] | fan Pfa excess side [dB] | bg-ID front (↓) | bg-ID side (↓) | cross-bg acc front | cross-bg acc side |
|---|---|---|---|---|---|---|---|---|
| sr_global | -0.020 | +0.029 | +20.6 | +29.3 | 0.992 | 1.000 | 0.333 | 0.501 |
| sr_cacfar | -0.025 | +0.066 | +9.0 | +17.0 | 0.912 | 0.998 | 0.466 | 0.504 |
| sr_clutmap | -0.022 | +0.012 | -1.4 | -4.9 | 0.821 | 0.980 | 0.475 | 0.638 |
| sr_pcen1 | -0.027 | +0.013 | -0.6 | -2.2 | 0.812 | 0.949 | 0.657 | 0.602 |
| sr_pcen013 | -0.034 | +0.022 | +0.6 | +1.7 | 0.857 | 0.981 | 0.567 | 0.632 |
| sr_pcen1_ca | -0.111 | -0.106 | -7.3 | -4.4 | 0.716 | 0.945 | 0.589 | 0.611 |