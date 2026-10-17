| Method | Pd@1e-2 | Pd@1e-3 | Pd@1e-4 | Worst-region Pfa excess [dB] | Pd moving | Pd stopped | ms/frame |
|---|---|---|---|---|---|---|---|
| Raw (global thr) | 0.000 | 0.000 | 0.000 | +11.6 | 0.120 | 0.000 | 0.00 |
| Static removal (global) | 1.000 | 0.290 | 0.040 | +4.1 | 0.295 | 0.000 | 0.00 |
| Static removal + CA-CFAR | 0.930 | 0.700 | 0.525 | +5.7 | 0.710 | 0.720 | 2.87 |
| Static removal + OS-CFAR | 0.795 | 0.655 | 0.550 | +4.0 | 0.660 | 0.000 | 19.11 |
| Clutter-map CFAR | 1.000 | 0.805 | 0.005 | +3.6 | 0.815 | 0.010 | 0.03 |
| PCEN (raw) | 1.000 | 0.785 | 0.010 | +3.7 | 0.800 | 0.010 | 0.21 |
| Static removal + PCEN | 1.000 | 0.805 | 0.010 | +3.6 | 0.850 | 0.000 | 0.20 |
| Static removal + PCEN + CA-CFAR | 1.000 | 0.975 | 0.405 | +4.1 | 0.985 | 0.050 | 2.78 |

Worst-region Pfa excess = 10·log10(max regional Pfa / 0.001) at the threshold giving global Pfa=0.001. 'Pd moving/stopped' use that same threshold.