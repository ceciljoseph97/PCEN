# Fan session 2 (P02), 2026-10-17

Recorded with the fixed recorder (signed int16, all 3 Rx valid; ADC check: Rx1 100 % negative values, Rx2/3
~70 % negative -> the session-1 storage had clipped them). Plan v2: fan 1.0 m; hand "front" 0.4 m and "near"
0.8 m directly in front of the fan; push-pull, circle, hold; ABBA over fan state; 20 reps; frame period 46.3 ms.
Analysis: `scripts/analyze_fan_v2.py` (tests fixed before looking at results), `scripts/hold_posthoc.py`
(exploratory), `scripts/pointcloud_v2.py`, `scripts/pointcloud_v2_figs.py`.

## Moving gestures (push-pull + circle), window AUC
| | front off | front on | near off | near on |
|---|---|---|---|---|
| SR + global | 0.938 | 0.958 | 0.979 | 0.954 |
| SR + CA-CFAR | 0.969 | 0.945 | 0.991 | 0.959 |
| SR + clutter map | 0.968 | 0.977 | 1.000 | 0.960 |
| SR + PCEN (1 s) | 0.943 | 0.972 | 1.000 | 0.967 |
| SR + C-PCEN | 0.941 | 0.971 | 0.994 | 0.962 |
T1 (near, fan on): PCEN vs CA-CFAR +0.008 [-0.004, +0.025], p = 0.17; vs global +0.013 [-0.009, +0.041], p = 0.17.
Not significant: with clean data and the hand in front of the fan, all front ends detect the gestures equally well.
The large session-1 gap (PCEN 0.955 vs CA-CFAR 0.791, hand beside the fan) is not reproduced; that geometry was
not repeated and those data were clipped.

## Fan false-alarm increase, empty room [dB] (threshold for Pfa 1e-3 with the fan off)
front / near: SR + global +29 / +30, SR + CA-CFAR +25 / +25, SR + clutter map +8 / +7, SR + PCEN +10 / +10,
SR + PCEN (0.13 s) +13 / +13, SR + PCEN + CA-CFAR +8 / +8, SR + C-PCEN +10 / +10.
PCEN and clutter map cut the fan's false alarms by 15-22 dB relative to CA-CFAR and the global threshold, but
not to the fan-off level (session 1, clipped: -1 / -2 dB).

## Hold (still hand)
Pre-stated test T2 failed by design: the negative window (4.0-4.9 s) contains the lowering movement, so all
AUCs are < 0.5 at front/off. Exploratory re-analysis (hand down 5.0-5.9 s, zero-Doppler power in the hand
gate): AUC 0.36-0.54 for every front end, including the global threshold without static removal - the still
hand is not distinguishable from the scene in these recordings, and the C-PCEN gate never fired (C-PCEN
identical to PCEN). No conclusion about C-PCEN can be drawn from the hold blocks.

## Point cloud, all 3 Rx (threshold per front end: 0.1 false points/frame, empty room, fan off)
Empty room, fan on, points/frame in the fan gate: published 12-14, global 4.7-5.0, CA-CFAR / clutter map /
PCEN / C-PCEN 0.00.  Push-pull 0.8 m in front of the running fan: PCEN Pd 0.95, FA 0.05, 178 hand points per
gesture window; clutter map 0.95 / 0.05 / 170; C-PCEN 0.95 / 0.15 / 558; CA-CFAR 0.85 / 0.05 / 3; global and
published 1.00 / 1.00 (the fan fills the hand gate). Circle: CA-CFAR Pd 0.50, PCEN 1.00 (FA 0.25).
Correction to the session-1 note: the full-range stripes at each gesture also appear with clean data
(0.2-0.9 m, from arm/body motion and range sidelobes, amplified by the per-cell normalization); they are not
only a clipping artefact.
