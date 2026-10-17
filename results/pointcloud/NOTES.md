# Point-cloud / dynamic-trail study (post hoc), 2026-10-17

Reference implementation: rsazid99/inf_radarpoint (radar_data/utils.py, Apache-2.0), re-implemented offline in
`pcen_radar/pointcloud.py` for our configuration. Spectra, angles (Rx phase differences, L-shaped array) and gates
(FOV +-45 deg, min range 0.2 m) are identical for all front ends; only the detection stage is exchanged.
Scripts: `scripts/pointcloud_study.py` (fan session), `scripts/kit_pointcloud.py` (KIT, 3-D),
`scripts/pointcloud_figs.py`, `scripts/kit_pointcloud_figs.py`.

## Issues found in the reference code
1. `CFAR_2D` thresholds the complex Doppler spectrum. numpy compares complex numbers lexicographically
   (real part first), so the test is not a power CFAR. Kept only in the "published" mode; fixed (power) otherwise.
2. The velocity axis uses `T_chirp = num_samples / fs` (sampling time) instead of the chirp repetition time
   (0.59 ms in its config), so velocities are too large by about 4.6x.
3. The SNR gate (|Rx3| / (median|Rx3|/0.6745) > 8) is a global amplitude threshold on top of CFAR.

## Data problem in our fan session (affects the paper's fan section)
- Rx1 is all zeros in all 16 recordings; Rx2/Rx3 have 68-74 % samples exactly 0 (values 0..~300 of 4095).
- For comparison, KIT BGT60TR13C data (same chip) sit at mid-scale (mean 0.44-0.51, no zeros).
- Probable cause: `record_fan_session.py` stored `clip(round(frames*4095), 0, 65535)` as uint16 - values < 0 were cut.
  Not proven: run `scripts/check_adc.py` on the live radar (prints min/negative/zero share per Rx).
- Fixed recorder: signed int16 without clipping + ADC warning; loader reads both formats.
- Consequence here: no azimuth (needs Rx1), detection on Rx2+Rx3 only, the published detector (AND over 3 Rx)
  returns 0 points. In the fan data every gesture produces points over the whole range 0.2-2 m with PCEN /
  clutter map (front gestures: more points beyond 1.3 m than in the hand gate); the clean KIT data do not show
  this, which points to the clipping. The fan-session numbers below are therefore indicative only.

## Fan session (Rx2+Rx3, threshold per front end at 0.1 false points/frame in the empty room, fan off)
| front end | fan points/frame (empty, fan on, side) | side+fan on: Pd / FA window | trail pts/gesture |
|---|---|---|---|
| published | 0 | 0 / 0 | 0 |
| CA-CFAR | 0.000 | 0.20 / 0.05 | 0 |
| global | 9.85 | 1.00 / 1.00 | 324 |
| clutter map | 0.000 | 0.97 / 0.10 | 29 |
| PCEN | 0.000 | 0.97 / 0.10 | 34 |
| C-PCEN | 0.000 | 0.95 / 0.11 | 39 |
| PCEN + CA | 0.001 | 0.98 / 0.48 | 26 |
CA-CFAR needs a very high threshold to keep 0.1 false points/frame (residual static lines are local peaks),
so it loses the hand. C-PCEN keeps the standing participant in the hand gate (FA windows 0.44-0.60 at the
fan-off/front positions), which is its intended behaviour but not what a gesture trail wants.

## KIT, 3-D (all 3 Rx; threshold at 0.1 points/frame in setup-1 no-gesture recordings; tested on 4v1 window / 7v1 free space)
| front end | hand pts/frame | frames with hand point | window background pts/frame (4v1) | no-gesture pts/frame |
|---|---|---|---|---|
| published (uncalibrated) | 2.4 | 0.54 | 0.45 | 1.5-1.9 |
| CA-CFAR | 0.26-0.29 | 0.16-0.17 | 0.055 | 0.17-0.20 |
| global | 0.62 | 0.13 | 0.014 | 0.39-0.43 |
| clutter map | 0.47-0.69 | 0.06-0.08 | 0.10 | 0.25-0.85 |
| PCEN | 0.49-0.72 | 0.06-0.09 | 0.10 | 0.26-0.85 |
| C-PCEN | 2.3-2.5 | 0.20-0.28 | 0.10 | 1.7-3.8 |
| PCEN + CA | 0.30-0.43 | 0.09-0.13 | 0.012 | 0.13-0.48 |
Reading: with static backgrounds PCEN gives no advantage in point counts; it keeps only what changes, so the
trail shows the motion (e.g. the arm approaching from 1.1 m to 0.25 m in GM_2_4v1_..._2360) and drops a hand
that then holds still, while CA-CFAR / global give fragments and the published detector also keeps the static
hand and many false points. C-PCEN keeps the approach and the held hand, at the cost of many points in
no-gesture recordings (a person standing still). KIT has no per-frame gesture timing, so these are counts
over whole recordings; 64-chirp frames are cut from the KIT chirp stream and may straddle its 6-ms frame gaps.

## Correction (session 2, 2026-10-17)
With clean session-2 data (all 3 Rx, no clipping) PCEN / clutter map also show range stripes at each gesture
(0.2-0.9 m), so the stripes are not only a clipping artefact (arm/body motion and range sidelobes, amplified by
the per-cell normalization). See results/fan_v2/NOTES.md and results/pointcloud_v2.
