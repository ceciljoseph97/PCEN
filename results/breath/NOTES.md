# Breathing session P03 (plan v3, normal breathing), 2026-10-17

Seated person, chest at 1.0 m (person bins found 0.60-1.20 m plus a multipath patch at 2.55-2.85 m); fan behind
the chair at 1.3 m, ~0.2 m to the side. One block fan OFF (R1), one fan ON (R2), 290 s each: 30 s empty, sit
down, 4 min normal breathing. R3/R4 (ABBA repeats) were skipped: fan state is confounded with order.
All 3 Rx valid (no zeros). Script: `scripts/analyze_breath.py` (plan fixed before running), figure contrast.png.
No external breathing reference: the rate at the person bins (SR) is the reference ("oracle range").

| front end | contrast 60 s / 200 s [dB] off | on | person localised off / on | RR agrees with oracle off / on |
|---|---|---|---|---|
| SR (20-s slow-time SR) | 27 / 30 | 26 / 26 | 1.00 / 1.00 | 0.98 / 0.86 |
| CA-CFAR (range) | 10 / 10 | 12 / 10 | 1.00 / 1.00 | 1.00 / 0.84 |
| PCEN tau = 10 s | 2 / 1 | 1 / 0 | 0.58 / 0.49 | 0.49 / 0.30 |
| PCEN tau = 100 s | 5 / 3 | 6 / 2 | 0.77 / 0.70 | 0.58 / 0.51 |
| PCEN tau = inf (empty-room division) | 15 / 16 | 17 / 19 | 1.00 / 1.00 | 0.98 / 0.77 |
| C-PCEN tau = 10 s | 16 / 15 | 18 / 15 | 1.00 / 1.00 | 0.98 / 0.84 |
(PCEN/C-PCEN outputs are compressed with r = 0.5, so their dB values are roughly half of SR's; compare time courses.)

Reading
- Plain PCEN forgets the seated person within ~1 min (contrast 0-2 dB) and then localises the fan or noise
  (fan lock 0.12-0.19), breathing rate correct in only 30-49 % of windows.
- C-PCEN keeps the person for the whole 4 min with the fan running (15-18 dB), localisation 1.00 and breathing
  rate 0.84 / 0.98 - the same as SR and CA-CFAR, not better.
- The fan behind the chair does not capture SR (fan lock 0) but lowers the breathing-rate agreement from ~0.98 to
  ~0.85 for all working front ends.
- Confirms the earlier calibration-session and Walabot results with a clean recording and a running fan:
  C-PCEN fixes PCEN's failure on static persons, but per-cell normalisation offers no gain over slow-time static
  removal for vital signs in this setting.
