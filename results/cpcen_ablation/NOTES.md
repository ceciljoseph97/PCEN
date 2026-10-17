# C-PCEN ablation (post hoc), 2026-10-16

Script: `scripts/cpcen_ablation.py` (one factor at a time; defaults hold 600 s, kappa 6 dB, fast average 0.5 s).
Walabot part in `results/cpcen_ablation_w_full/results.json` (all holds) and `results/cpcen_ablation_w/` (holds >= 5 s).

Why 600 s: chosen before the first C-PCEN evaluation to exceed the longest seated period tested (300 s). Not tuned.
None of our recordings is long enough to reach it (seated phase 300 s, Walabot ~174 s, fan blocks < 2 min),
so every hold >= 300 s gives identical results.

Findings
- Hold must exceed the stillness of the target: simulated 5-s stop Pd 0.12 / 0.33 / 0.88 / 0.94 for hold 1 / 2 / 5 / >=15 s;
  seated contrast after 60 s only with hold >= 60 s (12-14 dB), after 300 s only with hold >= 300 s (10-11 dB);
  Walabot localization 0.03 (5 s) -> 0.12 (60 s) -> 0.32 (120 s) -> 0.43 (>= 300 s).
- New clutter is held as a target for exactly the hold time: fan switched on during operation (fan-off and fan-on
  empty-room recordings joined back to back; synthetic in time) gives false alarms in the fan gate until release
  (t = 1, 2, 5, 15, 30 s for those holds; > 40 s otherwise, +29 dB side / +17 dB front). Plain PCEN: 0.2-0.5 s.
- Gesture detection beside a fan running from the start: unaffected by hold (AUC 0.953).
- kappa 3 dB: holds more (front fan also held, +22 dB); 10 dB: seated contrast after 300 s drops to 3 dB.
- Fast average 1-2 s: Pd at 5 % window FAR beside the fan drops to 0.77 / 0.63 (0.25-0.5 s: 0.93).
- Simulator default scene (fan switches on at 7.5 s): C-PCEN fails for every setting (stopped Pd 0) - the switched-on
  fan is held. Consistent with the earlier result.
- Earlier text said 95 % for the stopped target (3 seeds); with 5 seeds it is 94 %. Paper updated.
