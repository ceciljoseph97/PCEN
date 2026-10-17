# Range-gate decision for the fan session P01 (written 2026-10-16, before any method comparison was computed)

**Scene deviation.** The fan was placed at **0.8 m** (tape, radar → fan hub) instead of the planned 1.0 m, because of space limits in the room. Everything else followed the protocol: 16 blocks, 20 repetitions each, 0 dropped frames. The measured frame period was 48.5 ms instead of the nominal 50 ms, and this is used for the PCEN and clutter-map time constants.

**Where the radar sees the fan.** These checks use only static-removed power and are the same for every method. Source: `scripts/fan_check.py`, comparing the empty fan-OFF and fan-ON blocks.
- The strongest fan-ON increase is at **0.97–1.05 m** on the radar range axis (+19.7 dB). A second peak at 1.87 m is the double-bounce echo, at twice the range.
- The hand at 0.4 m (tape) appears at about 0.45–0.5 m. The radar range axis therefore reads 0.1–0.2 m more than the tape distances (internal delay, and the fan's housing behind the hub).
- The fan also raises moving power by about 10 dB at **all** ranges, likely through vibration or airflow moving the static clutter. Its strength differed between round B (front) and round C (side), so the fan placement was probably touched when the position changed. This is reported as an observation.

**Decision.** The gates live on the radar range axis, so the pre-registered rule (`pcen_radar/fan_scene.py`) is applied with the fan distance **as the radar measures it: r_f = 1.0 m**. That gives exactly the pre-registered gates:
- fan: 0.85–1.25 m
- side hand: 0.80–1.25 m
- front hand: 0.20–0.65 m

**Sensitivity analysis.** The same rule is applied with the tape distance, r_f = 0.8 m:
- fan: 0.65–1.05 m
- side hand: 0.60–1.05 m
- front hand: 0.20–0.60 m

Both are reported. The primary conclusion is drawn from r_f = 1.0 m.

## Addendum (written after the primary results; observation only, decisions unchanged)
- In the averaged maps the side-position hand appears at 0.67–0.82 m on the radar axis. The hand therefore reads close to its tape distance. The 1.0 m fan echo comes from the fan's housing behind the hub rather than from a uniform range offset.
- The primary side hand gate (0.80–1.25 m) covers only the upper part of the hand. The tape-distance gate (0.60–1.05 m) covers it better.
- Both were run. The conclusions agree. With the tape-distance gate, side, fan on, SR + PCEN vs SR + CA-CFAR: ΔAUC +0.057 [0.016, 0.101], p = 0.0015, significant; vs SR + global: +0.022, p = 0.051, not significant. The fan false-alarm increase pattern is the same. See `results/fan_sens_tape080/`.
