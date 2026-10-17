# Gated (adaptive) PCEN: design history and results (2026-10-16, post hoc)

Why: plain PCEN absorbs any target that stays in a cell longer than about tau. Examples: a seated, breathing person (RepMon) and a target that stops (simulator).

## Design steps (all on synthetic toy signals; real data was used only for the final evaluation, except where noted)
1. Smaller update rate while E/M >= 6 dB, frame by frame: failed. Single-frame fluctuations reopened the gate.
2. Explicit hold time with a slow leak: failed. Same fluctuation problem.
3. Two smoothers, with the held background following the fast one: never triggered. An EMA cannot jump 6 dB in one frame. This version WAS run once on the real data (simulator, BGT60 calibration session). Result: identical to plain PCEN on the calibration session.
4. Final, censored update (as in censored clutter maps):
   - a 0.5-s energy average A decides per cell and frame;
   - if A >= 6 dB above M: M is held (for at most 600 s, then released);
   - otherwise: plain PCEN update.
   - Parameters were fixed before the final run.

## Results (scripts/adaptive_eval.py -> results.json)
- **Fan session:**
  - AUC beside the fan 0.953 (PCEN 0.955); Pd at 5 % window false alarms 0.93 (PCEN 0.97);
  - fan false-alarm increase -1/-2 dB (same as PCEN).
- **Simulator** (5 seeds): PCEN 0 % both ways; for G-PCEN it depends on the clutter.
  - Clutter switching on mid-recording (pre-stated setting): G-PCEN fails, Pd moving 0.20 and Pd stopped 0. The clutter is held as a target.
  - Clutter present from the start (diagnostic, 3 seeds): stopped-target Pd 0.95.
- **Own BGT60 calibration session**, contrast 10 s / 60 s / 300 s after sitting down:
  - SR +15 / +23 / +23 dB;
  - PCEN tau = 10 s: +9 / +2 / 0 dB;
  - G-PCEN tau = 10 s: +18 / +14 / +11 dB;
  - G-PCEN tau = 30 s: +19 / +20 / +12 dB.
- **Walabot** (0.1 false detections per window):

  | Front end | Persons detected | RR within 10 % |
  |---|---|---|
  | SR | 0.52 | 0.31 |
  | PCEN tau = 10 s | 0.02 | 0.01 |
  | PCEN tau = inf | 0.50 | 0.30 |
  | G-PCEN tau = 10 s | 0.43 | 0.25 |
  | G-PCEN tau = 30 s | 0.45 | 0.26 |
  | Oracle (true range) | – | 0.58 |

- **Micro-Doppler fidelity** (hand bins 0.60-0.85 m, correlation of the cue-averaged fan-on signature with the fan-off signature):

  | Front end | Raw | Baseline-corrected |
  |---|---|---|
  | Global threshold | 0.78 | 0.67 |
  | CA-CFAR | 0.64 | 0.26 |
  | PCEN | 0.64 | 0.56 |
  | G-PCEN | 0.36 | 0.54 |

  The raw metric was dominated by stationary lines common to both conditions. The baseline-corrected metric subtracts the 0.5 s before the cue and was added after seeing the plots. Both are reported.

## Limitations
- Clutter that switches on after M was learned is held as a target (for up to 600 s).
- Target onsets must be faster than tau.
