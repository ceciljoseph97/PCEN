# PCEN vs. CFAR: radar background suppression on the BGT60TR13C

This toolkit benchmarks Per-Channel Energy Normalization (PCEN, from audio) against standard radar background suppression on FMCW range-Doppler and micro-Doppler maps. It uses the Infineon BGT60TR13C (58–63.5 GHz, 1 Tx / 3 Rx).

| Experiment | Question | Script |
|---|---|---|
| **A: Detection** | Does slow-time PCEN detect targets better, and keep the false-alarm rate more constant, than static removal + CA/OS-CFAR or clutter-map CFAR? | `scripts/benchmark_detection.py` |
| **A2: Sensitivity** | How do the time constant τ and gain exponent α affect detection? | `scripts/sweep_pcen.py` |
| **B: Classification** | As a (learnable) front end for a micro-Doppler CNN, does PCEN generalise better to unseen rooms than log scaling? | `scripts/train_classifier.py` |

Everything runs on simulated data too, so you can check the pipeline before you record anything.

---

## 1. Setup (Windows)

```bat
cd "C:\CecilPersonal\Work\NextCloud\Gemeinsame Dateien\forschung\AI-DISCO\4_code\PCEN\Git"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Infineon Radar SDK** (only needed for recording; it is not on PyPI):

1. Download the *Radar SDK* from the Infineon Developer Center. It also comes with the Radar Development Kit / Radar Fusion GUI install.
2. Install the Python wheel for your Python version:
   `pip install <sdk>\python_wheels\ifxradarsdk-<ver>-<pyver>-win_amd64.whl`
3. Plug in the board over USB and check it works: `python scripts\live_view.py`

`pcen_radar/acquire.py` targets the **SDK 3.x FMCW API** (`DeviceFmcw`, `FmcwSimpleSequenceConfig`). If your SDK version has different class names, `open_device()` in that file is the only place you need to change. Compare it with the `raw_data.py` example in the SDK's `examples/` folder.

**Check the pipeline without hardware:**

```bat
python scripts\benchmark_detection.py --sim --out results\sim_detection
python scripts\sweep_pcen.py --sim --out results\sim_sweep
python scripts\make_synthetic_dataset.py --out data\synth --clips 30
python scripts\train_classifier.py data\synth --test-sessions roomB
```

## 2. Radar configuration

The defaults are in `pcen_radar/config.py`:

| Parameter | Value | Result |
|---|---|---|
| Sweep | 59.5 → 61.5 GHz (2 GHz) | ΔR = 7.5 cm |
| Fast time | 128 samples at 1 MHz | R_max ≈ 4.8 m, beat frequency below the 500 kHz IF low-pass |
| Slow time | 64 chirps × 391.55 µs | Δv ≈ 0.10 m/s, v_max ≈ ±3.2 m/s |
| Frame rate | 20 Hz | PCEN/micro-Doppler time step of 50 ms |

To change the configuration, pass a JSON file of overrides to `record.py --config my.json`, for example `{"frame_repetition_time_s": 0.033}`. **Keep one configuration for the whole study.**

## 3. Recording protocol (what to measure for the paper)

Each recording is one `.npz` file holding the raw frames, the configuration and the labels. The detection metrics need to know where the target is, so record scenes whose ground truth you can control:

| # | Scene | Command (abbreviated) | What it measures |
|---|---|---|---|
| 1 | **Empty room**, 30 s | `record.py --session roomA --name empty --frames 600` | Pure false-alarm rate per region |
| 2 | **Empty room + moving clutter** (desk fan, curtain in an open window, rotating plant) | `--name fan` | CFAR behaviour in non-stationary clutter, the main claim |
| 3 | **Pendulum** (corner reflector or metal sphere on a string) swinging at a measured distance | `--name pendulum --target-range 2.0 2.3` | Repeatable Pd with exact ground truth |
| 4 | **Person walking** back and forth between two tape marks | `--name walk --target-range 1.5 3.0` | Realistic Pd |
| 5 | Scene 4 **plus the fan** | `--name walk_fan --target-range 1.5 3.0` | Pd in heterogeneous clutter |
| 6 | **Dwell test:** walk in, stand still for 10 s, walk out (call out the times) | `--target-range 1.0 3.5 --moving 0 200 --moving 400 -1` | Stopping-target suppression, PCEN's known weakness |
| 7 | **Two people** close in range | `--target-range 1.8 2.6` | Masking, where CA-CFAR fails |

Repeat each scene in **at least two rooms** (`--session roomA`, `--session roomB`) and on different days. Cross-session results are what reviewers trust.

Tips:
- Put the radar on a tripod at about 1 m height and do not move it between recordings in a session. The static background has to stay static.
- Use tape marks for the range gates. Add the ±15 cm tolerance using `--tol-m` at benchmark time, not in the labels.
- For the dwell test, say "stop" and "go" while counting seconds, or record a phone video. Then convert the times to frame numbers (time × 20).

For classification, record 1–2 minute takes per activity and room: `--class walk`, `--class wave`, `--class still`, `--class empty`, and add `sit_down`, `fall` and so on as needed. The windows are cut automatically.

## 4. Running the benchmarks on your recordings

```bat
:: Experiment A, per recording (one results folder each)
python scripts\benchmark_detection.py data\roomA\walk_fan_*.npz --out results\roomA

:: Parameter sensitivity on all labelled detection recordings
python scripts\sweep_pcen.py data\roomA\pendulum_*.npz data\roomA\walk_*.npz --out results\sweep_roomA

:: Experiment B, cross-room
python scripts\train_classifier.py data --test-sessions roomB --epochs 40 --seeds 5
```

### Methods compared (Experiment A)

All methods see the same range-Doppler sequence. Each one outputs a *test statistic* map, and the threshold is swept to draw a ROC curve, so no method gets a hand-tuned advantage.

| Method | Input | Background estimate |
|---|---|---|
| Raw (global thr) | raw RD | none |
| Static removal (global) | mean-over-chirps removed | none |
| Static removal + CA-CFAR | mean removed | mean of a spatial training ring (guard 2×2, train 8×4) |
| Static removal + OS-CFAR | mean removed | k-th order statistic (k = 0.75 N) of the ring |
| **Clutter-map CFAR** | raw RD | per-cell exponential average over frames. This is the closest classical relative of PCEN and must be included. |
| PCEN (raw) | raw RD | per-cell IIR over frames, gain α, compression (δ, r) |
| Static removal + PCEN | mean removed | as above |
| Static removal + PCEN + CA-CFAR | mean removed | temporal (PCEN) followed by spatial (CFAR) normalisation |

### Metrics

- **ROC**: Pd vs. Pfa. Pd counts moving-target frames in which any cell in the labelled range gate exceeds the threshold. Pfa counts background cells (outside the gate and its guard band) that exceed it. Zero-Doppler cells inside the gate are excluded from both.
- **Pd at Pfa = 1e-2, 1e-3 and 1e-4.**
- **Worst-region Pfa excess:** set the threshold for a global Pfa of 1e-3, then measure Pfa separately in the zero-Doppler, near-range, far-range and mid regions. The excess is 10·log10(max regional Pfa / 1e-3). A true CFAR gives 0 dB.
- **Pd moving / Pd stopped:** detection rate at that threshold while the target moves vs. while it stands still. This exposes the dwell problem.
- **ms/frame:** runtime of each method.

## 5. What the simulation already shows (`results/sim_detection`)

These numbers come from a synthetic scene, not from measurements, and must not be reported as results. They show what the pipeline can reveal:

| Method | Pd@1e-3 | Pd@1e-4 | Pd stopped |
|---|---|---|---|
| Static removal + CA-CFAR | 0.70 | 0.53 | 0.72 |
| Clutter-map CFAR | 0.81 | 0.01 | 0.01 |
| Static removal + PCEN | 0.81 | 0.01 | 0.00 |
| Static removal + PCEN + CA-CFAR | **0.98** | 0.41 | 0.05 |

- PCEN behaves almost exactly like clutter-map CFAR. Your contribution therefore has to come from the compression/learnability, the hybrid, or the classification front end, not from "temporal normalisation" alone.
- PCEN strongly suppresses the fan band. However, fluctuating clutter keeps heavy tails, so Pd collapses at Pfa = 1e-4.
- Combining temporal PCEN with spatial CFAR gave the best moving-target detection here.
- Every temporal method loses the standing person. Spatial CA-CFAR still sees the breathing micro-motion.

The rankings depend on the scene. In a second synthetic scene (a person approaching from 4 m to 0.8 m without stopping), plain static removal + PCEN came first with Pd@1e-4 = 0.93, and the hybrid dropped to 0.16. Report results per scene type, and check on real data whether these trends hold.

## 6. Layout

```
pcen_radar/
  config.py      radar parameters + derived resolution/axes
  acquire.py     BGT60TR13C acquisition (Infineon Radar SDK)
  io.py          .npz recording format, labels -> target/guard masks
  processing.py  range FFT, Doppler FFT, static removal, micro-Doppler
  pcen.py        PCEN (batch + streaming)
  cfar.py        CA-, OS-, clutter-map CFAR statistics + analytic thresholds
  metrics.py     ROC, Pd@Pfa, regional Pfa, detection over time
  simulate.py    raw-IF simulator (static clutter, fan, walking/stopping person)
  dataset.py     recordings -> micro-Doppler windows
  nn.py          LearnablePCEN, FixedPCEN, LogFrontEnd, SmallCNN (PyTorch)
scripts/
  record.py  live_view.py  benchmark_detection.py  sweep_pcen.py
  make_synthetic_dataset.py  train_classifier.py
data/      your recordings  (data/<session>/<name>_<timestamp>.npz)
results/   outputs
```

## 7. References

- Wang et al., *Trainable frontend for robust and far-field keyword spotting*, ICASSP 2017 (original PCEN).
- Lostanlen et al., *Per-Channel Energy Normalization: Why and How*, IEEE SPL 26(1), 2019.
- Richards, *Fundamentals of Radar Signal Processing* (CA/OS-CFAR); Nitzberg, clutter-map CFAR.

## 8. Real-data results: KIT FC-FMCW gesture data set

`results/kit/KIT_RESULTS.md` has the pre-registered comparison on the public KIT BGT60TR13C data set (raw IF, 681 recordings). In short, static removal + PCEN leaves the least background information in the features. It also gives a modest gain on unseen backgrounds that survives a multiple-comparison correction (+0.10 to +0.12 accuracy with the tree classifier). No front end, however, closes the gap between setups. To reproduce:

```bat
:: features are already extracted in results\kit\kit_features.npz; to rebuild from the raw .mat files:
python scripts\kit_extract.py C:\Users\OliverCJ\Downloads\Kit\10.35097-hCNNxXlJBdFTSYse\data\dataset\dataset\dataset data\kit_feat --selection results\kit\selection.json
python scripts\kit_experiments.py data\kit_feat results\kit
```
`pcen_radar/kit.py` loads the KIT `.mat` files and builds the maps (`load_mat`, `extract`).

### Learnable PCEN + CNN on KIT (run in conda)

```bat
conda activate lecture_radar_systems
python -c "import torch, numpy, scipy; print(torch.__version__, torch.cuda.is_available())"
:: if torch is missing:  conda install pytorch cpuonly -c pytorch   (or the CUDA build)
cd "C:\CecilPersonal\Work\NextCloud\Gemeinsame Dateien\forschung\AI-DISCO\4_code\PCEN\Git"
python scripts\kit_train_cnn.py --seeds 3            :: all three splits, 5 front ends
python scripts\kit_train_cnn.py --splits E2_cross_setup --seeds 5   :: main split only, more seeds
```
Input maps: `results/kit/maps/` (float16 log-power, 682 recordings × 128 channels × 256 time steps).
Output: `results/kit_cnn/results.md` and `results.json`, including the learned PCEN parameters.
