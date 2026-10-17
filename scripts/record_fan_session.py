"""Guided recording session: hand gestures with a running fan in the background (BGT60TR13C).

  conda activate lecture_radar_systems
  python scripts/live_view.py                                   # 1) check the scene first
  python scripts/record_fan_session.py --root C:/radar_data/fan --participant P01
  python scripts/record_fan_session.py --root ... --participant P01 --only B3_front_on_push --redo
  python scripts/record_fan_session.py --dry-run --yes --root /tmp/fan_sim   # no radar: simulated
  python scripts/record_fan_session.py --plan v2 --root C:/radar_data/fan --participant P02   # session 2

Scene (see MEASUREMENT_PROTOCOL.md)
  radar on a tripod/table, fan 1.0 m in front of the radar on boresight, fan blades facing the radar,
  fixed speed, oscillation OFF.
  position "front": hand gestures at about 0.4 m in front of the radar (between radar and fan)
  position "side" : hand gestures at the fan's range (about 1.0 m), 0.3-0.4 m to the side of the fan

Block plan (ABBA over the fan state, so fan state is not confounded with time / fatigue):
  A  front, fan OFF : empty, swipe, push, circle
  B  front, fan ON  : empty, circle, push, swipe
  C  side,  fan ON  : empty, swipe, push, circle
  D  side,  fan OFF : empty, circle, push, swipe
  E  (only with --osc) front, fan ON + oscillation : empty, swipe, push, circle
Plan v2 (--plan v2, session 2): fan at 1.0 m; positions "front" (hand 0.4 m) and "near" (hand 0.8 m,
directly in front of the fan, between radar and fan); gestures push-pull, circle, hold:
  A  front, fan OFF : empty, pushpull, circle, hold
  B  front, fan ON  : empty, hold, circle, pushpull
  C  near,  fan ON  : empty, pushpull, circle, hold
  D  near,  fan OFF : empty, hold, circle, pushpull
HOLD uses a 6-s cycle: GO -> raise the hand into position and hold it completely still; mid beep at
3.5 s -> lower the hand out of the beam; low beep (ready) at 5.0 s. Gesture window 0.2-3.5 s, of which
1.5-3.5 s is the still phase; rest window 4.0-4.9 s (hand lowered). Range gate of "near" (fixed before
recording): [r_f - 0.40, r_f - 0.10] = 0.60-0.90 m for r_f = 1.0 m.
A gesture block = 5 s lead-in (hand still, no gesture) + reps x 4 s cycles. In each cycle:
  t = -1.0 s "ready" (low beep), t = 0 "GO" (high beep) -> ONE gesture, finished within ~1.5 s,
  then hand back to the rest position and keep it still until the next "ready".
An empty block = nobody gestures (stand behind the radar / out of the beam), 45 s.

Files: <root>/<participant>/<block>.npz   (signed int16 raw IF, no clipping, + labels JSON; one file per block)
The script skips blocks that already exist (resume after a break); --redo overwrites.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcen_radar import RadarConfig

GESTURES = {"swipe": "SWIPE  (hand moves left -> right across the beam, once)",
            "push": "PUSH  (hand moves towards the radar and back, once)",
            "circle": "CIRCLE  (one circle with the hand, ~15 cm diameter)",
            "pushpull": "PUSH-PULL  (hand moves ~10 cm towards the radar and back, once)",
            "hold": "HOLD  (GO: raise the hand into position and hold it completely STILL; "
                    "mid beep: lower it out of the beam)"}
# per-gesture timing [s]: cycle, gesture window, rest window, time of the 'lower' beep (None = no beep)
TIMING = {"hold": (6.0, (0.2, 3.5), (4.0, 4.9), 3.5)}
DEFAULT_TIMING = (None, (0.2, 1.7), (2.3, 3.8), None)       # cycle None -> --cycle
POS_TEXT = {"front": "hand at ~0.4 m in front of the radar (between radar and fan)",
            "side": "hand at ~1.0 m (same distance as the fan), 0.3-0.4 m to the SIDE of the fan",
            "near": "hand at ~0.8 m, directly IN FRONT of the fan (between radar and fan, ~0.2 m before it)"}
FAN_TEXT = {"off": "fan OFF (blades standing still)", "on": "fan ON, fixed speed, oscillation OFF",
            "osc": "fan ON, same speed, oscillation ON"}
# range gates used later by the analysis (stored in every file)
from pcen_radar.fan_scene import gates as scene_gates
_G = scene_gates(1.0)                       # replaced in main() by --fan-range
HAND_GATE = {"front": _G["front"], "side": _G["side"], "near": (0.60, 0.90)}
FAN_GATE = _G["fan"]


def plan(include_osc: bool, version: str = "v1"):
    if version == "v2":
        rounds = [("A", "front", "off", ["pushpull", "circle", "hold"]),
                  ("B", "front", "on", ["hold", "circle", "pushpull"]),
                  ("C", "near", "on", ["pushpull", "circle", "hold"]),
                  ("D", "near", "off", ["hold", "circle", "pushpull"])]
        include_osc = False
    else:
        rounds = [("A", "front", "off", ["swipe", "push", "circle"]),
              ("B", "front", "on", ["circle", "push", "swipe"]),
              ("C", "side", "on", ["swipe", "push", "circle"]),
              ("D", "side", "off", ["circle", "push", "swipe"])]
    if include_osc:
        rounds.append(("E", "front", "osc", ["swipe", "push", "circle"]))
    blocks, k = [], 1
    for rnd, pos, fan, gests in rounds:
        for g in ["empty"] + gests:
            blocks.append({"name": f"{rnd}{k}_{pos}_{fan}_{g}", "round": rnd, "position": pos,
                           "fan": fan, "gesture": g})
            k += 1
    return blocks


def beep(freq, ms):
    def _b():
        try:
            import winsound
            winsound.Beep(int(freq), int(ms))
        except Exception:
            print("\a", end="", flush=True)
    threading.Thread(target=_b, daemon=True).start()


def schedule(block, a, fp):
    """Frame indices of the cues for one block."""
    if block["gesture"] == "empty":
        n = int(round(a.empty_s / fp))
        return n, []
    lead = int(round(a.lead_in / fp))
    cyc = (TIMING.get(block["gesture"], DEFAULT_TIMING)[0] or a.cycle) / fp
    cues = [int(round(lead + i * cyc)) for i in range(a.reps)]
    n = int(round(lead + a.reps * cyc + 1.0 / fp))
    return n, cues


# ----------------------------------------------------------------- simulated frames (dry run)
def simulated_frames(block, cfg, n, cues, seed):
    from pcen_radar.simulate import simulate, Scene, _beat
    rng = np.random.default_rng(seed)
    fp = cfg.frame_repetition_time_s
    r_hand = {"front": 0.40, "near": 0.80}.get(block["position"], 1.05)
    cue_t = np.array(cues) * fp
    g = block["gesture"]

    def hand(f, tf, tc):
        t = tf + tc
        R = np.full_like(t, r_hand)
        amp = 0.5
        for c in cue_t:
            u = (t - c - 0.25) / 1.0                       # gesture lasts ~1 s after reaction time
            m = (u > 0) & (u < 1)
            if g in ("push", "pushpull"):
                R = np.where(m, R - 0.12 * np.sin(np.pi * u), R)
            elif g == "swipe":
                R = np.where(m, R + 0.03 * np.sin(2 * np.pi * u), R)
            elif g == "circle":
                R = np.where(m, R + 0.07 * np.sin(2 * np.pi * u), R)
        if g == "empty":
            return 0.0
        b = _beat(cfg, R + 0.002 * np.sin(2 * np.pi * 0.3 * t), amp, 0.0, tc)
        if g == "hold":                                    # hand only present while held up
            pres = np.zeros_like(t, bool)
            for c in cue_t:
                pres |= (t > c + 0.3) & (t < c + 3.6)
            b = b * pres[:, None]
        return b

    fan_on = block["fan"] in ("on", "osc")
    sc = Scene(n_frames=n, static=[(0.15, 25.0), (1.0, 6.0), (2.4, 15.0), (3.6, 10.0)],
               fan_range_m=1.0, fan_amp=1.2 if block["fan"] != "osc" else 1.0,
               fan_frames=(0, n) if fan_on else (0, 0), target_path="none", limbs=False,
               seed=int(rng.integers(1 << 31)))
    frames, _, _ = simulate(cfg, sc, extra_fn=hand)
    return frames, np.arange(n) * fp


# ----------------------------------------------------------------- live recording
def record_block(block, cfg, n, cues, a):
    from pcen_radar.acquire import open_device, next_frame, close_device
    fp = cfg.frame_repetition_time_s
    ready = {c - int(round(1.0 / fp)): i for i, c in enumerate(cues)}
    go = {c: i for i, c in enumerate(cues)}
    tim = TIMING.get(block["gesture"], DEFAULT_TIMING)
    rest = {c + int(round(tim[1][1] / fp)): i for i, c in enumerate(cues)}
    lower = {c + int(round(tim[3] / fp)): i for i, c in enumerate(cues)} if tim[3] else {}
    frames = np.empty((n, cfg.num_rx, cfg.num_chirps, cfg.num_samples), np.float32)
    ts = np.empty(n)
    stats = {"dropped": 0}
    drop_at = []
    dev = open_device(cfg)
    try:
        for _ in range(int(round(0.5 / fp))):              # discard the first frames (PLL / FIFO settle)
            next_frame(dev, {})
        t0 = time.perf_counter()
        for i in range(n):
            d0 = stats["dropped"]
            frames[i] = next_frame(dev, stats)
            ts[i] = time.perf_counter() - t0
            if stats["dropped"] > d0:
                drop_at.append(i)
            if i in ready:
                beep(600, 80)
                print(f"\r  rep {ready[i] + 1:2d}/{len(cues)}  ready ...            ", end="", flush=True)
            elif i in go:
                beep(1200, 120)
                print(f"\r  rep {go[i] + 1:2d}/{len(cues)}  >>> GO {block['gesture'].upper()} <<<   ",
                      end="", flush=True)
            elif i in lower:
                beep(900, 100)
                print(f"\r  rep {lower[i] + 1:2d}/{len(cues)}  lower the hand            ", end="", flush=True)
            elif i in rest:
                print(f"\r  rep {rest[i] + 1:2d}/{len(cues)}  rest (hand still)         ", end="", flush=True)
            elif not cues and i % int(round(5 / fp)) == 0:
                print(f"\r  empty background  t = {i * fp:5.1f} / {n * fp:.0f} s   ", end="", flush=True)
    finally:
        close_device(dev)
        del dev
    print()
    return frames, ts, drop_at


# ----------------------------------------------------------------- quick quality check
def quick_qc(frames, cfg, block, cues):
    from pcen_radar.processing import rd_sequence
    P = rd_sequence(frames, clutter="mean")                   # all Rx (non-coherent)
    rr = cfg.range_axis(P.shape[1])
    D = P.shape[2]
    nz = np.r_[0:D // 2 - 1, D // 2 + 2:D]                    # without zero-Doppler +-1
    fp = cfg.frame_repetition_time_s
    fan = (rr >= FAN_GATE[0]) & (rr <= FAN_GATE[1])
    msg = [f"fan-gate moving power {10 * np.log10(P[:, fan][:, :, nz].mean() + 1e-30):6.1f} dB"]
    if cues:
        hg = HAND_GATE[block["position"]]
        hand = (rr >= hg[0]) & (rr <= hg[1])
        e = P[:, hand][:, :, nz].sum(axis=(1, 2))
        _, gw, rw, _ = TIMING.get(block["gesture"], DEFAULT_TIMING)
        on = np.concatenate([e[c + int(gw[0] / fp): c + int(gw[1] / fp)] for c in cues])
        off = np.concatenate([e[c + int(rw[0] / fp): c + int(rw[1] / fp)] for c in cues])
        msg.append(f"hand contrast (gesture vs rest) {10 * np.log10(on.mean() / (off.mean() + 1e-30)):+5.1f} dB")
    return "   ".join(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/fan", help="where to save (keep big raw data out of NextCloud)")
    ap.add_argument("--participant", default="P01")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--lead-in", dest="lead_in", type=float, default=5.0)
    ap.add_argument("--cycle", type=float, default=4.0)
    ap.add_argument("--empty-s", dest="empty_s", type=float, default=45.0)
    ap.add_argument("--osc", action="store_true", help="add round E (fan oscillating)")
    ap.add_argument("--plan", default="v1", choices=["v1", "v2"],
                    help="v1: session 1 (swipe/push/circle, front/side); v2: push-pull/circle/hold, front/near")
    ap.add_argument("--only", nargs="+", help="record only these block names")
    ap.add_argument("--redo", action="store_true", help="overwrite existing blocks")
    ap.add_argument("--dry-run", action="store_true", help="simulate frames, no radar needed")
    ap.add_argument("--yes", action="store_true", help="do not wait for Enter between blocks")
    ap.add_argument("--notes", default="")
    ap.add_argument("--fan-range", dest="fan_range", type=float, default=1.0,
                    help="measured distance radar -> fan hub [m]; range gates shift with it")
    a = ap.parse_args()
    global HAND_GATE, FAN_GATE
    G = scene_gates(a.fan_range)
    HAND_GATE, FAN_GATE = {"front": G["front"], "side": G["side"],
                           "near": (round(a.fan_range - 0.40, 3), round(a.fan_range - 0.10, 3))}, G["fan"]
    POS_TEXT["side"] = (f"hand at ~{a.fan_range:.2f} m (same distance as the fan), "
                        "0.3-0.4 m to the SIDE of the fan")
    print(f"fan at {a.fan_range:.2f} m -> gates: fan {FAN_GATE}, front hand {HAND_GATE['front']}, "
          f"side hand {HAND_GATE['side']}, near hand {HAND_GATE['near']}")

    cfg = RadarConfig()                       # 59.5-61.5 GHz, 64 chirps, 20 Hz (same as the rest of the repo)
    fp = cfg.frame_repetition_time_s
    out = Path(a.root) / a.participant
    out.mkdir(parents=True, exist_ok=True)
    blocks = plan(a.osc, a.plan)
    if a.only:
        blocks = [b for b in blocks if b["name"] in a.only]
    print(cfg.summary())
    print(f"{len(blocks)} blocks -> {out.resolve()}")
    last = None
    for bi, b in enumerate(blocks):
        f = out / f"{b['name']}.npz"
        if f.exists() and not a.redo:
            print(f"[skip] {b['name']} exists")
            continue
        n, cues = schedule(b, a, fp)
        setting = (b["position"], b["fan"])
        print("\n" + "=" * 72)
        print(f"BLOCK {bi + 1}/{len(blocks)}: {b['name']}   ({n * fp:.0f} s)")
        if setting != last:
            print(f"  CHANGE SETUP ->  {FAN_TEXT[b['fan']]}")
            print(f"                   {POS_TEXT[b['position']]}")
        if b["gesture"] == "empty":
            print("  EMPTY: nobody in front of the radar (stand behind it), stay still.")
        else:
            print(f"  GESTURE: {GESTURES[b['gesture']]}")
            if b["gesture"] == "hold":
                print(f"  {a.reps} reps: low beep = ready, high beep = GO (raise + hold still), "
                      "mid beep = lower the hand out of the beam.")
            else:
                print(f"  {a.reps} reps: low beep = ready, high beep = GO (one gesture), then rest still.")
        if not a.yes:
            k = input("  Enter = start,  s = skip,  q = quit > ").strip().lower()
            if k == "q":
                break
            if k == "s":
                continue
        if a.dry_run:
            frames, ts = simulated_frames(b, cfg, n, cues, seed=bi)
            drop_at = []
        else:
            for i in range(3, 0, -1):
                print(f"  starting in {i} ...", flush=True); time.sleep(1)
            frames, ts, drop_at = record_block(b, cfg, n, cues, a)
        last = setting
        dts = np.diff(ts)
        drops = int((dts > 1.5 * fp).sum())
        cyc_b, gw_b, rw_b, low_b = TIMING.get(b["gesture"], DEFAULT_TIMING)
        labels = dict(b, cue_frames=cues, frame_period_s=fp, reps=a.reps, lead_in_s=a.lead_in,
                      cycle_s=cyc_b or a.cycle, gesture_window_s=list(gw_b), rest_window_s=list(rw_b),
                      lower_beep_s=low_b, still_window_s=[1.5, 3.5] if b["gesture"] == "hold" else None,
                      plan=a.plan,
                      hand_gate_m=HAND_GATE[b["position"]], fan_gate_m=FAN_GATE,
                      participant=a.participant, dry_run=a.dry_run, fan_range_m=a.fan_range)
        meta = {"sensor": "BGT60TR13C", "recorded": time.strftime("%Y%m%d_%H%M%S"), "notes": a.notes,
                "measured_frame_period_s": float(np.median(dts)) if dts.size else fp,
                "late_frames": drops, "dropped_before_frames": drop_at}
        # signed int16 without clipping (the earlier uint16 version clipped negative values at 0)
        q = np.round(frames * 4095.0)
        if q.min() < -32768 or q.max() > 32767:
            raise ValueError(f"ADC values out of int16 range: {q.min()} .. {q.max()}")
        zero_share = [(frames[:, i] == 0).mean() for i in range(frames.shape[1])]
        if max(zero_share) > 0.05 or min(frames[:, i].std() for i in range(frames.shape[1])) == 0:
            print(f"  WARNING: ADC check: exact-zero share per Rx {[round(z, 3) for z in zero_share]}, "
                  f"std per Rx {[float(frames[:, i].std()) for i in range(frames.shape[1])]}")
        np.savez_compressed(f, frames_i16=q.astype(np.int16), adc_scale=np.float32(4095.0), t=ts,
                            config=cfg.to_json(), labels=json.dumps(labels), meta=json.dumps(meta))
        print(f"  saved {f.name}  {frames.shape}  {f.stat().st_size / 1e6:.0f} MB  "
              f"frame period {meta['measured_frame_period_s'] * 1e3:.1f} ms  late frames {drops}  "
              f"dropped {len(drop_at)}")
        print("  QC: " + quick_qc(frames, cfg, b, cues))
        if len(drop_at) > 0.01 * n:
            print("  WARNING: >1% dropped frames - consider --redo for this block (close other programs).")
    print("\nDone. Next: python scripts/analyze_fan.py --data", out)


if __name__ == "__main__":
    main()
