# Measurement protocol: hand gestures with a running fan (BGT60TR13C)

Goal: test whether PCEN suppresses **moving but steady** background (a running fan) better than static removal and CA-CFAR. Static removal cannot remove a fan, because its blades have non-zero Doppler. The KIT data set has no such background, so this is the one experiment only your own radar can provide. Everything below is fixed in advance: `scripts/analyze_fan.py` contains the pre-registered hypotheses.

Time needed: about 15 min setup, 35 min recording and 10 min analysis.

## 1. What you need
- The BGT60TR13C on a table or tripod, at about hand height, pointing horizontally into the room.
- A desk or standing fan. Use a fixed speed level (medium) with oscillation **off**.
- A tape measure. Use tape to mark the floor or table spots.
- Laptop in the `lecture_radar_systems` environment, with the Infineon SDK wheel (`ifxradarsdk`) installed. Sound on, because the script beeps.
- Free disk space: about 1 GB. Save to a folder **outside NextCloud**, for example `C:\radar_data\fan`, so the raw data isn't synced.

## 2. Scene (top view)

```
                       0.3-0.4 m
                    side hand  <---->  FAN  (centre at 1.0 m, blades facing the radar)
                                         |
                                         |  1.0 m
                     front hand (0.4 m)  |
                                         |
                                      [RADAR] ---> boresight
                     (you stand to the side/behind the radar, arm reaching in)
```

- Put the fan **centre at 1.0 m** on the radar boresight, with the blades facing the radar. Measure from the antenna face to the fan hub.
- **Front** position: gestures at about **0.4 m** in front of the radar, on boresight, between the radar and the fan.
- **Side** position: gestures at about **1.0 m** (the same distance as the fan), **0.3–0.4 m to the side** of the fan hub. Stay out of the fan's airflow so your hand isn't pushed.
- Keep everything else in the room fixed for the whole session: no walking through the beam, doors closed. Your body stays put; only your hand and forearm move.
- Rest position between gestures: hand still at the gesture spot. Don't drop your arm.

## 3. Check the radar and the scene first (3 min)

```
conda activate lecture_radar_systems
cd "C:\CecilPersonal\Work\NextCloud\Gemeinsame Dateien\forschung\AI-DISCO\4_code\PCEN"
python scripts\check_radar.py         :: 20 s acquisition only; it must say "dropped 0"
python scripts\live_view.py
```

- **Fan on:** in the "static removal" panel you should see a smeared vertical stripe at about 1.0 m, across many velocities. That is the fan. If you see nothing, angle the fan more towards the radar or use a higher speed.
- **Hand:** wave at 0.4 m and at the side spot. You should see a blob at the right range.
- Close the window when you're done.

## 4. Record (about 35 min, 16 blocks)

```
python scripts\record_fan_session.py --root C:\radar_data\fan --participant P01
```

The script guides you block by block and waits for **Enter** before each one. Its prompts tell you when to switch the fan or change position. The order is fixed so the fan state isn't confounded with time:

| Round | Position | Fan | Blocks |
|---|---|---|---|
| A | front | OFF | empty, swipe, push, circle |
| B | front | ON  | empty, circle, push, swipe |
| C | side  | ON  | empty, swipe, push, circle |
| D | side  | OFF | empty, circle, push, swipe |

- **Empty block (45 s):** nobody in front of the radar. Stand behind it and keep still. The fan stays as the round says.
- **Gesture block (about 85 s):**
  - first 5 s: hand still at the spot, no gesture;
  - then 20 repetitions, each with a **low beep** (get ready), a **high beep** (do ONE gesture, finished within about 1.5 s) and then rest with your hand still until the next low beep.
- Gestures:
  - **swipe:** hand moves left → right across the beam, once;
  - **push:** hand moves towards the radar and back, about 10–15 cm;
  - **circle:** one circle, about 15 cm across.
- **Wait about 10 s** after switching the fan on before you press Enter. The script also has a 3 s countdown.

After each block the script prints a quick check:
- `hand contrast` should be clearly positive (more than about +6 dB with the fan off). If it's near 0 dB, your gestures weren't seen. Check the position and redo.
- `dropped` should be 0. If it's more than 1 % of the frames, close other programs and redo. `late frames` is only information: the board buffers frames, so they can arrive in bursts.

**Redo or resume:** the script skips blocks that already exist. To repeat one block:

```
python scripts\record_fan_session.py --root C:\radar_data\fan --participant P01 --only B6_front_on_circle --redo
```

**Optional, if you have 10 more minutes:** add `--osc` to record round E (front, fan oscillating). The fan then sweeps its clutter across the beam, which is a harder and less stationary case.

## 5. Analyse (about 10 min, no GPU needed)

```
python scripts\analyze_fan.py --data C:\radar_data\fan\P01 --out results\fan
```

This writes `results\fan\results.md`, `results.json`, `windows.json` and `fan_maps.npz` (small range-time maps for the paper figure). Tell me when it's done and I'll make the paper figure and text.

## 6. Note down (for the paper's setup section)
- Fan model, blade diameter, speed level, and the exact distances you used.
- Room type (office or lab), and the radar height above the floor.
- Anything unusual, such as someone walking in or a block redone.
