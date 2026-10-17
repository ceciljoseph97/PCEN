"""Range gates of the fan-background experiment, defined relative to the fan distance.

Pre-registered for a fan at 1.0 m: fan gate 0.85-1.25 m, side hand gate 0.80-1.25 m, front hand
gate 0.20-0.65 m. The same rule shifted with the actual fan distance r_f (decided before looking
at any data, only the physical placement changed):
    fan gate   [r_f - 0.15, r_f + 0.25]     (blades + housing/stand behind the hub)
    side hand  [r_f - 0.20, r_f + 0.25]     (hand at the fan's distance)
    front hand [0.20, min(0.65, r_f - 0.20)] (kept clear of the fan gate)
For r_f = 1.0 this gives exactly the pre-registered gates.
"""


def gates(fan_range_m: float = 1.0) -> dict:
    r = float(fan_range_m)
    return {"fan": (round(r - 0.15, 3), round(r + 0.25, 3)),
            "side": (round(r - 0.20, 3), round(r + 0.25, 3)),
            "front": (0.20, round(min(0.65, r - 0.20), 3))}
