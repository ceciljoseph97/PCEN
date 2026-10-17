"""Raw data acquisition from the Infineon BGT60TR13C via the Infineon Radar SDK (ifxradarsdk).

The SDK is NOT on PyPI: download the "Radar SDK" from Infineon (Radar Development Kit / Developer
Center), then `pip install` the wheel from <sdk>/python_wheels/ifxradarsdk-*.whl that matches your
Python version. This module is written against the SDK 3.x FMCW API
(ifxradarsdk.fmcw.DeviceFmcw + FmcwSimpleSequenceConfig), tested with 3.6.4.

Frame drops: if the host does not fetch frames fast enough, the board buffer overflows and the SDK
raises ErrorFrameAcquisitionFailed / ErrorFifoOverflow. `next_frame()` catches that, restarts the
acquisition and counts the drop, so a recording survives a hiccup. Keep the reading loop light
(no plotting in it) - `FrameReader` runs it in a background thread for live displays.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .config import RadarConfig

_DROP_ERRORS = ("FrameAcquisitionFailed", "FifoOverflow", "Timeout")


def open_device(cfg: RadarConfig):
    try:
        from ifxradarsdk import get_version_full
        from ifxradarsdk.fmcw import DeviceFmcw
        from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "ifxradarsdk not found. Install the wheel shipped with the Infineon Radar SDK "
            "(python_wheels/ifxradarsdk-*.whl)."
        ) from e

    print("Radar SDK", get_version_full())
    dev = DeviceFmcw()
    print("Sensor:", dev.get_sensor_type(), "| UUID:", dev.get_board_uuid())
    seq_cfg = FmcwSimpleSequenceConfig(
        frame_repetition_time_s=cfg.frame_repetition_time_s,
        chirp_repetition_time_s=cfg.chirp_repetition_time_s,
        num_chirps=cfg.num_chirps,
        tdm_mimo=False,
        chirp=FmcwSequenceChirp(
            start_frequency_Hz=cfg.start_frequency_Hz,
            end_frequency_Hz=cfg.end_frequency_Hz,
            sample_rate_Hz=cfg.sample_rate_Hz,
            num_samples=cfg.num_samples,
            rx_mask=cfg.rx_mask,
            tx_mask=cfg.tx_mask,
            tx_power_level=cfg.tx_power_level,
            lp_cutoff_Hz=cfg.lp_cutoff_Hz,
            hp_cutoff_Hz=cfg.hp_cutoff_Hz,
            if_gain_dB=cfg.if_gain_dB,
        ),
    )
    sequence = dev.create_simple_sequence(seq_cfg)
    dev.set_acquisition_sequence(sequence)
    return dev


def _get(dev, timeout_ms):
    """get_next_frame with a timeout when this SDK version supports it (never block forever)."""
    import inspect
    if not hasattr(dev, "_pcen_has_timeout"):
        try:
            dev._pcen_has_timeout = "timeout_ms" in inspect.signature(dev.get_next_frame).parameters
        except (TypeError, ValueError, AttributeError):
            dev._pcen_has_timeout = False
    if dev._pcen_has_timeout:
        return dev.get_next_frame(timeout_ms=timeout_ms)
    return dev.get_next_frame()


def next_frame(dev, stats: dict | None = None, max_retries: int = 5, timeout_ms: int = 2000) -> np.ndarray:
    """One frame (rx, chirps, samples) float32. Recovers from dropped frames and timeouts
    (counted in stats['dropped']); re-raises any other error."""
    for _ in range(max_retries):
        try:
            return np.asarray(_get(dev, timeout_ms)[0], dtype=np.float32)
        except Exception as e:
            if not any(k in type(e).__name__ for k in _DROP_ERRORS):
                raise
            if stats is not None:
                stats["dropped"] = stats.get("dropped", 0) + 1
            try:                                   # restart the acquisition cleanly
                dev.stop_acquisition()
                dev.start_acquisition()
            except Exception:
                pass
    raise RuntimeError(f"{max_retries} frame acquisitions failed in a row - check the USB cable/port "
                       "and close other programs using the radar (Radar Fusion GUI).")


def close_device(dev):
    try:
        dev.stop_acquisition()
    except Exception:
        pass


def record(cfg: RadarConfig, n_frames: int, on_frame=None):
    """Record n_frames. Returns frames (T, rx, chirps, samples) float32, timestamps, stats."""
    dev = open_device(cfg)
    frames = np.empty((n_frames, cfg.num_rx, cfg.num_chirps, cfg.num_samples), np.float32)
    ts = np.empty(n_frames)
    stats = {"dropped": 0}
    t0 = time.perf_counter()
    try:
        for i in range(n_frames):
            frames[i] = next_frame(dev, stats)
            ts[i] = time.perf_counter() - t0
            if on_frame:
                on_frame(i, frames[i])
    finally:
        close_device(dev)
        del dev
    if stats["dropped"]:
        print(f"\nWARNING: {stats['dropped']} frame drop(s) during recording")
    return frames, ts


class FrameReader:
    """Reads frames in a background thread; `latest()` returns the newest frame (older ones are
    skipped), so a slow display can never overflow the radar buffer."""

    def __init__(self, cfg: RadarConfig):
        self.cfg = cfg
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.stats = {"dropped": 0, "read": 0}
        self._stop = threading.Event()
        self.error: Exception | None = None
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            dev = open_device(self.cfg)
        except Exception as e:
            self.error = e
            return
        try:
            while not self._stop.is_set():
                f = next_frame(dev, self.stats)
                self.stats["read"] += 1
                try:
                    self.q.get_nowait()            # drop the stale frame, keep only the newest
                except queue.Empty:
                    pass
                self.q.put(f)
        except Exception as e:
            self.error = e
        finally:
            close_device(dev)

    def start(self):
        self._t.start()
        return self

    def latest(self, timeout=2.0):
        if self.error:
            raise self.error
        return self.q.get(timeout=timeout)

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2)
