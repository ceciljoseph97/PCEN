"""PCEN vs. CFAR benchmarking toolkit for FMCW radar (Infineon BGT60TR13C)."""

from .config import RadarConfig
from .processing import range_fft, range_doppler, rd_sequence, micro_doppler, remove_static
from .pcen import pcen, ClutterMapState
from .cfar import ca_cfar_stat, os_cfar_stat, clutter_map_stat, ca_cfar_alpha, os_cfar_alpha

__all__ = [
    "RadarConfig", "range_fft", "range_doppler", "rd_sequence", "micro_doppler",
    "remove_static", "pcen", "ClutterMapState", "ca_cfar_stat", "os_cfar_stat",
    "clutter_map_stat", "ca_cfar_alpha", "os_cfar_alpha",
]
