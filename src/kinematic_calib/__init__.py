"""Kinematic-calibration workflow built on the robotkinecal solver."""

from .dataset import KinematicDataset, ObservationFrame
from .robust import HuberOptions
from .service import CalibrationFit, FitOptions, fit_dataset

__all__ = [
    "KinematicDataset",
    "ObservationFrame",
    "CalibrationFit",
    "FitOptions",
    "HuberOptions",
    "fit_dataset",
]
