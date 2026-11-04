"""Common read boundary for verified pose and image experiment bundles."""

import json
from pathlib import Path


def load_experiment_bundle(path: str | Path):
    """Return ``(report, validation_dataset, model)`` after full verification.

    Vision imports remain optional until an image bundle is requested. Each
    concrete loader validates its own schema, files and numerical semantics.
    """
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    artifact_type = manifest.get("artifact_type")
    if artifact_type == "image-kinematic-experiment":
        from .image_simulation import load_image_experiment

        return load_image_experiment(path)
    if artifact_type == "kinematic-calibration-experiment":
        from .experiment import load_experiment

        return load_experiment(path)
    raise ValueError(f"unsupported experiment artifact_type: {artifact_type!r}")
