# Working in this repository

- Use Python 3.10–3.12. Install locally with `uv venv .venv` and
  `uv pip install --python .venv/bin/python -e '.[dev,vision,visualization]'`.
- `src/robotkinecal` is the directly ported MIT POE solver. Preserve attribution,
  public API compatibility, and the `[v,w]` screw convention.
- `src/kinematic_calib` contains data contracts, fitting service, numerical
  validation, and the migrated URDF workflows.
- `src/kinematic_calibration` owns optional engines, acquisition, targets,
  hand-eye adapters, synthetic experiments, and the `kincal` CLI.
- Keep DexSim, RTB, native hand-eye, Viser, and OpenCV imports out of unrelated
  module initialization. The headless POE workflow must work without RTB.
- State transforms as `A_T_B` (B into A); use metres and radians. Never accept a
  nominal FK pose as an independent measurement when assessing calibration.
- Tests: `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`.
  This avoids unrelated ROS plugins in the shared workspace. Lint:
  `.venv/bin/ruff check src tests`; build: `.venv/bin/python -m build`.
- Preserve user modifications in reference repositories. Update the pinned
  `third_party/HandEyeCalibrationToolkit` gitlink intentionally; no auto-fetch or
  auto-build during normal imports. Never include generated captures or caches.
