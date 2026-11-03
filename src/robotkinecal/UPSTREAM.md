# Imported implementation

`calibration.py` and the public `SerialRobotKineCal` / `CalibrationResult` API
were directly imported from
[PhilNad/robot-arm-kinematic-calibration](https://github.com/PhilNad/robot-arm-kinematic-calibration)
at commit `c21f498357a851525fbc84ff0a67c4ebaccdc6ba` (2026-11-03).
The upstream implementation is copyright 2024 Philippe Nadeau and is licensed
under MIT; the complete notice is in `third_party/licenses/robotkinecal-MIT.txt`.

Local changes make Robotics Toolbox optional by importing it only when the
original robot-model constructor is used. `from_screw_axes` accepts a NumPy
product-of-exponentials model for simulation backends and custom URDF loaders.
The solver, adjoint regressor, four-parameter axis updates, weighting,
backtracking, and result API retain the upstream implementation. Local tests
exercise this independent constructor as well as the original RTB constructor.

Additional local correctness fixes: ERobot axes respect Rx/Ry/Rz and flipped
joints; reloading position/pose observations invalidates the dimension-dependent
basis cache; invalid non-real inputs and invalid poses are rejected; reported
position errors use Euclidean endpoint distance; convergence/divergence follows
the same weighted objective as the line search; singular regressors return
unavailable uncertainty instead of false certainty from a pseudoinverse. A stable
SE(3) logarithm uses SciPy rotation vectors and a half-angle inverse left
Jacobian near pi; independent matrix-exponential tests verify its linear/angular
convention, small-angle limit, and large-translation behavior.

The second-stage performance change adds `_batch.py`, array terminal FK and
adjoint regressor APIs, and shared-prefix batched linearization/residuals in
`solve`. The scalar `forward_kinematics`, `A_Matrix`, `Q_Matrix`, and `_pose_log`
remain as independent references. Equivalence tests cover arbitrary axes,
zero/pi angles, updates/reset, pose/point switching, and full optimization with
a scalar substitute. Reproducible profiles and before/after results are recorded
in `docs/performance.md` and `benchmarks/profile_solver.py`.

Third-stage stopping distinguishes a small proposed update with a stalled
relative objective from substantive backtracking failure. The explicit
`small_update` termination records its criterion and metrics; common weight
units are normalized only for stopping tolerances. See `docs/numerics.md` for
the noisy Python 3.12 reproduction, finite-difference evidence, and controls.
