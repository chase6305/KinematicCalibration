"""Loading validates saved plan identities and geometry without replanning."""

import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from kinematic_calibration.backends import NumpyBackend
from kinematic_calibration.experiment import example_urdf
from kinematic_calibration.planning import load_pose_plan, save_pose_plan


@pytest.fixture
def saved_plan(tmp_path):
    output = tmp_path / "plan.npz"
    save_pose_plan(output, count=8, candidates=20, seed=7)
    return output


def rewrite_plan(path, *, arrays=None, metadata=None):
    with np.load(path, allow_pickle=False) as archive:
        contents = {key: archive[key] for key in archive.files}
    if arrays:
        contents.update(arrays)
    if metadata:
        report = json.loads(contents["metadata_json"].item())
        report.update(metadata)
        contents["metadata_json"] = np.asarray(json.dumps(report))
    np.savez_compressed(path, **contents)


@pytest.mark.parametrize(
    "start,closed", [(None, False), (np.zeros(6), False), (np.zeros(6), True)]
)
def test_round_trip_keeps_saved_visit_rows_and_selection_identities(
    tmp_path, start, closed
):
    output = tmp_path / "plan.npz"
    report = save_pose_plan(
        output,
        count=8,
        candidates=20,
        seed=7,
        start_configuration=start,
        return_to_start=closed,
    )
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    plan = load_pose_plan(
        output,
        backend=robot,
        source_urdf_sha256=hashlib.sha256(example_urdf().read_bytes()).hexdigest(),
    )
    assert plan.metadata == report
    assert plan.joint_names == robot.joint_names
    with np.load(output, allow_pickle=False) as archive:
        for key in (
            "planned_joint_positions",
            "candidate_indices",
            "selected_candidate_indices",
            "visit_order",
            "segment_distances_rad",
            "joint_limits",
            "singular_values",
        ):
            np.testing.assert_array_equal(getattr(plan, key), archive[key])
    candidates = np.random.default_rng(7).uniform(-np.pi, np.pi, (20, 6))
    np.testing.assert_array_equal(
        plan.planned_joint_positions[np.argsort(plan.visit_order)],
        candidates[plan.selected_candidate_indices],
    )
    assert len(plan.segment_distances_rad) == 7 + (start is not None) + closed


def test_one_pose_rank_deficient_plan_remains_valid_for_preview(tmp_path):
    output = tmp_path / "one.npz"
    save_pose_plan(output, count=1, candidates=2)
    plan = load_pose_plan(output)
    assert plan.planned_joint_positions.shape == (1, 6)
    assert plan.segment_distances_rad.shape == (0,)
    assert plan.singular_values.shape == (6,)
    assert not plan.metadata["full_rank"]
    assert plan.metadata["condition_number"] is None


@pytest.mark.parametrize("field", ["planned_joint_positions", "joint_limits"])
def test_plan_rejects_values_that_overflow_when_converted_to_float64(tmp_path, field):
    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip("longdouble does not exceed float64 on this platform")
    output = tmp_path / "one.npz"
    save_pose_plan(output, count=1, candidates=2)
    plan = load_pose_plan(output)
    limits = plan.joint_limits.astype(np.longdouble)
    limits[:, 1] = np.longdouble("1e401")
    arrays = {"joint_limits": limits}
    if field == "planned_joint_positions":
        positions = plan.planned_joint_positions.astype(np.longdouble)
        positions[0, 0] = np.longdouble("1e400")
        arrays[field] = positions
    rewrite_plan(output, arrays=arrays)

    with pytest.raises(ValueError, match=f"{field}.*finite"):
        load_pose_plan(output)


@pytest.mark.parametrize(
    "key,value",
    [
        ("visit_order", np.arange(8, dtype=float)),
        ("visit_order", np.zeros(8, dtype=int)),
        ("visit_order", np.arange(1, 9)),
        ("candidate_indices", np.arange(20, 28)),
        ("selected_candidate_indices", np.zeros(8, dtype=int)),
        ("joint_names", np.asarray(["same"] * 6)),
        (
            "joint_names",
            np.asarray(["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Other"]),
        ),
        ("joint_limits", np.zeros((6, 2))),
        ("planned_joint_positions", np.full((8, 6), 10.0)),
        ("planned_joint_positions", np.full((8, 6), np.nan)),
        ("planned_joint_positions", np.ones((8, 6), dtype=bool)),
        ("segment_distances_rad", np.zeros(7)),
        ("segment_distances_rad", np.zeros(8)),
        ("singular_values", np.asarray([1.0])),
        ("singular_values", np.full(30, np.nan)),
        ("metadata_json", np.asarray(["{}"])),
        ("metadata_json", np.asarray("{}", dtype=object)),
    ],
)
def test_rejects_corrupt_arrays(saved_plan, key, value):
    rewrite_plan(saved_plan, arrays={key: value})
    with pytest.raises(ValueError):
        load_pose_plan(saved_plan)


def test_candidate_mapping_is_checked_even_for_valid_unique_indices(saved_plan):
    plan = load_pose_plan(saved_plan)
    rewrite_plan(
        saved_plan, arrays={"candidate_indices": np.roll(plan.candidate_indices, 1)}
    )
    with pytest.raises(ValueError, match="candidate_indices.*visit_order"):
        load_pose_plan(saved_plan)


def test_candidate_identity_does_not_require_regenerating_seeded_proposals(saved_plan):
    plan = load_pose_plan(saved_plan)
    selected = np.arange(len(plan.selected_candidate_indices))
    rewrite_plan(
        saved_plan,
        arrays={
            "selected_candidate_indices": selected,
            "candidate_indices": selected[plan.visit_order],
        },
    )
    loaded = load_pose_plan(saved_plan)
    np.testing.assert_array_equal(
        loaded.planned_joint_positions, plan.planned_joint_positions
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", 1),
        ("schema_version", 2.0),
        ("artifact_type", "measurements"),
        ("selected_count", 7),
        ("candidate_count", True),
        ("measurements_acquired", True),
        ("collision_checked", 0),
        ("visibility_checked", True),
        ("trajectory_generated", True),
        ("return_to_start", 1),
        ("return_to_start", True),
        ("start_configuration_rad", [10] * 6),
        ("start_configuration_rad", [0] * 5),
        ("total_distance_rad", 0),
        ("total_distance_rad", float("nan")),
        ("selection_order_distance_rad", 0),
        ("distance_metric", "wrapped_angles"),
        ("parameter_count", 29),
        ("regressor_rank", 0),
        ("full_rank", 1),
        ("condition_number", None),
        ("characteristic_length_m", False),
        ("source_urdf_sha256", "not a digest"),
    ],
)
def test_rejects_inconsistent_metadata(saved_plan, key, value):
    rewrite_plan(saved_plan, metadata={key: value})
    with pytest.raises(ValueError):
        load_pose_plan(saved_plan)


@pytest.mark.parametrize(
    "field", ["joint_names", "dof", "joint_limits", "base_link", "end_link"]
)
def test_rejects_a_different_backend_model(saved_plan, field):
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    model = SimpleNamespace(
        joint_names=robot.joint_names,
        dof=robot.dof,
        joint_limits=robot.joint_limits.copy(),
        base_link=robot.base_link,
        end_link=robot.end_link,
    )
    if field == "joint_names":
        model.joint_names = tuple(reversed(model.joint_names))
    elif field == "dof":
        model.dof -= 1
    elif field == "joint_limits":
        model.joint_limits[0, 1] -= 0.1
    else:
        setattr(model, field, "other_frame")
    with pytest.raises(ValueError, match="backend"):
        load_pose_plan(saved_plan, backend=model)


def test_source_hash_check_is_optional_and_frames_only_checked_when_exposed(saved_plan):
    plan = load_pose_plan(saved_plan)
    model = SimpleNamespace(
        joint_names=plan.joint_names,
        dof=6,
        joint_limits=plan.joint_limits,
    )
    load_pose_plan(saved_plan, backend=model)
    with pytest.raises(ValueError, match="SHA-256"):
        load_pose_plan(saved_plan, source_urdf_sha256="0" * 64)


def test_loading_does_not_import_optional_engines_or_replan(saved_plan):
    script = """
import sys
from unittest.mock import patch
blocked = {name: None for name in ('roboticstoolbox', 'dexsim', 'viser', 'cv2')}
with patch.dict(sys.modules, blocked), \\
     patch('kinematic_calibration.planning.select_informative_poses', side_effect=AssertionError), \\
     patch('kinematic_calibration.routing.plan_visit_order', side_effect=AssertionError):
    from kinematic_calibration.planning import load_pose_plan
    plan = load_pose_plan(sys.argv[1])
assert plan.planned_joint_positions.shape == (8, 6)
assert 'robotkinecal' not in sys.modules
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    subprocess.run(
        [sys.executable, "-c", script, str(saved_plan)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
