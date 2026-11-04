"""Generic calibration imports must work without optional backends."""

import subprocess
import sys
import textwrap
from pathlib import Path


def test_imports_do_not_load_optional_backends(tmp_path):
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        sys.path.insert(0, sys.argv[1])
        blocked = {"cv2", "handeye_calibration", "dexsim", "roboticstoolbox", "viser"}

        class BlockOptionalBackends(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] in blocked:
                    raise AssertionError(f"unexpected optional backend import: {fullname}")

        sys.meta_path.insert(0, BlockOptionalBackends())

        from kinematic_calib import KinematicDataset, ObservationFrame
        from kinematic_calibration.handeye import HandEyeResult, evaluate_eye_to_hand
        from kinematic_calibration.targets import BoardConfig, get_board_config
        from kinematic_calibration.targets.detection import BoardDetection

        assert not blocked.intersection(name.partition(".")[0] for name in sys.modules)
        """
    )
    source = Path(__file__).resolve().parents[2] / "src"
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(source)],
        cwd=tmp_path,
        check=True,
    )


def test_handeye_help_and_closure_math_need_no_optional_backend(tmp_path):
    script = textwrap.dedent(
        """
        import contextlib
        import importlib.abc
        import io
        import sys
        import numpy as np
        sys.path.insert(0, sys.argv[1])
        blocked = {"cv2", "handeye_calibration", "dexsim", "roboticstoolbox", "viser"}
        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition('.')[0] in blocked:
                    raise AssertionError(f"unexpected backend import: {fullname}")
        sys.meta_path.insert(0, BlockOptional())
        from kinematic_calibration.handeye_cli import main
        from kinematic_calibration.handeye.opencv import evaluate_eye_to_hand_solution
        for command in [[], ['solve'], ['verify'], ['export'], ['compare'],
                        ['inspect'], ['influence'], ['bootstrap']]:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                try:
                    main([*command, '--help'])
                except SystemExit as error:
                    assert error.code == 0
                else:
                    raise AssertionError('help must exit')
            assert 'usage:' in output.getvalue()
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        metrics = evaluate_eye_to_hand_solution(poses, poses, np.eye(4), np.eye(4))
        assert metrics['position_error_max_m'] == 0
        assert metrics['orientation_error_max_rad'] == 0
        assert not blocked.intersection(sys.modules)
        """
    )
    source = Path(__file__).resolve().parents[2] / 'src'
    subprocess.run([sys.executable, '-I', '-c', script, str(source)], cwd=tmp_path, check=True)


def test_missing_opencv_has_actionable_library_and_cli_error(tmp_path):
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        import numpy as np
        sys.path.insert(0, sys.argv[1])
        class MissingOpenCV(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition('.')[0] == 'cv2':
                    raise ModuleNotFoundError('cv2 is unavailable', name='cv2')
        sys.meta_path.insert(0, MissingOpenCV())
        from kinematic_calibration.handeye_cli import main
        from kinematic_calibration.handeye.opencv import measure_handeye_motion
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        try:
            measure_handeye_motion(poses)
        except RuntimeError as error:
            assert 'kinematic-calibration[vision]' in str(error)
            assert isinstance(error.__cause__, ModuleNotFoundError)
        else:
            raise AssertionError('missing backend must fail')
        np.savez('samples.npz', schema_version=1, archive_type='eye-to-hand-observations',
                 base_T_ee=poses, camera_T_board=poses, candidate_indices=np.arange(3),
                 translation_candidate_count=1, rotation_candidate_count=2)
        assert main(['inspect', '--samples', 'samples.npz']) == 2
        assert 'cv2' not in sys.modules
        """
    )
    source = Path(__file__).resolve().parents[2] / 'src'
    result = subprocess.run(
        [sys.executable, '-I', '-c', script, str(source)], cwd=tmp_path,
        capture_output=True, text=True, check=True,
    )
    assert 'kinematic-calibration[vision]' in result.stderr
    assert 'Traceback' not in result.stderr
