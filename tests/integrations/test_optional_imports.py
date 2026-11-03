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
