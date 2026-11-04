import xml.etree.ElementTree as ET

import numpy as np
import pytest

from kinematic_calibration.targets import BoardType, get_board_config


def test_multi_board_physical_pitch_and_legacy_detector_spacing():
    board = get_board_config("cctag-multi")
    np.testing.assert_allclose(
        board.points,
        [
            [0, 0, 0],
            [0.13, 0, 0],
            [0.065, 0.065, 0],
            [0.195, 0.065, 0],
            [0, 0.13, 0],
            [0.13, 0.13, 0],
        ],
    )
    assert board.toolkit_spacing == (0.13, 0.13)
    assert board.toolkit_size == (2, 3)
    # The first printed marker is 52 mm from each edge. The source OBJ uses
    # a centered frame with Y up and the printed face at z=1 mm.
    np.testing.assert_allclose(
        board.board_T_mesh @ [-0.098, 0.098, 0.001, 1], [0, 0, 0, 1], atol=1e-15
    )
    np.testing.assert_allclose(board.corners[0], [-0.052, -0.052, 0])


@pytest.mark.parametrize("kind", list(BoardType))
def test_packaged_assets_exist_and_material_texture_is_resolvable(kind):
    board = get_board_config(kind)
    for extension in ("obj", "mtl", "png", "svg"):
        assert board.asset(extension).is_file()
    material = board.asset("mtl").read_text()
    for line in material.splitlines():
        if line.startswith("map_Kd "):
            assert (board.asset("mtl").parent / line.split(maxsplit=1)[1]).is_file()


def test_six_marker_print_centers_match_points_in_detected_frame():
    board = get_board_config()
    root = ET.parse(board.asset()).getroot()
    centers = sorted(
        {
            (
                float(node.get("cx").removesuffix("mm")) / 1000,
                float(node.get("cy").removesuffix("mm")) / 1000,
            )
            for node in root.findall(".//{http://www.w3.org/2000/svg}circle")
            if node.get("fill") == "black"
        }
    )
    predicted = board.points[:, :2] + board.origin_from_top_left_m
    np.testing.assert_allclose(centers, sorted(map(tuple, predicted)))


def test_asymmetric_circle_geometry_matches_print():
    board = get_board_config("opencv-acircle")
    root = ET.parse(board.asset()).getroot()
    centers = sorted(
        (float(node.get("cx")) / 1000, float(node.get("cy")) / 1000)
        for node in root.findall(".//{http://www.w3.org/2000/svg}circle")
    )
    predicted = board.points[:, :2] + board.origin_from_top_left_m
    np.testing.assert_allclose(centers, sorted(map(tuple, predicted)))


def test_single_marker_is_one_point_and_centered():
    board = get_board_config("cctag-single")
    np.testing.assert_array_equal(board.points, [[0, 0, 0]])
    np.testing.assert_allclose(board.board_T_mesh @ [0, 0, 0.0005, 1], [0, 0, 0, 1])
