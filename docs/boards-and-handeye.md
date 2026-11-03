# 标定板与手眼标定

所有平移单位均为米，关节角和旋转误差单位为弧度。`A_T_B` 表示将 B 坐标系中的点变换到 A 坐标系。固定相机、末端安装标定板的约束为：

```text
Base_T_EE @ EE_T_Board = Base_T_Camera @ Camera_T_Board
Base_T_EE = Base_T_Camera @ Camera_T_Board @ inverse(EE_T_Board)
```

## 保留原有标定板

`kinematic_calibration.targets.get_board_config()` 默认返回原项目的六标记 CCTag 板。旧 SVG、PNG、OBJ 和材质已随包安装，不依赖 DexSim 的资源目录。

```python
from kinematic_calibration.targets import get_board_config

board = get_board_config("cctag-multi")
print(board.asset("svg"))       # 原始可打印文件
print(board.toolkit_size)        # (2, 3)
print(board.toolkit_spacing)     # (0.13, 0.13)
print(board.points)              # 检测坐标系中的六个中心，单位 m
```

此板为 **300 × 300 mm** 正方形，文件名中的 A4 不是 ISO A4 纸张尺寸。X 方向实际间距 130 mm，交错行的实际行距 65 mm。Toolkit 的旧接口使用 `board_space_y = 2 × 实际行距`，所以应传 **0.13 m**。直接将旧项目的 `space_y=0.065` 传给该接口会造成几何比例错误。

检测坐标系原点在第一个标记中心，X 朝图像右侧、Y 朝图像下侧。OBJ 原点在板中心，且 Y 朝上；`board.board_T_mesh` 显式记录二者之间的变换。`board.mesh()` 返回已经变换到检测坐标系的纹理网格，可以直接放到 `world_T_board` 下显示。它仅在调用时导入 trimesh/Pillow。

旧 CCTag 环形编码与 Toolkit 当前维护的 version-3 打印模板存在差异。`detect_toolkit_board()` 默认拒绝把旧板交给未知兼容性的检测器。已验证某个检测器版本支持现有实物板时，才使用 `allow_legacy_profile=True`。新板建议使用子模块维护的模板，并用对应的实测几何配置；模板和实物的实际尺寸必须一致。已有旧板检测程序可以继续输出中心点，再调用下面的 PnP 接口。

单圆标记无法唯一确定板面内的旋转。本项目的全位姿路径明确拒绝这种观测；可通过 Toolkit 的单标记加深度、`CalibrateAxyb` 点模型工作流使用它。`CalibrateAxyb` 估计板原点而非完整板安装旋转，不能把其结果当成完整 `EE_T_Board`。

## 从图像生成板位姿

安装 `vision` extra 后，OpenCV 非对称圆板可直接从 RGB/灰度图像检测。原板是黑底白圆，默认进行灰度反转。未检测到完整网格时返回 `None`；PnP 投影误差过大时抛出 `ValueError`。

```python
from kinematic_calibration.targets.detection import (
    detect_opencv_circles, pose_from_correspondences,
)

observation = detect_opencv_circles(rgb, intrinsic, distortion)
if observation is not None:
    camera_T_board = observation.camera_T_board
    print(observation.reprojection_rms_px)

# 或者继续用已有的 CCTag 检测程序提供中心点，顺序必须与 board.points 相同：
observation = pose_from_correspondences(
    board, ordered_pixel_centers, intrinsic, distortion,
    max_reprojection_rms_px=2.0,
)
```

`image_points` 是原始畸变图像中的像素坐标；PnP 接收相应内参和畸变系数，输出光学相机坐标系中的 `Camera_T_Board`。检测输入的彩色图像使用 RGB/RGBA 顺序。OpenCV 读入的 BGR 图像需要先转换颜色顺序。

Toolkit CCTag/圆板检测器通过 `detect_toolkit_board()` 懒加载，`depth_m` 的单位为米。CCTag 多板可不提供深度。当前 Toolkit 的 `DetectGridBoardPose` 提供成对的物点和像点，适配器按配置的板几何重排像点，并独立投影计算真实的 `reprojection_rms_px`。若旧版扩展只返回位姿、没有对应点，则像点和像素 RMS 返回 `None`，严格像素门限应拒绝该观测。其原生扩展须单独构建安装；仅添加子模块的 Python 源码路径不能替代编译。

## 独立求解手眼变换

```python
from kinematic_calibration.handeye import (
    solve_eye_to_hand, camera_observations_to_base_ee,
)

result = solve_eye_to_hand(base_T_ee, camera_T_board, backend="opencv", method="shah")
# 已构建安装 HandEyeCalibrationToolkit 后也可使用：
# result = solve_eye_to_hand(base_T_ee, camera_T_board, backend="toolkit")

measured_ee_poses = camera_observations_to_base_ee(
    camera_T_board, result.base_T_camera, result.ee_T_board,
)
```

两套输入均为同步采集的 `float64 [N,4,4]` 位姿，至少包含三对且须有两个独立方向的充分旋转。OpenCV 支持 SHAH/LI；Toolkit 适配器调用完整位姿 `Calibrate`，保留原生状态检查，并使用全部样本计算固定的板安装变换。适配器重新计算每个样本的物理闭环误差；不会把求解器内部重投影距离标为像素误差，也不会在后端缺失时静默切换算法。

闭环误差衡量当前模型与当前数据的一致性，不能代替独立验证数据或真实几何误差。用名义运动学位姿拟合的手眼结果继续反推同一批机器人位姿可能吸收运动学偏差；运动学标定应使用独立确定的外参，并保留未参与求解的验证集。

## 旧数据离线 CLI

`opencv-handeye-calib` 保留 `auto_calibrate_sim` 的离线工作流，包含：

```bash
opencv-handeye-calib solve --samples handeye_observations.npz --output handeye.json
opencv-handeye-calib verify --samples handeye_observations.npz --result handeye.json
opencv-handeye-calib export --samples handeye_observations.npz --result handeye.json \
  --base-t-camera base_T_camera.npy --ee-t-target ee_T_target.npy
```

同样保留 `inspect`、`compare`、`influence`、`bootstrap` 诊断命令。归档契约、候选姿态索引、内容指纹和结果验证随实现移植；旧项目产生的结果文件由于实现指纹不同，需要从原观察 NPZ 重新求解。CLI 报告中的 `accepted` 仅表示配置的闭环质量阈值通过。

所有写入命令均检查来源路径和已有输出，`--force` 不能覆盖输入样本。空或截断NPZ返回
简洁错误。`export` 先准备两个NPY，再发布到各自目录；普通失败或Python中断恢复原配对，
支持不同目录和文件系统，不保证进程崩溃时两个文件一起完成。

## 来源

运动学工作流相关的板资源、OpenCV 手眼 CLI、求解器和 RGBD 噪声实现移植自本地 `auto_calibrate_sim` 工作区快照；Git HEAD 为 `9488a2dbffc3e7635f3b0109a8839bd261a57cb0`，包含未提交修改，原文件哈希见 [来源记录](porting.md)。手眼工具包通过 [`third_party/HandEyeCalibrationToolkit`](../third_party/HandEyeCalibrationToolkit) 子模块独立维护。原始资源来源说明见包内 `assets/boards/ORIGIN.md`。

## 验证原生集成

`tests/integrations` 包含独立的合成手眼恢复、真实圆图检测、板资源/坐标转换和离线结果完整性测试。普通 `vision` 环境中，两个 native Toolkit 测试在扩展不可导入时明确跳过；在已构建并正确激活运行库环境的终端执行以下命令可验证原生路径：

```bash
python -m pytest -q tests/integrations/test_handeye_adapter.py \
  tests/integrations/test_detection.py -k optional_native
```

该命令分别检查 Toolkit 的完整位姿手眼恢复，以及生成 RGB 圆板图像的原生检测、像素 RMS 和已知 0.5 m 深度恢复。

### 平面位姿歧义

平面板在距离较远、接近正视或像素噪声较大时，可能有两个旋转相差明显、重投影误差却相近的候选解。仅通过像素 RMS 门限不能证明板姿态可靠。`BoardDetection.pose_diagnostics` 使用 OpenCV IPPE 的两个正深度候选，报告各自 RMS、候选间旋转/平移差和误差比。`ambiguous` 使用明确记录的启发式阈值：候选 RMS 比不超过 1.5，且旋转差至少 0.05 rad；该标志不是经过统计标定的置信区间。离线采集导入会记录该诊断并拒绝标记为歧义的帧，适合通过更近或有一定倾角的视角补采。共线物点或像点直接拒绝。

单标记深度接口返回的 4×4 数组是点测量的容器：当前 Toolkit 将其旋转部分设为单位矩阵，并未测出完整板旋转。多标记/PnP 路径则使用右手光学坐标系（相机 X 右、Y 下、Z 前），板 X 右、Y 下、Z 指向板内。两者的平移方向一致，但单标记的单位旋转不能作为多标记板姿态使用；本项目的全位姿入口会拒绝单标记输入。

现有旧板可以继续使用已有兼容检测器输出的有序中心点，再经 `pose_from_correspondences()` 得到板位姿。`legacy_artwork` 检查限制的是未经验证的原生检测器与环形编码组合，不要求为了本项目重新打印板。

### 本地原生验证范围

早期只读检查使用邻接工作区的既有二进制，只证明当时ABI/API可互操作。后续已从本仓库
固定子模块 `2de5138a` 单独构建验证；最新检测、图像实验和采集相关检查为91项通过。
各阶段范围见 [优化记录](worklog.md)，构建与运行库激活方法见 [原生构建](native-build.md)。
切换Python版本时须使用匹配ABI的扩展；正常导入不自动下载或构建子模块。
