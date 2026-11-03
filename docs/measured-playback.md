# 已有标定结果与观测数据回放

`kincal view-fit` 读取 `kinematic-calib solve` 已有的参数 NPZ、配套 JSON、名义 URDF 与
观测数据集，不要求合成实验目录，也不调用 DexSim 或 RTB。安装可视化 extra 后运行：

```bash
uv pip install --python .venv/bin/python -e '.[visualization]'
kincal view-fit --parameters outputs/fit.npz --dataset validation.npz \
  --urdf robot.urdf --host 127.0.0.1 --port 8080
```

JSON 默认取参数文件同名 `.json`；用 `--result` 指定其他位置。`--duration 5` 可在五秒后
停止服务。加载前核对参数与报告、URDF 的 SHA-256，以及模型模式、关节顺序和样本身份。
报告中的 `accepted` 是当时保存的拟合结论；回放不会重新授予标定验收或实机精度认证。

## 坐标模式

所有变换均为 `A_T_B`（B 到 A），长度为米、角度为弧度。

| 拟合与观测 | 显示坐标 | 处理方式 |
| --- | --- | --- |
| 机器人本体模型，`base_T_ee` 观测 | Base | 名义 URDF、拟合 POE 与观测直接对照 |
| 机器人本体模型，已知外参的 `camera_T_target` 观测 | Base | 将观测转换为 `Base_T_Camera @ Camera_T_Target @ inverse(EE_T_Target)` |
| 自校准复合模型，`camera_T_target` 观测 | Camera | 直接显示拟合的 Camera/Target POE；名义骨架使用保存的初始化 gauge |

回放仍为 `camera_T_target` 的已知外参数据时，需要拟合时绑定的两份原始文件，其字节
SHA-256 必须与保存的指纹一致，不能临时替换为一套新外参：

```bash
kincal view-fit --parameters outputs/fit.npz --dataset camera-validation.npz \
  --urdf robot.urdf --base-t-camera base_T_camera.npy --ee-t-target ee_T_target.npy
```

自校准模式的 `estimated_camera_T_base`、`estimated_EE_T_target` 仅用于对齐名义模型的
初始化坐标选择，不能解释为已独立恢复的物理相机或工具外参。该模式不接收另外指定的固定外参。
Camera 使用 OpenCV 光学轴（X右、Y下、Z前）设置观察方向；数据矩阵本身不作显示性改写。

## 样本与交互

界面区分训练观测、与训练有重叠的观测、留出观测；标签依据保存的样本 ID 与原始观测指纹，
不是仅凭文件名不同判定独立。留出标签表示归档中的样本身份未重叠，不证明采集过程物理独立。
显示数据的观测帧与训练来源不同、无法按原始指纹直接比较时，标签会注明独立性未验证。

滑块切换记录中的关节配置与真实观测，显示欧氏端点误差及 SO(3) 角度误差。
勾选 **Explore joint angles** 后可手动查看名义骨架和拟合端点；手动配置没有传感器观测，
界面会隐藏测量坐标框并显示 **No measurement for this configuration**，不借用邻近样本，
也不把名义 FK 当作测量。

真实工件回放仅显示名义骨架、拟合端点和已有观测，不虚构真值模型，也不默认放置某种标定板。
拟合 POE 未提供校准后的中间连杆坐标，因此蓝色端点并不代表完整校准机器人骨架。
