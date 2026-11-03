# 模块与扩展边界

## 数值核心

`robotkinecal` 保留上游 `SerialRobotKineCal` / `CalibrationResult` API 和 POE 轴空间迭代。
`from_screw_axes(axes, home, joint_names=...)` 绕过 RTB；`axes` 是空间螺旋轴，旋转关节每轴
4 个几何修正参数，末端零位姿 6 个参数。RTB 构造器按需导入，以支持原模型与 URDF 导出。
每次独立拟合应新建 calibrator 或 `reset()`，求解会更新模型内部状态。

`kinematic_calib.service.fit_dataset` 负责数据帧选择、初始化和拟合。它不持有仿真器、不
启动服务器、不写文件。CLI 负责编排工件、来源记录和显式验收；新后端也复用同一服务。

## 后端

实现 `KinematicsBackend` 的新引擎需提供关节名称与有限限位、Base/End 名称、
`forward(q)` 和 `link_transforms(q)`。引擎的输出使用相同的米/弧度和 A_T_B 约定。
NumPy 后端读取指定 URDF 路径，忽略 mesh，不执行物理与碰撞计算。Viser 是独立观察者，
展示名义/真值骨架和标定后的端点预测；板贴在显示用真值 EE 坐标，未声称恢复全部关节框架。

基础包依赖 NumPy、SciPy、SpatialMath。OpenCV、Viser、RTB 和原生工具包分别位于边界，
模块导入不触发下载、构建或 GPU 初始化。新增 MuJoCo / PyBullet / 实机控制器时，应实现
采集与 FK 边界，不把引擎对象传进数值求解器。

## 数据与观测

`KinematicDataset` 是同步姿态/关节对，检查有限值、旋转刚体性质、唯一 sample_indices。
图像检测和 PnP 只生成 `camera_T_target`；机器人 FK 不替代传感器测量。固定相机、板装在
末端的关系为：

```text
Base_T_EE @ EE_T_Board = Base_T_Camera @ Camera_T_Board
Base_T_EE = Base_T_Camera @ Camera_T_Board @ inverse(EE_T_Board)
```

未知外参时可以拟合 Camera/Target 复合模型；该模型不唯一确定物理 Base/EE 的边界。
手眼结果依赖用于求它的机器人位姿，若位姿来自待标定的名义 FK，要考虑模型误差反馈；
不能把同一批名义 FK 数据当作独立真值而宣称物理精度改善。

## 验证与工件

训练与验证样本必须隔离；通过不同随机种子生成的合成姿态使用不重叠身份。实验 schema 2
把数据、POE、URDF、报告的 SHA256 写入 manifest；加载时再核对矩阵、样本身份、模型 FK
和指标一致性。这用于检查意外修改与混用，不提供数字签名或不可信输入的安全认证。

求解诊断包括终止原因、参数数、秩、条件数。良好的训练残差不保证可观测、可部署或能
准确归因到每个物理关节。实验的验收范围是合成数值诊断；实机还需独立传感器、目标工作
空间数据及具体精度阈值。URDF 导出写入新文件，保留名义模型。

## 扩展建议

- 新检测器返回有顺序的板点、像素点、Camera_T_Board 和可核对的重投影误差。
- 新采集器记录编码器和相机时间戳，保留拒绝原因和原始 sample ID。
- 新求解器复用数据契约与验证，独立实现结果到 POE/URDF 的转换。
- 新可视化层消费已验证的结果，不改变训练样本或拟合参数。
