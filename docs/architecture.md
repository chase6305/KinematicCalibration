# 模块与扩展边界

## 数值核心

`robotkinecal` 保留上游 `SerialRobotKineCal` / `CalibrationResult` API 和 POE 轴空间迭代。
`from_screw_axes(axes, home, joint_names=...)` 绕过 RTB；`axes` 是空间螺旋轴，旋转关节每轴
4 个几何修正参数，末端零位姿 6 个参数。RTB 构造器按需导入，以支持原模型与 URDF 导出。
每次独立拟合应新建 calibrator 或 `reset()`，求解会更新模型内部状态。

`kinematic_calib.service.fit_dataset` 负责数据帧选择、初始化和拟合。它不持有仿真器、不
启动服务器、不写文件。CLI 负责编排工件、来源记录和显式验收；新后端也复用同一服务。
统一 `calibrate` 入口可选择 NumPy 的 geometry-only POE 路线，复用相同拟合与工件复核，
并明确跳过 URDF 导出。固定 link frame 的导出与零偏旧流程继续由可选 RTB 适配器提供。

RTB 的观测列仍按所选链排列；调用其 FK 时才映射到全局关节索引，未选关节置零。
完整 URDF 原点导出当前要求所选旋转关节直接相连、首关节直接连接基座，末端为最后
关节的直接固定工具边。部分可辨识原点也要求相邻旋转关节直接相连。现有表示不支持的
固定路径会明确拒绝；NumPy POE 路线可用于这类模型的端点拟合与验证。

## 后端

实现 `KinematicsBackend` 的新引擎需提供关节名称与有限限位、Base/End 名称、
`forward(q)` 和 `link_transforms(q)`。引擎的输出使用相同的米/弧度和 A_T_B 约定。
NumPy 后端读取指定 URDF 路径，忽略 mesh，不执行物理与碰撞计算。Viser 是独立观察者，
展示名义/真值骨架和标定后的端点预测；板贴在显示用真值 EE 坐标，未声称恢复全部关节框架。
NumPy 另提供 `forward_batch(joint_positions[N,DOF])`，返回 `Base_T_End[N,4,4]`，
支持空批次；实验验证用它批量计算模型预测。此扩展不改变其他后端的协议要求，
与单姿态 FK 一样不检查关节限位、碰撞或运动可行性。
可视化依赖Viser1.1及以上，使用世界坐标线宽；初始化失败或被中断时关闭已启动的服务，
成功返回后由调用者关闭。

`planning.select_informative_poses` 负责模型信息量选姿；`routing.plan_visit_order` 仅消费关节
配置并缩短访问距离，不依赖模型、传感器或求解器。`save_pose_plan` 编排两步，保留候选身份
和选姿/访问索引映射。重排不改变所选姿态集合，因此不改变回归矩阵的秩与奇异值谱。
`load_pose_plan` 检查归档身份与距离；实验按保存的访问顺序调用 `measure_configurations`。
完整计划纳入实验manifest，候选ID通过计划映射保留，实验内训练/验证样本ID仍互不重叠。
Viser显式选择训练或验证集，分别使用对应的观测、预测、指标及图像，不将训练误差称为验证。

`saved_model.load_fitted_model` 读取现有参数NPZ、配套报告和URDF，检查配对关系；不启动CLI，
也不重验缺少的原训练数据。`MeasuredPlayback` 将已有观测转换到拟合模型的显示坐标，
保留原始样本身份并区分训练重叠。它与合成 `ExperimentPlayback` 共用渲染函数，但不构造
真值模型；手动配置没有观测时不产生测量误差。详见 [已有工件回放](measured-playback.md)。

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
