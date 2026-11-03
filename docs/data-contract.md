# 标定数据中的关节列身份

`KinematicDataset` 的每一列必须对应一个确定的关节。新版 NPZ 使用 dataset schema 2，新增可选 `joint_names`：长度等于 DOF、名称唯一且非空，排列顺序与 `joint_positions[:, i]` 完全对应。名称区分大小写，不会自动排序、重排或根据列数猜测。

```python
dataset = KinematicDataset(
    joint_positions,           # [N, DOF]，实际编码器读数，rad
    observed_poses,            # [N, 4, 4]，独立观测，平移 m
    "camera_T_target",         # 或 base_T_ee
    sample_indices=sample_ids,
    joint_names=("Joint1", "Joint2", "Joint3"),
)
dataset.save("observations.npz")
```

加载模型后，`fit_dataset` 将数据名称与模型的正式关节名称按顺序比较。名称或顺序不一致会在求解前报错。NumPy/RTB URDF 工厂均提供正式关节名称；直接构造数值标定器时，应在 `from_screw_axes(..., joint_names=...)` 中声明名称，或显式向 `fit_dataset(..., model_joint_names=...)` 提供模型名称。只有列数相同不能证明列身份相同。

`as_robot_body`、数据拆分、去重、保存、加载和升级均保留名称。验证数据必须使用与保存模型一致的列顺序。采集清单现在要求 `joint_names`，导入器把它写入 NPZ；仿真测量使用后端的关节名称。二者都不会创建未经证实的占位名称。

旧 schema 1 及不含版本字段的数据仍可加载，旧的四个位置构造参数也保持原含义。旧数据没有名称时，新版不会凭空补齐：求解可以继续，但日志及结果中的 `joint_identity_verified` 明确标为 `false`。升级文件格式不会把未知身份变成已验证身份；需要先核对原采集系统的关节顺序，再填写真实名称。

实验工件仍使用 experiment schema 2。新实验保存带名称的训练和验证数据，并核对数据、模型和报告三者的列身份。已有实验目录中的无名数据仍可读取，返回的报告会记录身份未验证；加载不会改写原文件或哈希清单。

统一几何结果的 `joint_indices` 为零基索引，对应原求解报告中 `joint_names` 的模型列顺序，
与URDF声明顺序无关；固定关节没有编码器列，索引为 `null`。缺少名称的旧报告继续使用
XML中旋转关节的声明顺序作为回退，不能据此确认编码器列身份。

## 计划与实验样本身份

`load_pose_plan` 读取plan schema 2，保留访问顺序、原候选ID及选姿排列，并核对关节列、
限位、距离和诊断。它不重新选姿或检查碰撞；欠定但格式有效的计划可用于预览。

`demo --plan` 将原计划保存为实验目录中的 `plan.npz`，纳入manifest哈希。训练行 `i`
对应 `plan.candidate_indices[i]`，实验自身仍用 `0..N-1` 和 `N..N+V-1` 分别标识训练与
验证行。候选ID不混入实验样本ID命名空间。加载还会逐行检查训练配置与归档计划一致。

训练回放使用真实归档的训练观测并重新计算相应指标；默认回放使用独立验证观测。
这两种角色不会因行数相同而互换，也不会把手动探索的合成真值误差视为留出验证。

`plan --input-joints` 只读取给定NPY数组或NPZ中的 `joint_positions`、可选 `joint_names`
与 `sample_indices`，不要求也不读取 `observed_poses`。原采集ID按数据集相同的规则验证为
唯一、非负int64，随后保存在计划的 `source_sample_indices[candidate_count]` 中；访问行
`i` 对应 `source_sample_indices[candidate_indices[i]]`。训练回放同时显示该来源ID。
输入无名称时，计划报告的 `input_joint_identity_verified` 为false，输出的模型关节名称
不改变这一事实。来源文件哈希绑定读取的字节，移动计划后无需保留原输入文件。

## 输出与失败恢复

求解先准备完整的参数NPZ与JSON报告，报告绑定实际NPZ字节的哈希，再替换两个目标。
普通I/O失败或Python中断会恢复已有配对；恢复本身失败时保留备份并报告位置。多个文件
在进程崩溃时不保证一起完成，读取器继续核对配对哈希。

输入数据、URDF、外参和已消费的结果不能被同一次命令的其他输出覆盖，即使使用
`--force`。统一入口按启用阶段检查路径，仍允许复用或重新生成指定的手眼结果，以及
在输出目录内保留不冲突的名义URDF。实验和采集发布器也记录替换尝试，覆盖实际移动
已经完成后才收到中断的情况。
