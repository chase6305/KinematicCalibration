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
