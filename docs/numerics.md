# 数值约定与诊断

## POE 模型与标定残差

`SerialRobotKineCal.from_screw_axes(S, M)` 接受 `S.shape == (N, 6)`，
每行为基坐标系中表达的零位螺旋轴 `[v_x,v_y,v_z,w_x,w_y,w_z]`。
只支持转动关节，`||w|| = 1`、`v·w = 0`，且轴上点 `p` 对应
`v = p × w`。零位末端变换 `M` 为 `Base_T_EE`，平移单位米，关节角单位弧度。

正向运动学为

```text
Base_T_EE(q) = exp(S_1 q_1) ... exp(S_N q_N) M
```

完整姿态观测沿用上游 POE 伴随误差算法，其空间残差为

```text
y = log(Measured_Base_T_EE @ inv(Predicted_Base_T_EE))
y = [linear_twist, angular_twist]
```

`position_weight` 和 `orientation_weight` 分别乘上述线性、角向 twist 的
三个分量，最小化 `||W y||²`，因此数值权重在平方损失中也被平方。
完整姿态的线性 twist **不是独立的欧氏端点位移**：大旋转误差会与平移耦合，
权重也依赖基坐标原点的位置。只测位置时，残差才直接等于
`measured_position - predicted_position`。不能用空间 twist 的线性部分直接
声称毫米级位置精度；报告的 `position_errors` 和留出集验证使用实际端点
欧氏距离，`orientation_errors` 使用相对旋转的最小旋转角。

手眼外参不准确、位置观测与姿态观测互相冲突时，单纯调大 `position_weight`
未必能满足期望的端点精度。应先检查变换方向、同步、外参和原始观测，
并在独立验证集上检查实际位置和姿态误差。

## 零角和接近 π 的稳定性

旋转对数使用 SciPy 的旋转向量，以避免通过 `acos(trace(R)) / sin(theta)`
计算时在 π 附近损失精度。由 `w = rotvec(R)` 得到线性部分：

```text
v = (I - [w]/2 + c [w]²) t
c = (1 - theta/2 cot(theta/2)) / theta²
```

`theta` 很小时用 `c = 1/12 + theta²/720 + theta⁴/30240`。
这里 `[w]` 是叉乘矩阵。测试独立使用 `scipy.linalg.expm` 计算 4×4
李代数矩阵的指数，覆盖零角、接近 π、绕偏置点纯旋转及大平移。
恰好 π 时旋转轴正负不唯一，判定依据为重建变换一致，而非轴符号。

## 收敛、可辨识性与不确定度

回溯、收敛、发散判断使用相同加权目标。
`pre_update_twist_errors_norm` / `post_update_twist_errors_norm` 保留未加权范数；
`pre_update_objective_norm` / `post_update_objective_norm` 给出加权目标范数。
未接受的回溯步骤恢复模型并记录 `step_size = 0`。

公开 `IterationResult` 的端点误差与回归诊断描述更新前的线性化点。CLI几何报告的训练
位置/姿态误差则从最终保存的POE参数重新计算，并用同一结果检查误差阈值；秩和条件数
仍来自最后一次线性化。训练误差不替代 `validate-result` 的独立留出验证。

`has_converged` 表示迭代目标变化已达到停止阈值，不能替代精度或数据质量验收。
重复同一个关节配置，即使残差收敛，也无法确定完整模型。
应同时检查 `matrix_rank`、`N_PARAMS`、`singular_values`、
`condition_number` 和留出集误差。姿态不充分、测点位于末关节轴上或
联合估计未知外参时，都可能存在不可辨识自由度。

当回归矩阵秩不足或残差自由度不足时，`joints_uncertainty` 为 `NaN`，
表示无法可靠估计；伪逆在零空间中给出的零方差不能解释为高置信度。
满秩情况下保留上游局部线性估计，它假设所用残差统计模型合理，
不是实机精度保证。物理关节零位、几何误差、基座/工具外参间的规范自由度
还需由工作流层选择约束及可导出的参数。

只提供螺旋轴不能唯一确定 URDF 连杆坐标系，因此独立 POE 构造器
不直接输出 URDF joint origin。需要通过提供该模型的 URDF 后端处理坐标系
以及规范自由度，并验证导出前后 FK 一致。

## 噪声底附近的小更新停止

噪声观测的最小残差通常非零。回溯比较可能因浮点舍入拒绝所有缩步，
即使模型已经只剩约 `1e-12` 的参数校正；这与仍有显著下降空间但搜索区间
不合适的失败需要区别。

`solve` 提供 `update_tolerance=1e-10` 和
`relative_objective_tolerance=1e-8`。仅当回溯失败、先前至少存在一个已接受
更新、当前提出的参数校正范数不超过 `update_tolerance`，且相邻线性化点
平方目标的相对变化不超过 `relative_objective_tolerance` 时，报告
`has_converged=True`、`termination_reason='small_update'`、
`convergence_criterion='small_update_and_relative_objective_stall'`。
`IterationResult` 记录 `proposed_parameter_update_norm`、
`relative_objective_change`、`objective_scale`，原始未接受步长仍为零。
参数坐标混合米和弧度，这些是数值停止指标，不能作为机器人精度验收。

加权范数保留原数值供审计，停止比较则除以所有行权重中的最大值，
避免共同改变权重单位而改变绝对停止阈值。相对位置/姿态权重保持不变。
当 `damping=0` 时共同缩放权重不改变无正则最小二乘问题；
`damping>0` 时共同缩放会改变数据项与正则项的相对强度，不承诺结果不变。

该判据不把近似回归矩阵的梯度当作真实梯度。在 Python 3.12 的
30 样本噪声案例中，失败点的校正范数为 `1.595e-12`，独立有限差分
真实梯度范数仍为 `1.230e-6`；精确残差的 SciPy 继续优化仅改善约
`4.284e-9` 的相对平方目标。这是显式数值停滞，不是真实梯度严格为零。
对照中故意限制过大的搜索步长时，提出校正范数为 `0.03245`，
仍可降低约 `99.96%` 的目标，继续保留 `line_search_failed`。

## 每观测样本权重

`solve(..., sample_weights=weights)` 接受形状为 `(N_OBSERVATIONS,)`
的严格正、有限、实数向量，默认全部为 1。每个值是残差乘数；若目标是
平方损失中的 precision 权重 `p`，应传 `sqrt(p)`，而不是 `p`。
全局位置/姿态权重和样本权重相乘：

```text
W_i = sample_weights[i] * diag(position_weight × 3, orientation_weight × 3)
minimize sum_i ||W_i log(Measured_i @ inv(Predicted_i))||²
```

point 模式每个 `W_i` 只有三个位置分量。不接受零权重或隐式删除样本；
要排除观测需显式构建筛选后的数据集。若权重乘积在浮点运算中变成零或
非有限值，会拒绝求解。权重用于同一套线性方程、回溯目标和停止条件。
有阻尼时求解 `||W(Ak-y)||² + damping ||k||²`，共同缩放权重会改变
相对正则强度，因此不能当成正则问题的无影响单位变换。

每轮结果保留 raw `A_all`、`twist_errors`、`matrix_rank`、
`singular_values`、`condition_number`，同时提供：

- `row_weights`：展平后每行的完整残差乘数；
- `weighted_A_all`、`weighted_twist_errors`：实际进入数据项的矩阵和残差；
- `weighted_matrix_rank`、`weighted_singular_values`、
  `weighted_condition_number`：实际加权问题的数值可辨识性。

上述三个 weighted 诊断也由 `get_statistics()` 返回。正权重在精确数学中
不改变秩，但极小权重可能使部分方向在浮点精度下无法辨识；
验收加权拟合时应检查 weighted 诊断，不能仅使用 raw 满秩结论。

`joints_uncertainty` 使用实际加权回归矩阵及加权残差，通过设计矩阵的
SVD 计算局部方差信息，避免先构造 `AᵀA` 使条件数平方。秩不足或
残差自由度不足仍返回 `NaN`。为兼容上游，返回的是每关节四个参数方差的
范数，而不是直接的关节标准差或末端精度。方法标记为
`uncertainty_method='local_linear_weighted_residual_heuristic'`。
只有合理的统计噪声/白化假设才能赋予这类局部估计统计含义；
当权重来自 Huber 等 IRLS 的同一批数据残差时，它不包含权重估计的不确定度，
不能解释为校准参数置信区间。

## 显式 Huber IRLS

内存服务可选择样本级重加权；尺度由调用者按传感器噪声指定，不从验证集或拟合残差估计。
每轮工厂必须返回新的、几何相同的名义模型：

```python
from kinematic_calib import FitOptions, HuberOptions, fit_dataset
from robotkinecal import SerialRobotKineCal

fit = fit_dataset(
    dataset,
    calibrator_factory=lambda: SerialRobotKineCal.from_screw_axes(
        axes, home, joint_names=joint_names
    ),
    options=FitOptions(robust=HuberOptions(
        position_scale_m=0.0001, orientation_scale_rad=0.0002
    )),
)
```

当前仅支持固定外参的 robot-body 模式；相机观测需先提供两个已知外参。所有样本保留正权重，
诊断记录每个样本乘数、内外层停止状态及加权秩。权重来自端点位置与SO(3)角误差，内层仍
最小化空间twist平方残差，因此这是显式启用的IRLS启发式，不是精确的端点Huber最优化。
部署前仍需另用未参与拟合的数据验证。
