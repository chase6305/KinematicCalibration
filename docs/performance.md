# POE 求解与规划回归矩阵性能

2026-11-03 第二阶段先测量第一阶段参考实现，再引入批处理。
`benchmarks/profile_solver.py` 固定随机种子 `20261103`，生成 6/7 自由度
任意转轴串联模型，分别使用 100、500、2000 个训练姿态和独立 80 个留出姿态。
训练噪声为每坐标轴 50 µm / 50 µrad，留出集无噪声。数据真值由独立的
闭式刚体旋转计算，不调用被测的 `SpatialMath.Twist3` 求值。
这些是合成模型计算结果，不代表实机标定精度。

## 测量方法

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u \
  benchmarks/profile_solver.py --repeats 3 \
  --output outputs/solver-profile.json
```

只计 `solve()` 的 wall time 和 process CPU time；数据构造、观测加载、
留出集验证和 cProfile 都在计时区间之外。每组重复三次报告中位数。
脚本另外在同一轮分别计 scalar `A_Matrix` 和 batch `regressor_blocks`，
并检查两者数值一致。规划计时只含回归矩阵构造，不含候选姿态生成、
D-optimal 贪心选择、碰撞或相机可见性检查。

测试机：AMD Ryzen 9 9950X，Linux x86_64，Python 3.10.0，NumPy 1.26.4，
SciPy 1.15.3，SpatialMath 1.1.18；OpenBLAS / OMP 线程数固定为 1。
原生工具包编译结束后重新取得下表参考值；机器仍可能有其他交互进程。
脚本将 CPU affinity、环境变量、库版本、源码 SHA-256、每次原始计时及
profile 写入 JSON，便于比较实际环境下的结果。

第一阶段标量参考为 commit `373fa63`；可在独立 worktree 加载该版本的
`src/robotkinecal`，用当前 benchmark 脚本重新计时。脚本兼容没有 batch
API 的旧版本。此次参考核心 SHA-256：

```text
6ae8c5afadc894a5a8adc3da1ee0a22d668e5d1dad5758012a2a2e08b251e67c
```

本轮 batch 核心 SHA-256：

```text
calibration.py: 8af25034c6fdb6f2fc0f46e620e4306e7672ebdf9ae8070af10955c18515d6ef
_batch.py:      445210f7020e22585308d4dc7c84e6e47a6fcbb07bc3a9c87788ad5e8e37d5ed
```

## 结果

所有组均为 3 次迭代、`converged`、满秩。
单位为秒；留出位置误差单位为 µm。

| DOF | 训练样本 | 参考 solve | batch solve | 加速 | scalar 回归矩阵 | batch 回归矩阵 | 留出位置 RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 100 | 0.3924 | 0.0108 | 36.3× | 0.0348 | 0.00067 | 30.581 |
| 6 | 500 | 1.7283 | 0.0253 | 68.4× | 0.1786 | 0.00161 | 13.988 |
| 6 | 2000 | 6.7695 | 0.0900 | 75.2× | 0.7071 | 0.00461 | 6.009 |
| 7 | 100 | 0.3916 | 0.0123 | 31.8× | 0.0421 | 0.00079 | 32.140 |
| 7 | 500 | 4.2689 | 0.0286 | 149.1× | 0.1826 | 0.00134 | 16.198 |
| 7 | 2000 | 11.9585 | 0.1212 | 98.7× | 0.7150 | 0.00560 | 6.949 |

参考与 batch 的留出 RMSE 最大差为 `3.64e-13 m` / `2.11e-13 rad`。
加速比例依赖自由度、样本数、回溯次数和运行环境，不能作为通用倍率承诺。
本轮原始输出在 `outputs/solver-profile-baseline.json` 和
`outputs/solver-profile-batch-final.json`；这些生成输出不提交版本库。

## 热点与改动

参考实现的 7DOF / 500 样本 cProfile 中，`solve` 共 8.473 秒，
`_twist_error_vector` 累计 6.405 秒，`forward_kinematics` 累计 4.888 秒，
`Twist3.exp` 累计 4.513 秒。累计时间互相包含，不能相加。
虽然只有 3 次迭代，回溯共调用 16 次整批残差求值；大量时间花在 Python
循环、重复指数映射、SpatialMath 中间对象及逐项参数检查上。
线性最小二乘不是该参考实现的主要瓶颈。

`_batch.py` 只沿关节维度循环，对全部观测同时计算旋转与平移。
每关节一次计算链前缀，用相同前缀同时得到末端姿态及伴随误差回归矩阵。
完整 pose 残差使用已独立验证的稳定 SE(3) logarithm 的批处理形式。
回溯仅重算 batch FK 和残差；不缓存跨模型更新的数据。
求解方程、四参数轴更新、回溯目标、停止判据和公开结果保持原语义。

保留 scalar `forward_kinematics`、`A_Matrix`、`Q_Matrix` 和 `_pose_log`
作为独立对照。新增 API：

```python
poses = calibrator.forward_kinematics_batch(joint_positions)  # [N,4,4]
blocks = calibrator.regressor_blocks(joint_positions)         # [N,6,P]
```

`regressor_blocks` 需先 `set_observations` 选择 3D point / 6D pose 维度。
对于 point 模式，它和 `A_Matrix` 一样仍返回空间 6 行回归块，求解时才应用
`[I, -skew(predicted_position)]` 端点投影。

## 等价性验证

`tests/core/test_batch.py` 覆盖 1/6/7DOF 任意轴、零角、极小角、接近 ±π、
模型更新/重置、point/pose 维度切换、单姿态及输入错误。
FK 和回归矩阵逐项对照保留的标量实现，绝对误差阈值 `2e-12`。
批处理对数另用 `scipy.linalg.expm` 的 4×4 李代数矩阵指数独立反验。
完整优化再以标量 FK/回归矩阵替身求解，与 batch 路径比较独立留出姿态，
不只验证两条路径各自收敛。

本轮基础环境核心测试 `70 passed, 1 skipped`（可选 RTB 模块跳过）；
RTB 环境 `80 passed`。原 RTB 构造和上游兼容测试继续保留。

## 多姿态规划与诊断的后续优化

2026-11-03 后续测量使用相同机器、单 BLAS/OMP 线程与五次重复中位数。
选姿表计时包含完整 D-optimal 选择，不含访问顺序规划。输入为示例 UR10、固定种子 42，
候选关节角均匀分布在 `[-2, 2]` rad。

| 候选 / 选择 | 优化前 | 优化后 | 加速 |
| --- | --- | --- | --- |
| 400 / 40 | 0.09239 s | 0.01051 s | 8.79× |
| 2000 / 80 | 0.50026 s | 0.08212 s | 6.09× |
| 5000 / 120 | 1.54558 s | 0.35890 s | 4.31× |

只为一个预测姿态初始化六维回归参数，避免给全部候选重复计算 FK、构造 SE3；
增量累加信息矩阵，每轮仍重新求解小矩阵以限制逆更新漂移。三组的选中索引完全相同，
奇异值谱最大绝对差为零。专项测试另用保留的标量回归矩阵和完整信息矩阵 logdet 独立对照。
原始计时保存在本次会话的 `/tmp/kincal-selection-{baseline,optimized}.json`。

访问排序独立采用最近邻及最多 20 轮 2-opt。默认 seed 42、400 候选选 40 姿态、零关节角
出发并返回，关节空间距离由 **263.136 → 142.614 rad**，缩短约 **45.8%**。
该值仅表示关节角欧氏路径长度，不是运动时间，也不包含碰撞检查。

求解诊断也复用一次 SVD 推导 rank / spectrum / condition，避免同一矩阵重复分解。
`profile_solver.py --repeats 5` 的六组 solve 中位耗时缩短 **1.12–1.67×**；
六组独立留出位置 RMSE 的差均为零。原始结果位于本次会话的
`/tmp/kincal-20261103-8h-baseline/solver-profile{,-single-svd}.json`。
不确定度仍通过加权设计矩阵 SVD 计算，另以已知奇异值的病态线性模型解析协方差验证。

另两项数组实现优化以五次单线程中位数检查：

| 路径 | 优化前 | 优化后 | 前后数组最大差 |
| --- | --- | --- | --- |
| UR10 终端 FK，3000 配置 | 0.2210 s | 0.1637 s | 0 |
| Camera/Target → Base/EE，10000 姿态 | 0.1803 s | 0.02769 s | 0 |

终端 FK 与连杆变换共享同一条链计算，省去中间字典和副本；返回的各连杆数组仍互不别名。
坐标转换以 NumPy 批量矩阵乘法与批量 SO(3) 投影替代逐姿态 Python 循环。
回归测试检查任意外参、接近 π 的姿态和变换闭合关系；输入数组保持原样。
原始结果在本次会话的 `/tmp/kincal-{forward,frame-conversion}-{baseline,optimized}.json`。

## 独立验证器

保存的 POE 模型以 SciPy 的一般 4×4 矩阵指数验证，不调用求解器的 FK 或 Rodrigues
实现。本轮只把观测维度的指数、矩阵逆和旋转误差转成批处理；仍保留一般矩阵逆，以及
逐行的三维距离累加顺序。前者兼容容差内的近刚体零位姿，后者保留几乎等距样本的最差
样本身份。输入校验、返回字段和 `[v,w]` 约定不变。

复现完整验证器计时，包括输入校验和结果构造，不包括数据生成、结果检查和 profile：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  benchmarks/profile_validation.py --repeats 3 --profile \
  --output /tmp/validation-profile.json
```

该脚本也可配合另一 checkout 的 `PYTHONPATH` 比较版本；输入使用独立闭式刚体真值，
每次计时后核对行顺序、关节配置、原始样本ID和最差样本ID。环境与前述单线程测试相同。
以下为标量参考（`04c5c95`）与批处理的三次中位数，单位为秒：

| DOF | 样本 | 标量参考 | 批处理 | 加速 |
| --- | --- | --- | --- | --- |
| 6 | 100 | 0.01279 | 0.00362 | 3.53× |
| 6 | 1000 | 0.09659 | 0.02841 | 3.40× |
| 6 | 5000 | 0.54583 | 0.16262 | 3.36× |
| 7 | 100 | 0.01920 | 0.00685 | 2.80× |
| 7 | 1000 | 0.13960 | 0.03459 | 4.04× |
| 7 | 5000 | 0.59105 | 0.15903 | 3.72× |

上述六组及零角、近π、大平移、近刚体零位姿、近等距误差共12组对照，全部返回字段
与保留的标量源码相同。新增解析运动测试不复写矩阵指数实现。
NumPy 1.22.4 / SciPy 1.10.1 的相关264项测试通过；不提高依赖下限。
另在声明的NumPy 1.22.0 / SciPy 1.10.0组合通过147项验证与NumPy工作流测试。
原始计时和逐字段对照记录在本次会话 `/tmp/kincal-validation-performance.json`，
生成输出不提交版本库。

## 实验工件加载

完整工件加载会重新预测名义、拟合、真值三组POE，以核对归档内容。原逐姿态SpatialMath
对象构造占5000样本加载profile时间约99%；现复用验证模块的独立 `predict_saved_poe`，
仍保留全部哈希、URDF、参数、样本身份、指标和验收检查。旧 `predict_poe(axes, home, joints)`
包装及其类型转换保持兼容；没有新增预测类或引擎接口。

同机、单线程，UR10、100训练样本、seed=20261103的真实schema 2合成工件，五次暖缓存
完整加载的中位数如下。导入和工件生成不计时，优化前后分阶段运行，并非交错采样：

| 验证样本 | 原加载 | 共享预测后 | 加速 |
| --- | --- | --- | --- |
| 1000 | 0.87928 s | 0.09367 s | 9.39× |
| 5000 | 4.37937 s | 0.44001 s | 9.95× |

工件字节、返回报告及数据/模型数组均未改变。独立重算预测的最大矩阵差为 `1.11e-14`，
重算指标最大差为 `1.55e-15`；沿用原 `1e-9` 工件比较容差，不声称不同指数实现逐位相同。
原始数据与脚本保存在本次会话 `/tmp/kincal-shared-prediction-{baseline,optimized}.json`
和 `/tmp/kincal_shared_prediction_benchmark.py`。可对相同工件分别检出前后版本，以
`load_experiment(path)` 的完整调用重复计时；这些数据不代表磁盘冷启动或实机精度。

## 访问排序的临时数组

2-opt每轮使用连续后缀切片，减少临时索引与布尔掩码。候选遍历、比较顺序、首个最大值
的选择、容差和20轮上限不变。实际优化前后公开函数在1100组随机、重复、并列及不同
起止方式的输入上，访问索引、各段距离和总距离均一致。

同机单线程、五次暖调用，前后调用顺序交错，6/7DOF、40/200/1000点、三种起止方式
共18组的process CPU中位耗时改善 **1.34–1.90×**。以下为6DOF、明确零点出发的开放路线：

| 点数 | 原CPU / wall | 优化后CPU / wall |
| --- | --- | --- |
| 40 | 2.169 / 2.290 ms | 1.485 / 1.528 ms |
| 200 | 23.543 / 25.501 ms | 15.854 / 17.802 ms |
| 1000 | 371.990 / 463.658 ms | 265.590 / 305.430 ms |

共享机器在该轮的load average约74–80，wall time包含明显的调度延迟，因此同时保留CPU
时间。原始配对记录为本次会话 `/tmp/kincal-route-audit-final.json`；独立复现当前版本：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  benchmarks/profile_planning.py --route-counts 40 200 1000 --repeats 5 \
  --output /tmp/routing-profile.json
```

给定的全部姿态均需访问时，`save_pose_plan(input_joints=...)` 直接计算一次回归诊断，
避免再执行逐轮贪心筛选；显式选择子集继续使用原D-optimal语义。

## 手眼闭环误差

OpenCV手眼诊断保留一般矩阵逆、逐行位置范数和全部输入/退化检查，把矩阵乘法、逆、
trace、clip及反余弦沿样本批处理。输入是真值变换独立指定的SE(3)姿态，加相机噪声，
没有机器人模型或名义FK测量。原实现与实际生产代码的完整报告对照，除代码来源信息外
全部相同，包括样本ID、影响排名、并列最差值和原始输入哈希。

同机、Python3.10 / NumPy1.26.4 / OpenCV4.11，BLAS/OMP/OpenCV线程均为1。
1000姿态、bootstrap20次的完整CLI调用计时包含读档、校验、求解和报告写入，排除导入、
输入构造、预热与结果复核。以下为三个配对运行的CPU中位数；影响分析单独注明单次对照：

| 工作流 | 原CPU | 批处理后CPU | 减少 |
| --- | --- | --- | --- |
| SHAH/LI比较 | 81.39 ms | 61.08 ms | 25.0% |
| SHAH bootstrap ×20 | 0.6680 s | 0.4558 s | 31.8% |
| LI bootstrap ×20 | 0.8748 s | 0.6501 s | 25.7% |
| SHAH影响分析，单次配对 | 31.973 s | 18.646 s | 41.7% |

影响分析单次wall time为 **32.071 → 18.737 s**，没有把单次结果声称为中位数；
其逐个留一求解的工作量不变。实际源码指纹、原始记录及profile在本次会话
`/tmp/kincal-handeye-performance-implemented/`。复现当前版本：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  benchmarks/profile_handeye.py --samples 100 1000 --repeats 3 \
  --output /tmp/handeye-profile.json

# 影响分析开销较大，显式选择；脚本每组还会执行一次完整预热
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  benchmarks/profile_handeye.py --samples 1000 --workflows influence \
  --methods shah --repeats 1 --output /tmp/handeye-influence-profile.json
```
