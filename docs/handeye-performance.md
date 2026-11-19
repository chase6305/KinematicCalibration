# OpenCV 手眼重复求解的性能

2026-11-19 对比 `234ec74` 与本轮实现。两种原生方法仍使用 OpenCV 的
`calibrateRobotWorldHandEye`，帧关系仍为
`Base_T_EE * EE_T_Board = Base_T_Camera * Camera_T_Board`。

本轮减少重复计算和中间对象：

- 相对旋转矩阵按样本批量计算，旋转向量仍逐项调用 OpenCV Rodrigues，复用输出和
  Jacobian 数组；保留其近零及 π 附近的数值约定。
- 运动激励的秩和奇异值谱来自同一次 SVD，使用原 NumPy 默认秩容差。
- 每次求解只校验一次两组观测；内部运动诊断和闭环误差复用已校验数组。
  独立调用公开评估函数仍完整校验输入，求解返回的外参也继续校验。
- 原生求解直接接收按样本排列的 NumPy 数组，避免构造四组逐姿态 Python 列表。
- 普通求解报告直接使用求解器返回的运动诊断。

位置误差仍使用原逐向量范数和一般矩阵逆，保持近并列误差的样本身份及容差内近刚体
输入的含义。数据字段、激励门限、方法选择、bootstrap 抽样及留一分析的求解次数均未改变。

## 测量范围

AMD Ryzen 9 9950X，Linux x86_64，Python 3.10.0，NumPy 1.26.4，SciPy 1.15.3，
OpenCV 4.11.0；BLAS、OMP、MKL、NumExpr 和 OpenCV 线程均为 1。
seed 为 314159。输入由独立指定的 SE(3) 真值与小幅相机噪声构造，不使用机器人名义 FK。

同一进程加载前后两版完整 CLI 和求解模块，每轮交替调用顺序。每组先各预热一次完整
工作流，普通组重复五次，留一组重复三次，报告 process CPU 中位数。
计时包含 CLI 参数处理、读档、校验、原生求解、报告序列化及原子写入（如适用）；
导入、输入生成、预热、结果读取和逐字段检查在计时外。`inspect` 的 stdout 在内存中捕获。

| 1000 姿态工作流 | 原 CPU | 优化后 CPU | 减少 |
| --- | --- | --- | --- |
| 运动检查 | 4.613 ms | 3.651 ms | 20.8% |
| SHAH 求解并保存报告 | 18.759 ms | 14.469 ms | 22.9% |
| LI 求解并保存报告 | 25.079 ms | 20.110 ms | 19.8% |
| SHAH/LI 比较 | 28.647 ms | 25.065 ms | 12.5% |
| SHAH bootstrap ×20 | 208.682 ms | 170.365 ms | 18.4% |
| LI bootstrap ×20 | 333.227 ms | 300.314 ms | 9.9% |
| SHAH 留一影响分析，三次中位数 | 9.758 s | 7.925 s | 18.8% |

留一分析的 wall 中位数为 9.763 → 7.933 s。其余组的原始 wall/CPU 计时也保留在
本次会话 `/tmp/kincal-handeye-phase2/paired/report.json`。100/1000 姿态共13组完整报告
除时间和实现指纹外逐字段相同，包括版本信息、候选 ID、最差样本、影响排名、bootstrap
失败记录和输入指纹。输入归档字节未改动。上述比例只适用于这些测量，不代表实机精度。

## 数值与契约核对

NumPy 1.26.4 / OpenCV 4.11.0，以及 NumPy 1.22.0 / OpenCV 4.8.0 各完成464组对照。
每组均比较原公开实现与优化后的全部返回字段或异常类型和说明：370组成功结果完全
一致，94组拒绝结果一致。范围包括3–100姿态、两种原生方法、float32/64、C/F/跨步/
负步长只读输入、近单轴运动、近零/近 π 旋转和输入校验顺序。

提交的回归测试另以解析旋转与平移范围检查激励量，核对原列表调用与新数组调用的
原生结果，覆盖激励门限、单次校验/单次分解和报告复用。测试不会把通过激励检查等同于
原生方法在任意数据上均可成功求解。

复现当前版本的完整调用计时：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python benchmarks/profile_handeye.py --samples 100 1000 \
  --workflows inspect solve compare bootstrap --repeats 5 \
  --output /tmp/handeye-profile.json

# 留一分析成本更高，显式选择；每组还执行一次完整预热
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python benchmarks/profile_handeye.py --samples 1000 \
  --workflows influence --methods shah --repeats 3 \
  --output /tmp/handeye-influence.json
```

JSON记录源码哈希、依赖版本、线程环境、CPU affinity、启动时负载、每次原始计时、
输入指纹及去除来源记录后的报告指纹。测量按共享机器的资源约定串行运行。
