# 移植来源

1. [PhilNad/robot-arm-kinematic-calibration](https://github.com/PhilNad/robot-arm-kinematic-calibration)
   commit `c21f498357a851525fbc84ff0a67c4ebaccdc6ba`。直接复制核心和测试后保留原 API，新增
   不依赖 RTB 的构造入口、按需导入与本地边界修复。MIT 原文保留于
   `third_party/licenses/robotkinecal-MIT.txt` 并随分发包打包；具体说明在 `src/robotkinecal/UPSTREAM.md`。
2. 用户授权的本地 `auto_calibrate_sim` 工作区快照（2026-11-03），Git HEAD
   `9488a2dbffc3e7635f3b0109a8839bd261a57cb0`。源目录含未提交改动，因此不能以 HEAD
   代表全部移入内容。原始文件哈希在 `local_source_snapshot.json` 和
   `src/kinematic_calib/SOURCE.json`。代码、板资源和简化 UR10 几何来自该工作区；未修改
   参考目录，也未搬入 DexSim、下载模型、生产采集或缓存。源工作区未提供统一 LICENSE，
   这里不额外宣称其代码和资源采用 MIT。
3. [chase6305/HandEyeCalibrationToolkit](https://github.com/chase6305/HandEyeCalibrationToolkit)
   固定子模块 commit `2de5138a7467e5bdc80a3e1c50d59d50e80c4cf4`。原生依赖和许可证以其
   仓库文件为准。只把已提交对象检出为子模块，未复制旁边工作区的未提交 C++/测试修改。

## 设计性变更

- 将数值求解、数据/验证、仿真与可视化拆分，旧命名空间保留兼容。
- 基础运行链无 DexSim、GPU、ROS 或 RTB 依赖；Viser 和视觉依赖按 extra 安装。
- 独立 NumPy URDF FK 与移植求解器进行数值交叉验证。
- 显式处理 CCTag spacing、旧 artwork 编码版本和 OBJ/检测帧转换。
- 保留手眼→几何→零偏路径，同时修复 base_T_ee 的模式选择和输出路径。
- 使用真正的位移范数统计位置误差，记录求解器代码来源，防止工件混用。
- 合成结果采用暂存发布、回滚、完整 manifest 及加载时一致性验证。

这些变更发生在本仓库中；上游仓库本身未被编辑或推送。
