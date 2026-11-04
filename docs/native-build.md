# 构建固定的手眼子模块

基础运动学和 OpenCV 路线无需原生构建。需要 Toolkit CCTag 检测或其手眼算法时，可以在
项目输出目录中单独构建并安装，避免写入现有 Python/系统环境。

先准备 C++17 编译器、CMake 3.23+、Conan 2.9+ 和已配置的 Conan profile，初始化子模块。
指定用于 Python 扩展的目标解释器，然后执行：

```bash
git submodule update --init --recursive
python scripts/build_toolkit.py --python .venv/bin/python --jobs 2
```

已有全部依赖缓存时增加 `--offline`，不会联网或构建缺失的依赖。CMake 不在 PATH 或系统
版本较旧时，用 `--cmake /path/to/cmake` 指定；Conan 也可由 `--conan` 指定。脚本不会自动
创建或覆盖 Conan profile，也不往子模块写 CMakeUserPresets.json。

默认依赖输出、编译、CTest 和安装均位于 `outputs/native-toolkit/`；`--build-dir` 可以
改位置，安装不会写入目标解释器。成功后 `native-build.json` 记录源 commit、工作区状态、
目标解释器和安装位置。不同 Python ABI 应使用不同构建目录重新构建。

启用运行时环境并加载安装包：

```bash
source outputs/native-toolkit/build/Release/generators/conanrun.sh
export PYTHONPATH="$PWD/outputs/native-toolkit/install/lib/python${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -c 'from handeye_calibration import calibration; print(calibration)'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/integrations/test_handeye_adapter.py tests/integrations/test_detection.py \
  -k optional_native
```

`--skip-tests` 可以关闭构建与执行原生单元测试；默认会执行。失败保留构建目录和完整工具
输出，便于定位依赖/编译/API 问题。脚本不改变子模块版本；更新源版本需显式更新 Git 子模块。
