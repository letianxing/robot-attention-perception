# robot-attention-perception

## Mac 第一次真实测试

```bash
cd ~/Golands/robot-attention-perception
scripts/setup_mac_first_test.sh
scripts/run_mac_first_test.sh
```

停止整套服务：

```bash
scripts/stop_mac_first_test.sh
```

统一控制台：`http://127.0.0.1:8092`。摄像头与麦克风由 macOS 原生进程采集，ROS2 Humble、ROS4HRI 和 C++ S1-S5/BCI attention 在 Docker Desktop 中运行。点击“启动感知”后，页面显示实际走过 ROS2 的 attention 结果。

本地 ASR 可选安装：

```bash
cd ~/Golands/voice-detection
scripts/setup_local_asr.sh
```

模型就绪后，一键脚本会自动启用 `Vosk 中文离线 ASR -> ROS2 attention gate -> S2S -> macOS say`。若另行提供 whisper.cpp 模型则优先使用 whisper；没有模型时视觉、声学、ROS2 和注意力仍正常运行。

RealSense 到货后选择 `realsense` source；Sipeed 到货后选择 `sipeed_6_plus_1_usb_array` profile。两者都复用 `/perception/*` 强类型接口。

用于机器人“鸡尾酒效应”感知的仿真与架构项目。目标不是把 ASR、视觉、权限、对话混成一个黑盒，而是把每层边界拆清楚：

- 声学层：多设备采集、VAD、DOA/声源跟踪、回声抑制、可选分离、ASR 转写。
- 视觉层：复用 `~/Golands/vision-detection` 的人物、身份、朝向、表情、手势信号。
- 注意力层：融合声音方向、视觉朝向/视线/手势、机器人自播报状态，输出“看向哪里、听哪个声源、这句话是否像是在对机器人说”。
- ASR 测试层：只记录最终转写、语言/清晰度、差异和时间，不负责意图、权限或动作结论。

## 当前结论

`~/Golands/vision-detection` 已扩展 `/vision/people`，可以给注意力层提供每个人的视觉方位、凝视、朝向、邻近空间、手势和情绪信号。`~/Golands/voice-detection` 已有 C++ 核心和 Python 工程入口，支持 Mac/Windows 单麦降级、Sipeed 6+1 8ch profile、DOA、beamforming、VAD、轻量降噪、self-echo 判定、ROS4HRI voice topic 映射和 4090 远端算法服务骨架。

基于一张 RTX 4090 的机器，声学 + 视觉 + 注意力融合可以放在同一台机器上跑。风险点不是显存是否够用，而是低延迟链路不要把神经语音分离、长上下文 LLM、高清多相机和高频表情模型全部常驻在关键路径里。

## 快速运行

## ROS4HRI 标准模式

感知后端仍使用 `vision-detection` 与 `voice-detection` 的现有算法，但对外优先使用 REP-155 的
`/humans/*` 语义。ROS2 工作区建议安装官方接口、库和人员融合节点：

```bash
sudo apt install ros-humble-hri-msgs ros-humble-hri-actions-msgs \
  ros-humble-hri ros-humble-human-description
```

在已构建本仓库接口后启动标准注意力栈：

```bash
ros2 launch attention_stack ros4hri_attention.launch.py \
  use_person_manager:=true use_pyhri_adapter:=true
```

该 launch 会启动官方 `hri_person_manager`（消费 `/humans/candidate_matches`），并用
`pyhri.HRIListener` 读取标准人类数据，转换到现有的强类型 attention 输入。GUI 验证工具独立启动：

```bash
ros2 run rqt_human_radar rqt_human_radar
ros2 launch hri_visualization visualization.launch.py
```

自动检查标准 topic 和活动节点：

```bash
scripts/check_ros4hri_topics.sh
```

`/attention/look_at` 保留为内部融合输出；需要实际驱动头眼时，执行器应实现 ROS4HRI
`interaction_skills/action/LookAt`，服务端点为 `/skill/look_at`。同理，语音输出可逐步适配
`/skill/say`，对话可适配 `/skill/chat`。

```bash
cd ~/Golands/robot-attention-perception
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m robot_attention_perception.cli algorithms
python3 -m robot_attention_perception.cli simulate --scenario fixtures/cocktail_effect_three_people.json
```

仿真会输出每一帧的注意力目标、声源 track、匹配到的视觉 person、是否建议 listen、是否像在对机器人说，以及 ASR 记录。

C++ 核心：

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/robot_attention_cli algorithms
./build/robot_attention_cli simulate
```

Mac 本地注意力 dashboard：

```bash
python3 -m robot_attention_perception.cli dashboard \
  --host 127.0.0.1 \
  --port 8091 \
  --scenario fixtures/cocktail_effect_three_people.json
```

打开 `http://127.0.0.1:8091`。

无 ROS2 时的 JSONL bridge：

```bash
python3 scripts/attention_json_bridge.py --input-jsonl /tmp/perception_events.jsonl
```

软件联调启动脚本：

```bash
# 只启动本地软件，不依赖阵列麦硬件
scripts/run_local_software_stack.sh

# 单独启动
scripts/run_memory.sh
scripts/run_voice_dashboard.sh
scripts/run_attention_dashboard.sh
scripts/run_vision_dashboard.sh

# 4090 机器上启动声学算法服务
scripts/run_4090_acoustic_server.sh
```

默认端口：memory `8788`、voice dashboard `8090`、attention dashboard `8091`、vision dashboard `8080`。可通过 `HOST`、`PORT` 或对应端口环境变量覆盖，例如 `START_VISION=0 scripts/run_local_software_stack.sh`。

## 文件

- `docs/ARCHITECTURE.md`：完整声学、视觉、注意力融合架构。
- `docs/ARCHITECTURE_DIAGRAMS.md`：四系统架构图、声学回路图、ROS4HRI 数据链路图。
- `docs/TOPICS.md`：视觉、声学、注意力、记忆服务 topic/API 清单。
- `docs/FREQUENCY_AND_LATENCY.md`：频率不一致处理和低延迟优化预算。
- `docs/PURCHASE_LIST.md`：Mac + Sipeed + 4090 + 未来 Jetson 的最小采购清单。
- `docs/RUNBOOK.md`：四系统最小启动和联调顺序。
- `docs/ASR_TEST_BOUNDARY.md`：对你给的语音识别测试文档做系统边界映射。
- `docs/HARDWARE_AND_LATENCY.md`：麦克风设备兼容、延迟、显存、磁盘估算。
- `interfaces/ros2_msgs/`：后续落 ROS2 消息时建议的接口。
- `robot_attention_perception/`：可插拔 Python 融合内核、异步融合缓存和 dashboard。
- `include/`、`src/`：C++ 注意力核心，默认 `ros4hri_native`，可注册替代 `AttentionAlgorithm`。
- `scripts/ros4hri_attention_node.py`：ROS2 轻量注意力节点，消费视觉/声学 bridge 输出并发布 attention gate。
- `../vision-detection/scripts/ros4hri_vision_bridge.py`：视觉到 ROS4HRI 和 `/vision/people_json` 的桥接脚本。

## 和现有项目的关系

```text
vision-detection             -> 视觉输入、dashboard、/vision/people
voice-detection              -> 鸡尾酒声学前端、ROS4HRI voice 输入、S2S 前置目标音频
robot-attention-perception   -> ROS4HRI attention、mixed-rate fusion、S2S gate
hri-memory-service           -> Rust 本地可追溯事件/向量/文本/结构化存储
```
