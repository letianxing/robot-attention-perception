# 最小打通 Runbook

## 0. 推荐：ROS 2 Jazzy + ROS4HRI 标准全链路

在 Jazzy workspace 中安装官方核心：

```bash
sudo apt install ros-jazzy-hri-msgs ros-jazzy-hri-actions-msgs \
  ros-jazzy-hri ros-jazzy-pyhri ros-jazzy-human-description \
  ros-jazzy-hri-rviz
```

从源码安装并编译 `hri_face_detect`、`hri_person_manager`、`hri_engagement`、
`hri_visualization`、`rqt_human_radar`。然后在本仓库 ROS2 workspace 编译：

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch attention_stack ros4hri_attention.launch.py \
  use_person_manager:=true use_pyhri_adapter:=true
```

若使用 Docker 的 Jazzy 运行时：

```bash
ROS_DISTRO=jazzy ROS_BASE_IMAGE=ros:jazzy-ros-base \
  docker compose -f docker-compose.mac.yml up -d --build
```

如果要完全使用官方视觉感知节点（而不是本仓库的 `vision-detection` backend），改用：

```bash
ros2 launch attention_stack ros4hri_attention.launch.py \
  use_official_detectors:=true use_person_manager:=true use_pyhri_adapter:=true
```

如果没有官方 GUI 包，设置 `use_visualization:=false`，仍可运行完整感知和注意力主链路。

## 1. Mac + Docker 真实 ROS2 全链路

```bash
cd ~/Golands/robot-attention-perception
scripts/setup_mac_first_test.sh
scripts/run_mac_first_test.sh
```

打开 `http://127.0.0.1:8092`，选择摄像头、`mac_builtin` 和 MacBook 麦克风后启动。绿色 ROS2 状态表示页面显示的 attention 已经经过 Docker 内 C++ 节点。默认 rosbridge 使用宿主机 `9090`。

摄像头和麦克风是两个独立选择器，可以任意组合。选择 `mac_builtin` 时界面只保留至少 1 个输入通道的设备；选择 `sipeed_6_plus_1_usb_array` 时只保留至少 8 个输入通道的设备。点击“停止”会同时停止视觉和声学采集。

独立 ROS roundtrip：

```bash
docker compose -f docker-compose.mac.yml up -d
.venv-mac/bin/python scripts/test_ros2_roundtrip.py --url ws://127.0.0.1:9091
docker compose -f docker-compose.mac.yml down
```

最小 Sipeed 原始数据测试：

```bash
cd ~/Golands/voice-detection
VENV=~/Golands/robot-attention-perception/.venv-mac
$VENV/bin/python -m voice_detection.cli list-devices
$VENV/bin/python -m voice_detection.cli record-live \
  --profile sipeed_6_plus_1_usb_array \
  --device <Sipeed设备编号> --seconds 10 \
  --output data/sipeed_8ch_48k_test.wav
$VENV/bin/python -m voice_detection.cli analyze-wav \
  --profile sipeed_6_plus_1_usb_array \
  --wav data/sipeed_8ch_48k_test.wav
```

不要用 QuickTime 或系统录音做阵列原始数据入口。报告应为 `channel_count=8`、`sample_rate_hz=48000`；随后拍手确认 8 通道都有响应，再从不同固定角度说话。控制台的方位显示两位小数，但真实 DOA 精度需要用已知角度治具标定。

心理物理机制回归：

```bash
.venv-mac/bin/python scripts/run_psychophysics_benchmark.py
```

输出双闪时间曲线、麦格克一致/冲突置信差，以及双人鸡尾酒目标锁定率和切换时延。它是机制回归，不替代真人实验。

## 2. 视觉：Mac 本地摄像头

```bash
cd ~/Golands/vision-detection
python3.12 -m venv .venv312
.venv312/bin/python -m pip install opencv-python aiohttp numpy mediapipe py-feat
brew install libomp ffmpeg
OPENCV_AVFOUNDATION_SKIP_AUTH=1 .venv312/bin/python scripts/local_vision_dashboard.py \
  --host 127.0.0.1 \
  --port 8080 \
  --camera-index 0 \
  --emotion-backend pyfeat \
  --stream-fps 20 \
  --inference-fps 5
```

浏览器打开 `http://127.0.0.1:8080`。确认右侧有 `people` 列表，字段包含 azimuth、gaze、facing、engagement 和 proxemic。

## 3. 声学：Mac 本地麦或 Sipeed 6+1

```bash
cd ~/Golands/voice-detection
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install sounddevice
python -m voice_detection.cli list-devices
python scripts/local_voice_dashboard.py --host 127.0.0.1 --port 8090
```

浏览器打开 `http://127.0.0.1:8090`。Mac 内置麦选择 `mac_builtin`；Sipeed 阵列选择 `sipeed_6_plus_1_usb_array`，设备选择 `max_input_channels >= 8` 的 UAC2 设备。

命令行直跑：

```bash
python -m voice_detection.cli run-live --profile mac_builtin --device default
python -m voice_detection.cli run-live --profile sipeed_6_plus_1_usb_array --device 2
```

没有硬件时先跑验收模板和离线 WAV 分析：

```bash
python -m voice_detection.cli write-asr-template \
  --output ~/Golands/voice-detection/data/asr_acceptance_template.jsonl

python -m voice_detection.cli analyze-wav \
  --profile sipeed_6_plus_1_usb_array \
  --wav /path/to/8ch_48k_s16le.wav
```

简单 TTS/self-echo 调试：

```bash
python -m voice_detection.cli speak --text "我听到了"
python -m voice_detection.cli speak --text "我听到了" --dry-run
```

## 4. 声学：4090 远端算法回路

4090 机器：

```bash
cd ~/Golands/voice-detection
python -m voice_detection.cli run-remote-server --host 0.0.0.0 --port 9097
```

Mac/Jetson 上位机：

```bash
python -m voice_detection.cli run-live-remote \
  --profile sipeed_6_plus_1_usb_array \
  --device 2 \
  --server-host 4090机器IP \
  --server-port 9097
```

## 5. 注意力：Mac 本地

```bash
cd ~/Golands/robot-attention-perception
python3 -m robot_attention_perception.cli dashboard \
  --host 127.0.0.1 \
  --port 8091 \
  --scenario fixtures/cocktail_effect_three_people.json
```

浏览器打开 `http://127.0.0.1:8091`。算法默认 `ros4hri_native`，也可以切到 `weighted_audio_visual`。

无 ROS2 联调时，用 JSONL bridge：

```bash
python3 scripts/attention_json_bridge.py --input-jsonl /tmp/perception_events.jsonl
```

JSONL 每行示例：

```json
{"type":"vision","stamp_ms":1000,"people":[{"person_id":"owner","voice_id":"voice_1","azimuth_deg":4.0,"face_visible":true,"gaze_score":0.8,"body_facing_score":0.8,"engagement_status":"engaged","proxemic_space":"social"}]}
{"type":"acoustic","stamp_ms":1040,"tracks":[{"track_id":"voice_1","voice_activity":true,"speech_probability":0.9,"clarity":0.82,"azimuth_deg":6.0}]}
```

有 ROS2 时启动 bridge：

```bash
# 视觉：/vision/people -> /humans/faces,bodies,persons + /vision/people_json
cd ~/Golands/vision-detection
python3 scripts/ros4hri_vision_bridge.py

# 声学：JSONL/工程输出 -> /humans/voices + /voice/acoustic_tracks
cd ~/Golands/voice-detection
python3 scripts/ros4hri_voice_bridge.py --input-jsonl /tmp/voice_tracks.jsonl

# 注意力：消费 voice/vision/robot topic，输出 attention gate
cd ~/Golands/robot-attention-perception
python3 scripts/ros4hri_attention_node.py
```

## 6. 记忆服务：Mac 本地

```bash
cd ~/Golands/hri-memory-service
cargo run -- --host 127.0.0.1 --port 8788 --data-dir ./data
```

健康检查：

```bash
curl http://127.0.0.1:8788/v1/health
```

批量导入 JSONL 事件：

```bash
cd ~/Golands/hri-memory-service
python3 scripts/ingest_jsonl.py \
  --input-jsonl /tmp/perception_events.jsonl \
  --subject-id owner \
  --robot-id reachy-mini \
  --session-id lab-001
```

## 7. 一键软件联调

```bash
cd ~/Golands/robot-attention-perception
scripts/run_local_software_stack.sh
```

默认会启动 memory、voice dashboard、attention dashboard、vision dashboard。当前不插硬件也可以启动页面和仿真；如果暂时不想打开摄像头：

```bash
START_VISION=0 scripts/run_local_software_stack.sh
```

## 8. 验收顺序

1. 先单独验证视觉 dashboard：确认摄像头、检测框、`people` JSON。
2. 再单独验证声学 dashboard：确认 Mac 麦或 Sipeed 8ch 可打开，VAD/azimuth 有输出。
3. 再启动 4090 远端回路：确认 Mac/Jetson 能把 raw 8ch 发过去并收到 track。
4. 再跑注意力 dashboard/JSONL bridge：确认声学和视觉 ID/azimuth 能匹配。
5. 最后把 ASR 场景测试记录写入记忆服务，按 Scope 查询和回放。

真实验收必须用实物麦克风、房间、噪声源和测试者跑你给的场景；当前自动化测试只验证软件链路和核心算法边界。
