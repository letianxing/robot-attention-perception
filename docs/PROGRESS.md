# 四体仿生感知系统进度

最后更新：2026-09-08

## 当前状态

- 测试服务：已关闭；`8080`、`8090`、`8092`、`8788` 不监听。
- Docker：本项目 ROS2 容器已关闭；已按要求删除无关的 `brain-ros2` 容器和镜像。
- 当前硬件：Sipeed 6+1 已到但尚未连接；视觉先用 Mac 内置摄像头；RealSense D405/D435i 尚未购买/接入；INDEMIND M1 预留 Linux/ROS source。
- 已删除无关的 `brain-ros2` 容器和镜像；本项目 rosbridge 默认恢复宿主机 `9090`。
- 摄像头和麦克风是独立选择器；可组合为 Mac camera + Mac mic、Mac camera + Sipeed 8ch、RealSense + Sipeed 等，不改变 attention 决策器。
- 统一控制台停止按钮现在会同时停止视觉和声学采集；麦克风设备按 profile 通道数自动过滤。
- 统一控制台角度显示为两位小数；C++ 输出不再使用 10° bin 中心作为最终方位，bin 仅用于竞争。
- 首测目标：Mac 原生摄像头 + 可选 Mac 内置麦或 Sipeed 阵列，真实 ROS2/Docker attention 链路。

## 已完成并验证

### `voice-detection`

- C++ 鸡尾酒前端：VAD、GCC-PHAT DOA、delay-and-sum、轻量 AEC、噪声抑制、声源跟踪。
- Python 设备 profile：Mac/Windows 单通道、Sipeed 8ch 48kHz、Reachy Mini。
- TTS `AudioFrame` 兼容、播放参考时间对齐、后台 JSONL AEC 参考源。
- 流式 ASR endpoint、异步 ASR worker、可插拔 external command/Vosk adapter。
- `record-live`：按 profile 录制原始多通道 WAV；`analyze-wav` 检查通道、采样率、电平、相关矩阵和 DOA。
- ASR 场景模板现为 49 条，包含 `test_stage` 与 `required_capabilities`，不包含 intent/permission。
- 测试：30 个 Python 测试、C++ CTest 通过。

### `vision-detection`

- Mac OpenCV camera source。
- 稳定 transient face track ID，嘴部 ROI、mouth open ratio、lip motion。
- 视觉人物输出 face/body/gesture/gaze/emotion/engagement/proxemics。
- 深度 source：RealSense/通用 ROS depth topic；Mac 无深度时输出低置信 monocular size proxy。
- 预留 INDEMIND M1 ROS topic：left image、depth、IMU；真实 topic 按厂商 SDK 版本 remap。
- ROS C++ 构建已在补齐 `cv_bridge` 的 Humble 容器中通过。

### `robot-attention-perception`

- Python：异步 TTL 融合、ROS4HRI native 算法、PersonManager、AV Binder、BCI common-source inference、心理物理回归。
- C++ ROS2 `attention_stack`：S1 36-bin 方位显著性，S2 时间绑定/fission，S3 WTA/侧抑制，S4 习惯化/工作记忆/Hebbian，S5 FSM。
- Docker ROS2 Humble + `hri_msgs` + rosbridge + 强类型 `perception_interfaces`。
- `test_ros2_roundtrip.py` 已通过：真实 rosbridge -> C++ attention -> attention topic，目标锁定和 `listen` 正确。
- 统一 Mac 控制台：摄像头与麦克风独立选择，支持组合切换，页面显示 ROS2 transport、目标、角度、VAD、Person 概率和延迟。
- 当前控制台地址：`http://127.0.0.1:8092`，仅启动脚本运行时有效。
- 统一控制台停止接口同时停止视觉和声学采集；麦克风列表按 profile 所需通道数过滤。
- attention 最终角度保留原始视觉/声学小数；36-bin 只用于显著性竞争，不代表最终角度分辨率。
- 2026-09-08 本阶段回归完成：voice 31 Python tests + C++ CTest，attention 35 Python tests + C++ CTest，vision 2 Python tests，memory Rust/Python tests均通过；attention Docker image 可构建。
- 已生成 PDF：`/Applications/权重/robot_perception_architecture_and_test_guide.pdf`，9页，包含四体架构图、机制说明、topic清单、设备组合和49条逐项场景。
- 新增命令：`scripts/run_acoustic_test.sh`、`scripts/run_visual_test.sh`、`scripts/run_full_test.sh`、`scripts/check_sipeed.sh`；统一入口保持摄像头/麦克风独立选择。

### `hri-memory-service`

- Rust JSONL 事件存储、Scope、文本/向量/结构化数据、精确检索和 source topic 可追溯。
- 已接入 attention、Person、AV binding、dialogue 状态事件。
- Rust tests 与 Python ingest tests 通过。

## 当前 ROS2 数据链路

```text
Mac camera/mic
  -> vision-detection / voice-detection native HTTP dashboards
  -> rosbridge websocket :9091
  -> perception_interfaces/msg/People + AcousticTracks
  -> C++ attention_stack
  -> /perception/attention/state
  -> /attention/state + /attention/look_at + /attention/target_voice_id
  -> ROS4HRI /humans/* + /tf
```

默认端口：视觉 `8080`、声学 `8090`、统一控制台 `8092`、memory `8788`、本项目 rosbridge 宿主机 `9090`。

## 下一步：Sipeed 到货测试

1. 连接 Sipeed USB，运行 `python -m voice_detection.cli list-devices`，确认设备 `max_input_channels >= 8`。
2. 录制原始 8ch：

   ```bash
   cd ~/Golands/voice-detection
   ~/Golands/robot-attention-perception/.venv-mac/bin/python -m voice_detection.cli record-live \
     --profile sipeed_6_plus_1_usb_array --device <设备编号> --seconds 10 \
     --output data/sipeed_8ch_48k_test.wav
   ```

3. 校验：

   ```bash
   ~/Golands/robot-attention-perception/.venv-mac/bin/python -m voice_detection.cli analyze-wav \
     --profile sipeed_6_plus_1_usb_array --wav data/sipeed_8ch_48k_test.wav
   ```

4. 对阵列做单人固定方向测试，再做 `-60/-30/0/30/60/120` 度测试；记录 DOA 原始小数和真实角度，先不要宣称精度。
5. 通过原始 8ch 检查后，在统一控制台选择 `mac_builtin` 摄像头组合 + `sipeed_6_plus_1_usb_array` 麦克风 profile。

测试完成后用 `scripts/stop_mac_first_test.sh` 停止，避免摄像头或声卡被占用。

当前确认：本项目测试服务已经停止，未占用摄像头、麦克风和 `8080/8090/8092/8788`；Sipeed 尚未进行实际录音。

可排列组合：

```text
Mac camera + Mac built-in mic
Mac camera + Sipeed 6+1
RealSense D405/D435i + Sipeed 6+1
INDEMIND M1 + Sipeed 6+1 (Linux/Jetson/ROS)
```

## 后续相机替换

- RealSense D405/D435i：安装 `realsense-ros` 或 Python SDK，选择对应 source；使用对齐到彩色图的 depth topic。
- INDEMIND M1：在 Linux/Jetson 启动厂商 ROS SDK，映射 `image_topic/depth_topic/imu_topic`；不改变 `/perception/vision/people` 和 attention 输入。
- 替换相机不改变 PersonManager、AV Binder、attention_stack、S2S 和 memory 契约。

## 训练决策

第一版不训练基础模型。先用开源 YOLO/YuNet/MediaPipe、ODAS/HARK 插件、WebRTC AEC3、Asteroid/SpeechBrain、FunASR/SenseVoice/Whisper/Vosk、ECAPA-TDNN、SyncNet/Light-ASD/TalkNet。只有硬件验收出现不可由标定、阈值和热词解决的稳定误差时，才考虑 active-speaker 校准、远场声纹域适配、目标说话人提取或中文 AVSR 微调。
