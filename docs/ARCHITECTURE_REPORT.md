# 仿生视听感知系统：架构、数据流与手工测试指南

生成日期：2026-09-11

## 0. 可直接执行的测试入口

以下命令以 macOS + Docker Desktop + ROS 2 Jazzy + Sipeed 6+1 为准。

```bash
cd ~/Golands/robot-attention-perception
ROS_DISTRO=jazzy ROS_BASE_IMAGE=ros:jazzy-ros-base \
  docker compose -f docker-compose.mac.yml up -d --build
scripts/setup_mac_first_test.sh
OPEN_BROWSER=0 scripts/run_mac_first_test.sh
```

控制台地址：`http://127.0.0.1:8092`；视觉：`http://127.0.0.1:8080`；语音：`http://127.0.0.1:8090`；
记忆：`http://127.0.0.1:8788/v1/health`；ROS bridge：`ws://127.0.0.1:9090`。

在控制台选择 `profile=sipeed_6_plus_1_usb_array` 和 `device=MicArray`（本机通常为编号 1），点击“启动感知”。

如果 8092 的“声学”仍为红色，先看：

```bash
curl -fsS http://127.0.0.1:8090/api/state
```

必须满足 `running=true` 且 `error=""`。若出现 `PortAudio -9986`，先执行
`scripts/stop_mac_first_test.sh`，确认没有其他程序占用麦克风，再重新运行
`OPEN_BROWSER=0 scripts/run_mac_first_test.sh`；本机 Sipeed 枚举应为 `MicArray`、8 输入通道、
设备编号通常为 `0`。可用以下命令独立确认句柄：

```bash
cd ~/Golands/voice-detection
~/Golands/robot-attention-perception/.venv-mac/bin/python -m voice_detection.cli list-devices
```

### 0.1 硬件 smoke test

```bash
cd ~/Golands/voice-detection
~/Golands/robot-attention-perception/.venv-mac/bin/python -m voice_detection.cli list-devices
mkdir -p data
~/Golands/robot-attention-perception/.venv-mac/bin/python -m voice_detection.cli record-live \
  --profile sipeed_6_plus_1_usb_array --device 1 --seconds 3 --output data/sipeed_smoke.wav
~/Golands/robot-attention-perception/.venv-mac/bin/python -m voice_detection.cli analyze-wav \
  --profile sipeed_6_plus_1_usb_array --wav data/sipeed_smoke.wav
```

通过条件：`channel_count=8`、`sample_rate_hz=48000`，所有通道 `silent=false` 且 `clipping=false`。

### 0.2 ROS4HRI 与注意力验证

```bash
cd ~/Golands/robot-attention-perception
scripts/check_ros4hri_topics.sh || true
docker exec robot-attention-ros2 bash -lc \
  'source /opt/ros/jazzy/setup.bash; ros2 node list'
docker exec robot-attention-ros2 bash -lc \
  'source /opt/ros/jazzy/setup.bash; ros2 topic echo /humans/voices/tracked'
docker exec robot-attention-ros2 bash -lc \
  'source /opt/ros/jazzy/setup.bash; ros2 topic echo /attention/state'
```

说话时应看到 `voice_id/track_id`、DOA、`target_id`、`confidence`、`listen` 和 `reasons` 更新；
机器人未被寻址时 `listen=false` 是正常门控结果。

## 0.3 停止与重新启动

```bash
curl -fsS http://127.0.0.1:8090/api/stop || true
docker compose -f docker-compose.mac.yml down
```

重新测试时再次执行 0 节命令。长时场景中途重启 ASR 后，必须确认 `/api/state` 的 `running=true` 且新 transcript 的
`is_final=true`。

## 0.4 49 个场景的统一操作步骤

场景文件为 `~/Golands/voice-detection/data/asr_acceptance_scenarios.jsonl`，共 49 条。每一条都按以下步骤：

1. 记录场景 ID、日期、操作者、设备、距离、方向和噪声条件。
2. 确认 `/api/state` 的 `running=true`，清空上一条 transcript。
3. 严格按场景原句说话；需要重复时保持重复次数一致。
4. 等待 `last_transcript.is_final=true`，再记录最终文字。
5. 记录 `voice_id`、`azimuth_deg`、`clarity`、`overlap_probability`、`self_echo_probability`。
   同时记录 `voice.last_streaming_transcript.text`、`first_result_latency_ms` 和 `final_latency_ms`，
   用来判断 ASR 是否真正流式输出，以及最终端点等待是否过长。
6. 同时记录 `/attention/state` 的 `target_id`、`listen`、`addressed_to_robot` 和 `confidence`。
7. 对照场景重点判断专名、数字、否定、顺序、多人目标和机器人播放抑制是否正确。
8. 每完成 10 条保存一次结果；完成后导出 memory 事件。

单条记录可用：

```bash
curl -fsS http://127.0.0.1:8090/api/state > /tmp/voice_state.json
curl -fsS http://127.0.0.1:8092/api/state > /tmp/attention_state.json
curl -fsS http://127.0.0.1:8788/v1/health
```

### 49 条场景分组

| 编号 | 分组 | 执行方法 | 重点判定 |
| --- | --- | --- | --- |
| 1–14 | 语言内容 | 安静、正面、1.5m，逐句执行 | 自然句、专名、数字、否定、转述、条件和改口 |
| 15–20 | 说话方式 | 保持位置，改变语速、音量、口音和非语言声音 | VAD 截断、口头语、笑声/咳嗽误识别 |
| 21–27 | 距离方向遮挡 | 0.5/1.5/3/6m，侧身、背向、左右和局部遮挡 | DOA、视觉 track、person/voice 匹配 |
| 28–34 | 噪声混响 | 空调、电视、音乐、风雨、突发噪声、走廊/玻璃门 | clarity、VAD、overlap、ASR 降级 |
| 35–38 | 机器人状态 | TTS 播放、barge-in、相近内容、机器人运动 | self-echo、playback 状态和用户语音放行 |
| 39–44 | 多人鸡尾酒 | 2–3 人依次/重叠/换位说话，每条至少 5 次 | voice_id 串位、目标切换时延和注意力目标 |
| 45–49 | 长时恢复 | 长静默、连续多轮、长时运行、重启、网络中断恢复 | 进程稳定性、队列、ID 重建和 memory 补写 |

### 通过标准

- ASR：否定、数字、专名、时间关系不能改变语义。
- 声学：方向基本正确；重叠时有明确 overlap；机器人播放不应被识别为用户。
- 注意力：目标选择合理，信号过期时不应继续 `listen=true`；视觉不可用时允许声学降级。
- ROS4HRI：`/humans/*`、`/tf`、`/attention/*` 持续发布且 ID 关系可追溯。

## ROS4HRI/Jazzy 使用说明（当前版本）

本指南的对外接口以 ROS4HRI REP-155 为准：Person 是长期身份，Face/Body/Voice 是临时观测，所有标准
HRI 数据位于 `/humans/` 命名空间。四个仓库保留原有算法和测试方式，但通过 ingress 和 `pyhri.HRIListener`
接入标准接口。ROS 2 Jazzy 是首选发行版，Humble 作为兼容目标。

### 标准安装

```bash
sudo apt update
sudo apt install ros-jazzy-hri-msgs ros-jazzy-hri-actions-msgs \
  ros-jazzy-hri ros-jazzy-pyhri ros-jazzy-human-description ros-jazzy-hri-rviz
```

另外从 ROS4HRI 源码安装 `hri_face_detect`、`hri_emotion_recognizer`、`hri_face_identification`、
`hri_body_detect`、`hri_person_manager`、`hri_engagement`、`hri_visualization` 和 `rqt_human_radar`。

Docker 也支持 Jazzy：`ROS_DISTRO=jazzy ROS_BASE_IMAGE=ros:jazzy-ros-base docker compose -f docker-compose.mac.yml up -d --build`。
自研 `vision-detection` 可以替代 detector，但必须继续通过 `/vision/people` + bridge 输出 REP-155 数据。

### 一眼可执行的启动/验证命令

```bash
cd ~/Golands/robot-attention-perception
colcon build --symlink-install
source install/setup.bash
ros2 launch attention_stack ros4hri_attention.launch.py \
  use_person_manager:=true use_pyhri_adapter:=true
```

若采用 ROS4HRI 官方视觉节点而不是自研视觉 backend：

```bash
ros2 launch attention_stack ros4hri_attention.launch.py \
  use_official_detectors:=true use_person_manager:=true use_pyhri_adapter:=true
```

另一个终端启动本地视觉、声学后端（保留原有命令）：

```bash
scripts/run_vision_dashboard.sh
scripts/run_voice_dashboard.sh
```

检查标准输出和注意力：

```bash
ros2 topic list | grep '^/humans/'
ros2 topic echo /humans/persons/tracked
ros2 topic echo /humans/voices/tracked
ros2 topic echo /attention/state
ros2 topic echo /attention/look_at
ros2 run rqt_human_radar rqt_human_radar
```

官方 `hri_person_manager` 负责 `/humans/candidate_matches` 的概率人员融合；`pyhri_people_adapter.py`
使用 `hri.HRIListener` 读取标准人类对象并转换到 attention 的强类型消息。注意力算法、声学处理、视觉模型、
ASR 场景测试和 memory HTTP API 均保持原有实现。

## 结论

本系统由四个可以分别启动、分别替换、通过 ROS2 契约互通的仓库组成：`vision-detection`、`voice-detection`、`robot-attention-perception` 和 `hri-memory-service`。Mac 第一次测试使用 Mac 摄像头与用户选择的麦克风；Sipeed 6+1 到位后只切换麦克风 profile；RealSense D405/D435i 或 INDEMIND M1 到位后只切换视觉 source。

当前身份链路采用可落地的预训练/特征编码器边界：视觉使用 OpenCV SFace 产生人脸向量；人脸向量注册写入本地缓存并同步到 `hri-memory-service` 的 `identity-registry` 作用域。检索使用余弦相似度，在线注意力只消费已经确认的 Person 身份，不把单帧低置信度匹配当成权限结论。声学说话人向量模型仍需单独接入，不能把当前临时 `voice_id` 当成跨会话声纹身份。

## 一、四体总架构图

```text
┌──────────────────────┐       ┌──────────────────────┐
│ Mac / RealSense / M1 │       │ Mac / Sipeed / Reachy│
│ RGB、深度、IMU        │       │ 原始音频、播放参考     │
└──────────┬───────────┘       └──────────┬───────────┘
           │                               │
           v                               v
┌──────────────────────┐       ┌──────────────────────┐
│ vision-detection     │       │ voice-detection      │
│ face/body/gaze/depth │       │ DOA/beam/AEC/VAD/ASR │
└──────────┬───────────┘       └──────────┬───────────┘
           │ People                         │ AcousticTracks
           └──────────────┬────────────────┘
                          v
                 ┌─────────────────────┐
                 │ ROS2 + rosbridge    │
                 │ typed interfaces    │
                 └──────────┬──────────┘
                            v
                 ┌─────────────────────┐
                 │ attention_stack C++ │
                 │ S1-S5 + BCI + WTA   │
                 │ PersonManager       │
                 └──────┬───────┬──────┘
                        │       │
             look_at/ID │       │ gated target audio
                        v       v
                 ┌──────────┐  ┌──────────────┐
                 │ head/eye │  │ S2S / TTS    │
                 │ control  │  │ playback AEC │
                 └──────────┘  └──────┬───────┘
                                      │
                                      v
                            ┌──────────────────┐
                            │ hri-memory-service│
                            │ Rust event/trace │
                            └──────────────────┘
```

Mac 原生进程负责访问 macOS 摄像头和 CoreAudio/PortAudio；Docker Desktop 运行 ROS2 Humble、rosbridge、ROS4HRI ingress 和 C++ `attention_stack`。这样既是真实 ROS2 通信，又不要求 Docker 直接访问 Mac 摄像头和麦克风。

## 二、四个仓库的职责

| 仓库 | 负责什么 | 不负责什么 |
| --- | --- | --- |
| `vision-detection` | RGB/RGB-D 采集；人、脸、身体、手势、gaze、嘴部运动、情绪、深度、稳定 transient ID。 | 不决定是否应该回应，不处理权限和意图。 |
| `voice-detection` | Mac/Windows/Sipeed/Reachy 设备抽象；8ch DOA、波束、AEC、降噪、VAD、重叠提示、声源跟踪、ASR、播放参考。 | 不决定身份权限、意图和最终回应。 |
| `robot-attention-perception` | 异步多率缓存、Face/Body/Voice 概率绑定、匿名/长期 Person、BCI 共同来源、S1-S5 attention、S2S 门控。 | 不把 ASR 文字解释成动作意图。 |
| `hri-memory-service` | Rust 本地事件、文本、向量、结构化数据、媒体 URI、source topic、trace 查询。 | 不直接改变在线注意力决策。 |

身份向量操作：

```bash
# 在摄像头前保持正脸，先注册视觉身份（8092 的“注册并绑定”会同时执行）
curl -X POST http://127.0.0.1:8080/api/enroll \
  -H 'content-type: application/json' \
  -d '{"user_id":"owner","user_role":"owner"}'

# 查询身份向量库（跨测试 session，固定使用 identity-registry）
curl -X POST http://127.0.0.1:8788/v1/search \
  -H 'content-type: application/json' \
  -d '{"scope":{"subject_id":"local-user","robot_id":"reachy-mini","session_id":"identity-registry"},"query_embedding":[...],"kind":"face_identity_profile","limit":5}'
```

## 三、声学链路

```text
原始 PCM
  -> DOA/GCC-PHAT + 6通道空间波束
  -> AEC / 稳态降噪 / 可选神经增强
  -> VAD / overlap / source tracker
  -> 分离目标音频 / ASR
  -> attention gate
  -> S2S/TTS
  -> playback reference 回到 AEC
```

Sipeed 6+1 采用原始 CH0-CH5 做空间处理，CH6 不直接作为主输入，CH7 作为设备参考候选，具体角色由实测确认。原始阵列上不能先做会破坏相位的降噪。Mac/Jetson 获取原始8ch，4090 可运行 ODAS/HARK/Asteroid/SpeechBrain/FunASR 等重模块，回传 `AcousticTracks` 和增强音频。

机器人播放的 `tts_service/msg/AudioFrame` 经过播放参考缓冲按时间戳对齐。机器人播放状态通过 `/robot/playback_state` 进入 attention；self-echo 高时拒绝把机器人声音作为注意目标，防止自说自听。

## 四、视觉和深度链路

Mac 摄像头输出人物方位和低置信的 `monocular_size_proxy` 距离，只用于首测。RealSense 使用彩色图和对齐深度；D405 适合近距离约7cm-50cm，不适合替代远距离深度相机。D435i 用于更广泛的 RGB-D/IMU 场景。INDEMIND M1 使用 Linux/Jetson/ROS 厂商 SDK 输出双目图像、深度和 IMU，再映射到统一 `People` 接口。

设备组合完全独立：

```text
Mac camera       + Mac built-in microphone
Mac camera       + Sipeed 6+1
RealSense D405   + Sipeed 6+1
RealSense D435i  + Sipeed 6+1
INDEMIND M1      + Sipeed 6+1
```

## 五、视听融合机制

### S1：方位显著性

视觉和声学统一到 `36-bin × 10°` 的 egocentric 方位网格，供竞争和记忆使用。最终 `/attention/look_at` 和 attention JSON 保留原始视觉/声学加权小数角度，界面显示两位小数；10° bin 不代表物理定位精度。

### S2：时间绑定和 fission

默认绑定窗口为 `±100ms`。以视觉/声学 timestamp 为整数毫秒，计算时间一致性和共同来源概率。一个视觉人物对应两个突发声音时激活 fission，避免把两个声音错误合成一个来源。

### S3：WTA、侧抑制和唇音一致性

声学和视觉先保留独立证据，再用 WTA、邻近侧抑制、overlap/self-echo 惩罚竞争。嘴部运动和 active-speaker 只提高共同来源概率，不覆盖清晰的声学证据。麦格克式一致/冲突只能作为重打分和 ASR 辅助，不能让错误唇形强行改变文字。

### S4：习惯化、工作记忆和 Hebbian 绑定

默认习惯化时间常数约5秒，工作记忆约8秒，维护视觉方位与声学方位的 Hebbian 关联。PersonManager 另维护5秒离场重识别状态。一次 Face/Body/Voice 跳变只改变概率表，不直接改变机器人决策目标。

### S5：状态机

```text
IDLE -> ORIENTING -> ENGAGED -> RELEASE
```

默认300ms防抖。声音可先触发定向，视觉在后续帧确认 gaze、body facing、嘴部运动和 Person。长期不稳定的身份关系触发 `clarify_person_identity` 主动行为建议，但不直接冒充权限或意图结论。

## 六、ROS2 Topic 清单

### 强类型工程主链路

| Topic | 类型 | 发布者 | 消费者 |
| --- | --- | --- | --- |
| `/perception/vision/people` | `perception_interfaces/msg/People` | Mac/Jetson视觉桥 | `attention_stack`、ROS4HRI ingress |
| `/perception/voice/tracks` | `perception_interfaces/msg/AcousticTracks` | Mac/Jetson声学桥 | `attention_stack`、ROS4HRI ingress |
| `/perception/attention/state` | `perception_interfaces/msg/AttentionState` | C++ attention | S2S、运动、memory、UI |
| `/attention/look_at` | `geometry_msgs/msg/PointStamped` | C++ attention | 头眼控制 |
| `/attention/target_voice_id` | `std_msgs/msg/String` | C++ attention | S2S 目标流路由 |
| `/attention/person_hypotheses` | `std_msgs/msg/String` JSON | PersonManager | UI、memory、身份系统 |
| `/attention/active_behaviors` | `std_msgs/msg/String` JSON | PersonManager | 问询等主动行为系统 |

### ROS4HRI 标准兼容输出

| Topic | 类型 | 说明 |
| --- | --- | --- |
| `/humans/faces/tracked` | `hri_msgs/IdsList` | 当前 Face transient IDs |
| `/humans/bodies/tracked` | `hri_msgs/IdsList` | 当前 Body transient IDs |
| `/humans/voices/tracked` | `hri_msgs/IdsList` | 当前 Voice transient IDs |
| `/humans/persons/tracked` | `hri_msgs/IdsList` | 当前 Person IDs |
| `/humans/persons/<id>/face_id` | `std_msgs/String` | Person 到 Face 关联 |
| `/humans/persons/<id>/body_id` | `std_msgs/String` | Person 到 Body 关联 |
| `/humans/persons/<id>/voice_id` | `std_msgs/String` | Person 到 Voice 关联 |
| `/humans/persons/<id>/engagement_status` | `hri_msgs/EngagementLevel` | engagement 状态 |
| `/humans/voices/<id>/is_speaking` | `std_msgs/Bool` | VAD 状态 |
| `/humans/voices/<id>/speech` | `hri_msgs/LiveSpeech` | 增量/最终文字 |
| `/tf` | `tf2_msgs/TFMessage` | `face_*`、`gaze_*`、`body_*`、`voice_*` |

### S1-S5 调试输出

| Topic | 含义 |
| --- | --- |
| `/attention/debug/s1_visual_saliency` | 36-bin视觉显著性 |
| `/attention/debug/s1_audio_saliency` | 36-bin声学显著性 |
| `/attention/debug/s2_fission` | fission是否激活 |
| `/attention/debug/s3_competition` | WTA/侧抑制结果 |
| `/attention/debug/s4_memory` | 习惯化/工作记忆/Hebbian结果 |
| `/attention/debug/s5_phase` | FSM状态 |
| `/attention/debug/common_source_probability` | 贝叶斯共同来源后验 |

## 七、异步频率和延迟

- 声学20-50Hz，TTL 180ms，过期快速丢弃。
- 视觉5-10Hz，TTL 450ms，过期逐步降低 gaze/facing/confidence。
- 机器人播放/运动20-50Hz，TTL 250ms。
- attention最多50Hz，不等待视觉和声学同一 timestamp。
- Mac实测当前融合周期约1-4ms；真实总反应延迟还包括采集buffer、网络、模型和电机。
- 4090链路必须使用固定小块、后台队列、共享内存或二进制协议替代高频 JSON 序列化。

## 八、开源模型策略

第一版不训练：YOLOv8n/YuNet/MediaPipe、ODAS/HARK、WebRTC AEC3、Asteroid/SpeechBrain、Vosk/Whisper/FunASR/SenseVoice、ECAPA-TDNN、SyncNet/Light-ASD/TalkNet 都保留为插件或下游模型。SoundSpaces只用于离线 RIR/场景仿真；AV-HuBERT只作为离线教师候选。

只有出现稳定且不可由阵列标定、阈值、热词、时间窗和概率校准解决的错误，才考虑按以下顺序训练：active-speaker 概率校准、远场 ECAPA 域适配、目标说话人提取、中文 AVSR。不要从零训练基础视觉、ASR 或分离模型。

## 九、49场景手工测试

完整逐项手册：`~/Golands/voice-detection/docs/ASR_MANUAL_TEST_GUIDE.md`。

统一记录：原话、场所、距离、相对方向、背景声、遮挡、机器人状态、语速、dB(A)、口音、语言、笑声/咳嗽/轻声/喊话、最终转写、语言/清晰度、句首句尾、数字/名称/否定词、背景混入、多人拼接、开始/结束/结果时间。

| 阶段 | 场景 |
| --- | --- |
| `mac_first_test` | 基础语句、短词、长句、名称、数字、中英文、否定、多重否定、转述、时间、条件、多项信息、改口、语速、音量、口音、咳嗽、停顿、吃东西、安静首句、连续多轮、基础噪声 |
| `spatial_hardware` | 0.5/1.5/3/6m、前后左右、侧向背向、遮挡、空调、电视、音乐、户外噪声、强噪声、多人重叠和移动换位 |
| `robot_hardware` | 机器人播放、插话、相近内容、关节/行走/风扇噪声 |
| `networked_asr` | 4090 网络正常、变慢、断开、恢复和 ASR 重启 |

### 49 条逐项清单

<!-- SCENARIO_TABLE -->

ASR场景文件：`~/Golands/voice-detection/data/asr_acceptance_scenarios.jsonl`，共49条。它不包含 intent、permission、身份或动作结论。

### 9.1 Sipeed 6+1 现场摆位与逐条执行卡

这 49 条不是“在面板里点一下”的测试，而是可重复的物理验收。每次只改变一个变量，并在句前、句后各留 2 秒静音。

| 项目 | 固定要求 |
| --- | --- |
| 麦克风 | Sipeed 6+1 环形阵列水平放置，阵列中心朝向摄像头；不要遮挡阵列孔 |
| 摄像头 | Mac 原生摄像头置于眼睛高度，人物脸部位于画面中线 |
| 单人基线 | 人坐在镜头正前方，眼睛到镜头 1.5 m，嘴到阵列约 1.0–1.5 m |
| 距离组 | 0.5 m、1.5 m、3 m；6 m 仅做“能否触发”极限记录 |
| 方向组 | 正前 0°、左/右约 45°、侧面 ±90°；以阵列正前为 0° |
| 音量组 | 正常 60–65 dB(A)@1m；轻声 45–50 dB(A)；喊话 70–75 dB(A)，禁止削波 |
| 环境 | 基线 35–45 dB(A)；噪声场景记录空调/电视/音乐的实际位置和音量 |
| 说话方式 | 一条场景一句话；不要自行改写原句；句首和句尾说完整 |

每条场景按以下顺序执行：

1. 在 `~/Golands/voice-detection/data/asr_acceptance_scenarios.jsonl` 找到对应 `scenario_id`，按本节摆位。
2. 确认 8092 显示 `声学=绿`、`视觉=绿`，并用 `curl -fsS http://127.0.0.1:8092/api/state` 保存起始快照。
3. 测试者按规定音量说原句，必要时按场景要求让第二人/机器人同时发声。
4. 等待 1 秒，保存 `curl -fsS http://127.0.0.1:8090/api/state` 和 8092 快照。
5. 记录 `last_transcript.text/is_final`、`last_track.track_id/speaker_label/speaker_similarity/azimuth_deg/clarity/overlap_probability/self_echo_probability`、`last_vad.active/probability/rms_dbfs/noise_floor_dbfs/snr_db`，以及 `attention.target_id/listen/addressed_to_robot/confidence`。
6. 同一场景至少重复 3 次；三次中 2 次达到通过条件才记为通过。把失败归因到 VAD、前端、ASR 或融合层，不要只记“识别错”。

#### 弱声、沙哑和远距离专门判定

- 原始波形有变化但 `last_vad.active=false`：先判为 VAD/AGC/降噪问题。若 `rms_dbfs` 上升而 `snr_db < 6`，继续降低噪声底估计速度或检查阵列通道增益；不要先换 ASR。
- `last_vad.active=true`、`speech_probability` 稳定但最终文字为空/错：判为 ASR 输入或模型域问题。确认送入 ASR 的文件为单声道、16-bit、16 kHz，并检查句首是否因预缓存不足丢失。
- 轻声和感冒沙哑测试必须使用 `VOICE_PRE_ROLL_MS=320`、`VOICE_ENDPOINT_SILENCE_MS=600` 的默认值；若仍漏检，逐步把 `VOICE_MIN_SPEECH_MS` 调到 40，不要把 VAD 阈值一次降到 0。
- 机器人播放场景必须同时记录 `self_echo_probability` 和播放参考是否接入；只有麦克风波形而无 playback reference 时，不能宣称 AEC 已验证。

#### 现场调参顺序（硬件不变）

先执行“静音 10 秒 + 正常语音 3 次”，再按顺序只改一个参数：`VAD threshold 6→5 dB`；
`min_rms -58→-62 dBFS`；`AGC max gain +12→+18 dB`；确认降噪不超过 0.35；
最后才检查 ASR 重采样和模型。每次改动都要回跑场景 1、15、21、35、45，避免只优化单个距离。

## 十、启动命令

### 声学单独

```bash
cd ~/Golands/robot-attention-perception
scripts/run_acoustic_test.sh
```

面板：`http://127.0.0.1:8090`。选择 `mac_builtin` + Mac麦克风，或 `sipeed_6_plus_1_usb_array` + 至少8输入通道的 Sipeed 设备。

### 视觉单独

```bash
cd ~/Golands/robot-attention-perception
scripts/run_visual_test.sh
```

面板：`http://127.0.0.1:8080`。默认 Mac camera。后续 source 可切 RealSense；M1建议在 Linux/Jetson ROS topic 上运行。

### 四体完整 ROS2

```bash
cd ~/Golands/robot-attention-perception
scripts/run_full_test.sh
```

统一面板：`http://127.0.0.1:8092`。在面板中独立选择摄像头和麦克风，然后点击启动。ROS2容器使用 `ws://127.0.0.1:9090`，memory 使用 `http://127.0.0.1:8788`。

```bash
scripts/stop_mac_first_test.sh
```

### Sipeed 原始8ch预检

```bash
cd ~/Golands/robot-attention-perception
scripts/check_sipeed.sh <设备编号> 10
```

期望 `channel_count=8`、`sample_rate_hz=48000`。拍手和从已知方向说话后，再开始完整组合测试。

## 十一、当前限制

软件自动化和 ROS2 roundtrip 已通过，但不能代替物理验收。DOA绝对角度、多人完全重叠、远距离、遮挡、机器人运动噪声、真实扬声器 AEC 和所有 ASR 文字正确性必须按49条记录实际测试。每次阶段完成后更新 `docs/PROGRESS.md`，以便切换 session 后继续。

参考：ROS4HRI standard / REP-155；HARK；ODAS ROS2；RealSense ROS；INDEMIND M1 SDK；SpeechBrain；AV-HuBERT。
