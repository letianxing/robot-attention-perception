# ROS2 Topic 与接口清单

本文按 ROS4HRI 标准组织四个系统之间的数据边界。ROS4HRI 标准要求 HRI 相关 topic 放在 `/humans/` 下，并把 Person 作为长期身份，把 Face、Body、Voice 作为 transient observation ID；对应 TF frame 建议为 `face_<faceID>`、`body_<bodyID>`、`voice_<voiceID>`。

## 强类型感知主链路

| Topic | 类型 | 发布者 | 消费者 |
| --- | --- | --- | --- |
| `/perception/vision/people` | `perception_interfaces/msg/People` | Mac/Jetson bridge | C++ attention、ROS4HRI ingress |
| `/perception/voice/tracks` | `perception_interfaces/msg/AcousticTracks` | Mac/Jetson bridge | C++ attention、ROS4HRI ingress |
| `/perception/attention/state` | `perception_interfaces/msg/AttentionState` | C++ attention | 控制台、对话、行为、memory |
| `/attention/look_at` | `geometry_msgs/msg/PointStamped` | C++ attention | 头眼控制器 |
| `/attention/target_voice_id` | `std_msgs/msg/String` | C++ attention | S2S target stream router |
| `/attention/person_hypotheses` | `std_msgs/msg/String` JSON | PersonManager | 身份概率表、UI、memory |
| `/attention/active_behaviors` | `std_msgs/msg/String` JSON | PersonManager | 问询等主动行为建议 |

## S1-S5 调试输出

| Topic | 类型 | 含义 |
| --- | --- | --- |
| `/attention/debug/s1_visual_saliency` | `std_msgs/msg/Float32MultiArray` | 36 bin 视觉显著性 |
| `/attention/debug/s1_audio_saliency` | `std_msgs/msg/Float32MultiArray` | 36 bin 声学显著性 |
| `/attention/debug/s2_fission` | `std_msgs/msg/Bool` | 单视觉/双声音事件 fission |
| `/attention/debug/s3_competition` | `std_msgs/msg/Float32MultiArray` | WTA 与侧抑制 |
| `/attention/debug/s4_memory` | `std_msgs/msg/Float32MultiArray` | 习惯化、工作记忆、Hebbian 结果 |
| `/attention/debug/s5_phase` | `std_msgs/msg/String` | IDLE/ORIENTING/ENGAGED/RELEASE |
| `/attention/debug/common_source_probability` | `std_msgs/msg/Float32` | BCI 共同来源后验 |

## vision-detection 输出

| Topic | 类型 | 频率 | 消费方 | 说明 |
| --- | --- | ---: | --- | --- |
| `/vision/people` | `vision_detection/msg/PeopleSignals` | 5-10 Hz | attention、ROS4HRI bridge、memory | 每个人的 `person_id/face_id/body_id/azimuth/gaze/body_facing/proxemic/engagement/gesture/emotion`。这是注意力主输入。 |
| `/vision/signals` | `vision_detection/msg/VisionSignals` | 5-10 Hz | dashboard、attention fallback | 聚合信号和最近人状态，兼容已有视觉功能。 |
| `/vision/annotated_image` | `sensor_msgs/msg/Image` | 10-20 Hz | dashboard | 调试图像，不进入注意力关键路径。 |
| `/vision/available_cameras` | `vision_detection/msg/CameraList` | 0.2 Hz/latch | dashboard | Mac 本地摄像头选择。 |
| `/vision/near_human_present` | `std_msgs/msg/Bool` | 5-10 Hz | fallback | 是否有人靠近。 |
| `/vision/v_user_raw` | `std_msgs/msg/Float32` | 5-10 Hz | emotion consumers | 最近人情绪 valence。 |
| `/vision/face_orient` | `vision_detection/msg/FaceOrientation` | 5-10 Hz | fallback | 最近人 yaw/pitch/roll。 |
| `/vision/gesture_events` | `vision_detection/msg/GestureEvents` | 5-10 Hz | attention fallback | 手势事件。 |

推荐 ROS4HRI bridge 输出：

| Topic | 类型 | 说明 |
| --- | --- | --- |
| `/humans/faces/tracked` | `hri_msgs/IdsList` | 当前 face ID 列表。 |
| `/humans/bodies/tracked` | `hri_msgs/IdsList` | 当前 body ID 列表。 |
| `/humans/persons/tracked` | `hri_msgs/IdsList` | 当前 person ID 列表。 |
| `/humans/candidate_matches` | `hri_msgs/IdsMatch` | face/body/voice/person 候选关联。 |
| `/humans/persons/<personID>/face_id` | `std_msgs/String` | person 到 face observation 的当前关联。 |
| `/humans/persons/<personID>/body_id` | `std_msgs/String` | person 到 body observation 的当前关联。 |
| `/humans/persons/<personID>/engagement_status` | `hri_msgs/EngagementLevel` | ROS4HRI 标准 engagement level。 |
| `/tf` | `tf2_msgs/TFMessage` | `face_<faceID>`、`gaze_<faceID>`、`body_<bodyID>`。 |

## voice-detection 输出

| Topic | 类型 | 频率 | 消费方 | 说明 |
| --- | --- | ---: | --- | --- |
| `/humans/voices/tracked` | `hri_msgs/IdsList` | 20-50 Hz | ROS4HRI person manager、attention | 当前 voice ID 列表。 |
| `/humans/voices/<voiceID>/audio` | audio frame | 20-50 Hz | attention/S2S gate | 单路增强音频或目标分离流。 |
| `/humans/voices/<voiceID>/features` | `hri_msgs/AudioFeatures` | 20-50 Hz | attention、debug | RMS、ZCR、pitch、HNR、MFCC。 |
| `/humans/voices/<voiceID>/is_speaking` | `std_msgs/Bool` | 20-50 Hz | attention | VAD 结果。 |
| `/humans/voices/<voiceID>/speech` | `hri_msgs/LiveSpeech` | partial/final | ASR 测试、memory | 增量/最终转写，只记录文字、语言、置信度，不做意图判断。 |
| `/voice/acoustic_tracks` | JSON or custom msg | 20-50 Hz | attention、memory | 工程扩展字段：DOA、clarity、overlap、self_echo、speaker_label。 |
| `/voice/asr_records` | JSONL | final | ASR 测试、memory | 你的验收场景记录字段。 |

部署增强输出：

| Topic/流 | 说明 |
| --- | --- |
| `raw_8ch_pcm -> 4090` | Mac/Jetson 从 Sipeed UAC2 采集 8ch 48 kHz S16_LE，推送到 4090 远端算法服务。 |
| `frontend_output <- 4090` | 4090 回传增强音频、`AcousticTrack` 和可选 ASR。当前最小协议为 JSONL-over-TCP。 |
| `/tf` | `voice_<voiceID>` 声源方位 frame，生产 ROS2 集成时补。 |

## robot-attention-perception 消费

| 输入 | 类型 | 频率 | 处理方式 |
| --- | --- | ---: | --- |
| `/humans/voices/tracked` 与 voice 子 topic | ROS4HRI | 20-50 Hz | 进入音频 TTL 缓存，默认 180 ms。 |
| `/voice/acoustic_tracks` | JSON/custom | 20-50 Hz | 包含注意力需要的非标准工程字段。 |
| `/humans/faces/tracked`、`/humans/bodies/tracked`、`/humans/persons/tracked` | ROS4HRI | 5-10 Hz | 进入视觉/person TTL 缓存，默认 450 ms。 |
| `/humans/candidate_matches` | `hri_msgs/IdsMatch` | event | 优先使用 voice/person 关联。 |
| `/vision/people` | `vision_detection/msg/PeopleSignals` | 5-10 Hz | 没有完整 ROS4HRI person manager 时的直接输入。 |
| `/robot/playback_state` | bool/RMS/custom | 20-50 Hz | AEC/self-echo 与 barge-in 处理。 |
| `/robot/motion_state` | bool/custom | 20-50 Hz | 机器人运动噪声降权。 |

## robot-attention-perception 输出

| Topic | 类型 | 频率 | 消费方 | 说明 |
| --- | --- | ---: | --- | --- |
| `/attention/state` | `robot_attention_perception/AttentionState` | <=50 Hz | dashboard、memory、decision | 当前目标、置信度、listen、addressed、理由。 |
| `/attention/look_at` | `geometry_msgs/PointStamped` or azimuth/elevation msg | <=50 Hz | 头部/眼睛/运动控制 | 先用声学快速看向，视觉更新后修正。 |
| `/attention/target_voice_id` | `std_msgs/String` | event/50 Hz | S2S gate | 被放行的 voice ID。 |
| `/speech/target_stream` | audio frame | 20-50 Hz | speech-to-speech | 注意力门控后的目标音频；S2S 不直接吃全场混音。 |
| `/memory/events` 或 HTTP `/v1/events` | JSON/custom | event | hri-memory-service | 可追溯 attention decision。 |

当前可运行桥接脚本：

- `vision-detection/scripts/ros4hri_vision_bridge.py`：订阅 `/vision/people`，发布 ROS4HRI face/body/person tracked、candidate matches、person 子 topic，并额外发布 `/vision/people_json` 给轻量注意力节点。
- `voice-detection/scripts/ros4hri_voice_bridge.py`：从 JSONL 读取声学轨迹，发布 `/humans/voices/...` 和 `/voice/acoustic_tracks`。
- `robot-attention-perception/scripts/ros4hri_attention_node.py`：订阅 `/voice/acoustic_tracks`、`/vision/people_json` 或 `/vision/people`、`/robot/playback_state`，发布 `/attention/state`、`/attention/target_voice_id`、`/attention/look_at`。

## hri-memory-service 接口

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `POST /v1/events` | HTTP JSON | 写入 `face_observation/body_observation/voice_observation/asr_transcript/attention_state/dialogue_turn`。 |
| `GET /v1/events` | HTTP JSON | 按 Scope 查看最近事件。 |
| `POST /v1/search` | HTTP JSON | 文本、向量、kind、entity 过滤检索。 |
| `GET /v1/health` | HTTP JSON | 存储文件和事件数。 |

## 频率不一致处理

注意力系统不等待完全同 timestamp 的视觉和声学消息。`AsyncPerceptionBuffer` 使用 mixed-rate cache：

- 声学 TTL 默认 `180 ms`，过期直接丢弃。
- 视觉/person TTL 默认 `450 ms`，旧视觉逐步降权。
- robot playback/motion TTL 默认 `250 ms`。
- 输出最多 `50 Hz`，避免消息风暴。

这样能保证机器人先在 100 ms 级响应声音方向，再由视觉/person manager 在下一帧确认身份、凝视和是否对话。
