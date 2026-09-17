# ROS4HRI 视听感知与注意力骨架

本项目采用 ROS4HRI REP-155 作为对外语义契约，保留四个仓库已有的检测、声学和注意力算法。内部工程 topic
(`/vision/*`、`/voice/*`) 通过 ingress 转换为标准 `/humans/*` topic；这样 detector 可以替换而 attention
消费者不需要修改。

## 0. Jazzy 标准启动路径

ROS4HRI 官方当前以 ROS 2 Jazzy 为主（Humble 可用但覆盖较少）。目标环境建议使用 Jazzy，并安装：

```bash
sudo apt update
sudo apt install ros-jazzy-hri-msgs ros-jazzy-hri-actions-msgs \
  ros-jazzy-hri ros-jazzy-pyhri ros-jazzy-human-description
```

官方感知/融合包从源码安装：`hri_face_detect`、`hri_emotion_recognizer`、`hri_face_identification`、
`hri_body_detect`、`hri_person_manager`、`hri_engagement`。本仓库的 detector 仍可替代前四者，但必须输出
等价的 REP-155 topics/TF；`hri_person_manager` 默认作为标准人员融合层。

启动顺序：

```bash
ros2 launch attention_stack ros4hri_attention.launch.py \
  use_person_manager:=true use_pyhri_adapter:=true
```

需要官方 detector 时将 `use_official_detectors:=true`；该选项会加载 `hri_face_detect`、
`hri_emotion_recognizer` 和 `hri_body_detect`，自研 `vision-detection` 可不启动。

随后启动视觉/声学后端和 ingress。验证：

```bash
ros2 topic list | grep '^/humans/'
ros2 topic echo /humans/persons/tracked
ros2 topic echo /attention/state
ros2 run rqt_human_radar rqt_human_radar
```

`pyhri_people_adapter.py` 通过官方 `hri.HRIListener` 访问 `/humans/*`，只负责适配到现有 attention
强类型输入；不会复制或替换 S1-S5 算法。

## 1. Reachy Mini 的语音形态

公开资料里 Reachy Mini 可以走实时语音对话链路。Hugging Face 的 `speech-to-speech` 服务是兼容 OpenAI Realtime 风格的级联方案，内部通常按 VAD、STT、LLM、TTS 组织；Reachy Mini 自身的 media stack 则负责设备音频输入输出、AEC 和 DOA 等底层能力。

所以可以把 Reachy Mini 的“对话系统”理解成 speech-to-speech/voice-to-voice 后端，而不是传统只做 ASR 的模块。做机器人鸡尾酒效应时，不要直接把全场混音塞给 speech-to-speech；应该先由声学前端选出目标声源或分离流，再送给 ASR/S2S。

## 2. 分层

```text
Audio Device Layer
  Mac CoreAudio / Windows WASAPI / USB UAC2 / Reachy Mini audio / robot arrays
        |
        v
Audio Front-End
  capture -> channel mapping -> phase-safe DC removal -> DOA/SSL -> beamforming
        |                                                        |
        |                                                        v
        |                              AEC/self-echo -> NS/dereverb -> VAD/source tracking
        v
Speech Layer
  ASR transcript records -> optional language + clarity
  optional diarization, speaker embedding, neural separation when overlap is detected
        |
        v
Attention Fusion
  audio source tracks + vision people + robot playback/motion state
        |
        +-> /attention/state       看向哪里、听哪个声源、置信度、理由
        +-> /attention/look_at     给头部/眼睛控制的方向
        +-> /speech/target_stream  给 speech-to-speech 的目标语音流
        +-> /asr/transcripts       给语音识别测试记录的转写结果

Vision Layer
  vision-detection -> /vision/people -> ROS4HRI face/body/person IDs + TF
```

## 3. 声学层是否足够

豆包截图里的声学模块方向是对的，但需要拆清边界：

- `PortAudio / sounddevice / miniaudio`：适合做跨平台采集。Mac 走 CoreAudio，Windows 走 WASAPI，USB UAC 阵列也能统一接入。它们解决“怎么拿到音频”，不解决鸡尾酒效应。
- `ODAS`：适合做阵列 DOA、声源跟踪、基础分离，是机器人听觉里的关键模块。前提是有原始多通道音频、正确的通道顺序、麦克风几何参数、机器人坐标系标定。
- `pyroomacoustics`：适合仿真、算法验证和离线评估，不建议作为机器人实时主链路。
- `asteroid / torchaudio / SepFormer / MossFormer`：适合多人重叠说话时做神经语音分离，但计算和缓冲会增加延迟。建议只在 overlap 高时启用，或者用于测试回放和质量提升。
- `pyannote.audio`：适合说话人分离/说话人日志，判断“谁在什么时候说话”。它不是低延迟转头的核心，低延迟转头主要靠 VAD + DOA + 视觉。
- HF `speech-to-speech`：适合作为目标音频流之后的对话后端。它不应该承担声源定位、目标选择和权限判断。

结论：这些模块覆盖面够。当前已补上轻量 `AEC/self-echo` 接口、麦克风设备抽象、设备 profile、ASR 测试日志边界、注意力融合层和 4090 远端服务骨架；生产质量还需要真实 Sipeed 通道标定、ODAS/WebRTC APM/RNNoise/DeepFilterNet 等增强模块接入和实地调参。

## 4. 视觉层现状

`~/Golands/vision-detection` 已有：

- `/vision/signals`
- `/vision/people`
- `near_human_present`
- 最近人身份/角色
- `v_user_raw` 和高级 emotion
- 最近人 face yaw/pitch/roll
- gesture events
- `person_count`
- 最近 person bbox 面积比例

`/vision/people` 已补齐注意力层需要的多人输入：

- 每个人的 `person_id`、`face_id`、`body_id`、`voice_id` 占位。
- 从 bbox 中心和相机 FOV 估算的 azimuth/elevation。
- `gaze_score`、`body_facing_score`、`proxemic_space`、`engagement_status`。
- 手势、情绪、身份置信度字段。

ROS4HRI 标准集成时，bridge 应把它映射到 `/humans/faces`、`/humans/bodies`、`/humans/persons`、`/humans/candidate_matches` 和对应 TF frame。完整 topic 表见 `docs/TOPICS.md`。后续仍需要实机标定相机 FOV、相机到机器人 base frame 的 transform，以及更稳定的多人视觉 track。

## 5. 注意力机制

注意力层每 20-50 ms 接收声学 track，每 100-250 ms 接收视觉状态。两者不同频，不做阻塞同步；`AsyncPerceptionBuffer` 使用 TTL 缓存和新鲜度降权：

- 声学 TTL 默认 `180 ms`，过期直接丢弃。
- 视觉/person TTL 默认 `450 ms`，过期前逐步降低 gaze/facing/face confidence。
- 机器人播放/运动状态 TTL 默认 `250 ms`。
- 注意力输出限频 `<=50 Hz`。

候选目标的主要分数：

- 声学：VAD、speech probability、clarity、DOA track 稳定性、overlap、self echo probability。
- 视觉：人脸可见、看向机器人、身体朝向机器人、手势、身份角色、距离。
- 场景：机器人是否正在播放声音、是否在运动、当前是否已有稳定目标。
- 文本：wake word 或机器人名字可作为辅助特征，但 ASR 测试不把它当意图结果。

输出：

- `target_id`：`person:owner` 或 `sound:a3`。
- `listen`：是否建议把该目标流送入 speech-to-speech。
- `addressed_to_robot`：是否像是在对机器人说。
- `azimuth_deg`：给头部/眼睛控制。
- `reasons`：调试用，解释为何选择该目标。

实现上不要把算法写死。`robot-attention-perception` 的核心是插件容器：

- 默认算法：`ros4hri_native`，按 ROS4HRI 的 person/face/body/voice ID、候选匹配、engagement/proxemics 语义融合。
- 内置备用算法：`weighted_audio_visual`，使用声学和视觉加权分数，适合离线仿真和没有完整 ROS4HRI person manager 时运行。
- 自定义算法：实现 `AttentionAlgorithm`，并通过 `module:ClassName` 加载。

这意味着后续可以直接替换成自研注意力网络、行为树策略、Bayesian filter 或 LLM 辅助策略，而不改三仓库之间的数据协议。

## 6. 和 ASR 测试边界

你补充的测试文档是合理的。ASR 只记录：

- 实际说话内容。
- 环境、距离、方向、噪声、遮挡、机器人状态。
- 说话方式。
- ASR 最终文字。
- 句首、句尾、否定词、数字、名称、转述词等差异。
- 用户开始说话、结束说话、结果出现时间。
- 如果版本支持，再记录语言和声音清晰度。

注意力层、身份层、权限层、对话层和运动层另测。这个边界对系统设计很重要：ASR topic 不能偷偷输出“这是命令”“应该停止”“这是主人”这类结论。

## 7. 推荐落地路线

1. 已完成软件骨架：`voice-detection` 采集/DOA/VAD/beamforming/ROS4HRI 映射，`vision-detection` `/vision/people`，本仓库 attention core/async fusion/dashboard，`hri-memory-service` 本地存储。
2. 硬件标定：接 Sipeed 6+1，确认 8ch 48 kHz S16_LE、CH0-CH5 顺序、阵列 0 度方向、CH6/CH7 用途。
3. 生产增强：把 ODAS/WebRTC APM/RNNoise/DeepFilterNet/SepFormer 等替换进可插拔位置。
4. ROS2 bridge：把 `/vision/people` 和 `/voice/acoustic_tracks` 映射成标准 `/humans/*` 和 TF。
5. speech-to-speech：只吃注意力门控后的目标 beam/separated stream，输出机器人语音；播放状态回馈到注意力层和 AEC。S2S 是下游对话，不是 ROS4HRI 前置输入。
6. 场景验收：用你提供的 ASR 场景做真实硬件测试，记录文字差异、时间、噪声、方向、遮挡、机器人播放和多人重叠状态。
