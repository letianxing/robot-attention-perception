# 架构图与数据链路

## Mac + Docker 首测

```mermaid
flowchart LR
  CAM[Mac Camera / RealSense] --> V[vision-detection]
  MIC[Mac Mic / Sipeed 8ch] --> A[voice-detection]
  V -->|People typed + JSON| RB[rosbridge]
  A -->|AcousticTracks typed + JSON| RB
  RB --> S[C++ attention_stack S1-S5 + BCI]
  S --> H[ROS4HRI /humans + TF]
  S --> G[attention gate]
  G --> D[S2S / TTS]
  V --> M[hri-memory-service]
  A --> M
  S --> M
  D --> M
```

Mac 原生进程负责 CoreAudio/AVFoundation 采集，Docker Desktop 运行 ROS2 Humble、ROS4HRI ingress 和 C++ attention。换 Jetson 后只迁移采集端，强类型 topic 与融合算法不变。

## 四系统总图

```mermaid
flowchart LR
  CAM[Mac camera / robot camera] --> VISION[vision-detection]
  MIC[Sipeed 6+1 / Mac mic / Reachy mic] --> VOICE_HOST[voice-detection host capture]
  SPEAKER[Robot TTS speaker] --> VOICE_HOST
  VOICE_HOST <-- raw 8ch PCM / frontend output --> GPU4090[4090 acoustic algorithms]
  VISION -->|/vision/people + /humans/faces,bodies,persons| ATT[robot-attention-perception]
  VOICE_HOST -->|/humans/voices + /voice/acoustic_tracks| ATT
  ATT -->|/speech/target_stream| S2S[speech-to-speech downstream]
  S2S -->|TTS audio + playback_state| SPEAKER
  ATT -->|/attention/look_at| MOTION[eyes/head/motion control]
  VISION --> MEMORY[hri-memory-service]
  VOICE_HOST --> MEMORY
  ATT --> MEMORY
  S2S --> MEMORY
```

## 声学内部回路

```mermaid
sequenceDiagram
  participant Mic as Sipeed USB 8ch
  participant Host as Mac/Jetson voice-detection
  participant GPU as 4090 acoustic service
  participant Att as robot-attention-perception
  participant S2S as speech-to-speech
  participant Spk as Robot speaker

  Mic->>Host: 8ch 48 kHz S16_LE raw PCM
  Spk->>Host: playback reference
  Host->>GPU: raw multichannel chunks + profile_id
  GPU->>GPU: DOA/beamforming/AEC/NS/separation/ASR optional
  GPU-->>Host: AcousticTrack + enhanced target audio
  Host->>Att: /humans/voices/* and /voice/acoustic_tracks
  Att->>S2S: gated target stream only
  S2S->>Spk: TTS playback
  S2S->>Host: playback_state/reference feedback
```

这个回路是正常的：上位机离 USB 硬件近，4090 离重模型近。不要用 QuickTime/系统录音替代 host capture，因为它们通常拿不到原始 8ch。

## ROS4HRI 数据链路

```mermaid
flowchart TB
  VPeople[/vision/people/] --> VBridge[ROS4HRI vision bridge]
  VBridge --> Faces[/humans/faces/tracked/]
  VBridge --> Bodies[/humans/bodies/tracked/]
  VBridge --> Persons[/humans/persons/tracked/]
  VBridge --> Matches[/humans/candidate_matches/]
  VBridge --> VJson[/vision/people_json/]

  VTracks[/voice/acoustic_tracks/] --> VoiceBridge[ROS4HRI voice bridge]
  VoiceBridge --> Voices[/humans/voices/tracked/]
  VoiceBridge --> VoiceSpeech[/humans/voices/<voiceID>/speech/]
  VoiceBridge --> VoiceFeat[/humans/voices/<voiceID>/features/]

  Faces --> Async[AsyncPerceptionBuffer]
  Bodies --> Async
  Persons --> Async
  Matches --> Async
  Voices --> Async
  VoiceSpeech --> Async
  VoiceFeat --> Async
  VJson --> Async
  Async --> Algo[pluggable AttentionAlgorithm]
  Algo --> State[/attention/state/]
  Algo --> LookAt[/attention/look_at/]
  Algo --> Target[/speech/target_stream/]
```

当前实现提供两条联调路径：

- 轻量 JSONL：`attention_json_bridge.py` 直接读 vision/acoustic/robot 事件，适合无 ROS2 的 Mac 仿真和回归测试。
- ROS2 bridge：`ros4hri_vision_bridge.py`、`ros4hri_voice_bridge.py`、`ros4hri_attention_node.py` 把工程 topic 转成 ROS4HRI 风格 topic，再进入 attention gate。

## 频率不一致的融合

```mermaid
flowchart LR
  A[Audio track 20-50 Hz] --> AC[Audio TTL cache 180 ms]
  V[Vision people 5-10 Hz] --> VC[Vision TTL cache 450 ms]
  R[Robot playback/motion 20-50 Hz] --> RC[Robot TTL cache 250 ms]
  AC --> F[Fusion tick <=50 Hz]
  VC --> F
  RC --> F
  F -->|fresh audio quick reaction| Look[/look_at/]
  F -->|fresh or decayed vision| Gate[S2S gate]
```

核心规则：

- 不阻塞等待视觉；音频先触发转头。
- 视觉过期不是瞬间失效，而是 gaze/facing/confidence 降权。
- 声学过期很快丢弃，防止机器人追着旧声音。
- 机器人正在播放且 self-echo 高的 track 直接拒绝，防止自说自听。

## ASR 测试数据链路

```mermaid
flowchart LR
  Test[测试者/环境记录] --> Voice[voice-detection ASR adapter]
  Voice --> Log[/voice/asr_records/]
  Log --> Mem[hri-memory-service]
  Voice --> Speech[/humans/voices/<voiceID>/speech/]
  Speech --> Mem
  Speech --> Att[attention only as text feature]
```

ASR 测试只记录文字、语言/清晰度、差异和时延，不在 ASR 层判断意图、权限、是否该回应。
