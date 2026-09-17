# 频率、延迟与优化原则

目标是让机器人先像“听见并看过去”，再完成更慢的转写和回答。低延迟路径必须和重模型路径分离。

## 默认频率

| 模块 | 输入频率 | 输出频率 | 关键路径 |
| --- | ---: | ---: | --- |
| 音频采集 | 48 kHz PCM，10-20 ms block | 50-100 Hz raw blocks | 是 |
| VAD/DOA/beamforming | 20-50 Hz | 20-50 Hz `AcousticTrack` | 是 |
| AEC/self-echo | 20-50 Hz | 20-50 Hz | 是 |
| 视觉 people | 5-10 Hz | 5-10 Hz | 半关键 |
| 注意力融合 | mixed-rate | <=50 Hz | 是 |
| ASR partial | streaming | 2-5 Hz partial | 否 |
| ASR final | end-of-utterance | event | 否 |
| S2S/LLM/TTS | gated target stream | event/audio | 否 |
| 记忆写入 | event | event | 否 |

## 延迟预算

| 链路 | 目标 |
| --- | ---: |
| 声音出现到 VAD active | 20-60 ms |
| 声音出现到粗略 DOA | 40-120 ms |
| 声音出现到 `/attention/look_at` | 60-150 ms |
| 视觉人物更新 | 100-250 ms |
| 视觉修正 attention target | 150-350 ms |
| ASR partial 首字 | 250-700 ms |
| ASR final | 600-1500 ms after speech end |
| S2S 首音 | 0.8-3 s，取决于模型和网络 |

## 已实现的 anti-lag 机制

- `AsyncPerceptionBuffer`：声学/视觉/机器人状态不同频时不强同步。
- 声学 TTL `180 ms`：旧声音快速过期。
- 视觉 TTL `450 ms`：旧视觉逐步降权，不会立刻丢掉刚看到的人。
- 输出限频 `<=50 Hz`：避免 ROS2 消息风暴和 UI 卡顿。
- self-echo hard reject：机器人说话且回声概率高时，不进入候选排名。
- transcript 只做辅助 wake word，不阻塞转头。

## 后续优化开关

- Mac/Jetson capture 线程独立于 4090 网络发送线程；队列满时丢旧帧。
- 4090 服务内分离实时路径和重模型路径：DOA/AEC/VAD 常开，SepFormer/MossFormer2 按 overlap 阈值启用。
- 视觉推理与 MJPEG/dashboard 分线程；调试图像不进入注意力关键路径。
- 记忆服务异步写入，不阻塞 `/attention/look_at` 和 `/speech/target_stream`。
- ASR/S2S 只消费门控后的单路目标音频，避免多人混音拖慢或污染对话。
