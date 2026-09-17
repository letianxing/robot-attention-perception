# 硬件、延迟和资源判断

## 1. 麦克风兼容

| 设备 | 能力判断 | 推荐用法 |
| --- | --- | --- |
| Mac 原生麦克风 | 能做 ASR/VAD，不能可靠做多人方位定位 | 开发调试、单人近场 |
| Windows 原生麦克风 | 能做 ASR/VAD，WASAPI 可统一采集 | 开发调试、单人近场 |
| Sipeed 6+1 Mic Array USB | 截图显示 UAC2.0、8CH S16_LE 48 kHz，另有串口 16x16 声场图/2 Mbps；适合阵列 DOA 原型 | 原始 8ch 给 ODAS，或串口声场图给快速 DOA；ASR 用 beamformed/selected channel |
| Reachy Mini 自带音频 | 官方资料说明有 4 麦阵列、默认 AEC，并提供 DOA 读取能力 | 机器人原生部署、回声消除、播放状态闭环 |
| ReSpeaker/Matrix/其他环麦 | 只要能暴露 raw multichannel PCM，就能接入同一抽象 | 需要 profile：通道数、采样率、麦克风坐标、通道顺序 |

关键点：内置单麦/双麦不等于鸡尾酒效应。多人方位定位需要阵列几何、通道同步和标定。

## 2. 延迟预算

| 链路 | 目标延迟 | 说明 |
| --- | ---: | --- |
| 采集 frame | 10-20 ms | 低延迟音频块，48 kHz 下 480-960 samples |
| VAD | 10-30 ms | 可实时 |
| DOA/声源跟踪 | 30-80 ms | ODAS/声场图可满足转头和注意力 |
| AEC/self-echo 判定 | 10-40 ms | 机器人说话时必须常开 |
| 注意力融合 | <10 ms | 规则/轻模型即可 |
| 视觉人物状态 | 100-250 ms | 5-10 Hz 已够融合，转头可以由声学先触发 |
| ASR partial | 250-600 ms | 取决于本地模型或云服务 |
| ASR final | 600-1500 ms after speech end | 对测试文档必须记录 end-to-final 时间 |
| 神经分离 | +100-800 ms | 只建议 overlap 高或回放测试时启用 |
| Speech-to-speech 首音 | 0.8-3 s | 注意力门控后的下游对话延迟，取决于 ASR、LLM、TTS 和网络/本地推理 |

注意力转头不应等待 ASR final。机器人应先靠 VAD + DOA + 视觉估计看过去，再把目标声流送进 ASR/S2S。

## 3. RTX 4090 单机可行性

RTX 4090 24 GB 显存足够支撑一个实用版本：

- 视觉：YOLOv8n/人脸/手势/轻 emotion 占用很小；多相机或高分辨率才明显增加。
- 声学前端：VAD、DOA、AEC 多数在 CPU 上就能跑。
- ASR：Whisper/Parakeet 类中小模型通常可接受；大模型要看精度和延迟目标。
- diarization：可作为异步日志和多人测试辅助，不建议卡在 100 ms 注意力路径。
- neural separation：显存可承受，但延迟会增加，建议按需启用。
- S2S：如果同机还跑 LLM/TTS，建议用量化 7B/14B 或云端 LLM；32B 以上常驻会压缩视觉/ASR余量。

推荐生产调试机器：

- GPU：RTX 4090 24 GB。
- RAM：64 GB 更稳，32 GB 可做轻量版本。
- 磁盘：预留 80-200 GB 给 Hugging Face、ASR、TTS、LLM、视觉模型缓存；固定部署后可裁剪到 20-60 GB。

## 4. 模型和进程放置

低延迟进程：

- audio capture
- AEC
- VAD
- DOA/source tracking
- attention fusion

中延迟进程：

- ASR partial/final
- vision detection
- diarization

高延迟或按需进程：

- neural speech separation
- long-context LLM
- high-quality TTS
- offline test scoring

这样做的原因是：机器人“看向说话人”的体验由 100 ms 级感知决定，而“回答内容”可以慢一些。

## 5. 对 Sipeed 6+1 的接入检查单

1. 用系统音频枚举确认 8 个输入通道和 48 kHz。
2. 录制敲击每个麦克风附近的测试音，确认通道顺序。
3. 写入麦克风相对中心的 3D 坐标。
4. 确认机器人正前方对应的 azimuth=0。
5. 如果使用串口声场图，确认 16x16 网格到机器人 azimuth/elevation 的映射。
6. 给 ODAS 或本项目 `AcousticTrack` 输出同一坐标系。
7. 机器人播放声音时，把播放参考音频送入 AEC/self-echo 判定。

## 6. 现有视觉项目要补的内容

当前 `vision-detection` 对主目标足够，但多人测试需要：

- `PersonObservation[]`：每个人独立 track。
- 每个人的 bearing：从 bbox 中心和相机内参估计 azimuth/elevation。
- 视觉 track id 的稳定性。
- 每个人的 face/body facing score。
- 手势和身份绑定到具体 person。

没有这些字段，注意力层只能在“最近人/主脸”场景可靠工作，无法严谨验证多人互换位置和相近方向场景。
