# 资料链接

这些链接用于约束架构判断，避免把 speech-to-speech、采集、DOA 和注意力层混为一谈。

- Reachy Mini 官方文档：`https://huggingface.co/docs/reachy_mini`
- Reachy Mini media architecture：`https://huggingface.co/docs/reachy_mini/SDK/media-architecture`
- Reachy Mini advanced media controls：`https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/media_advanced_controls`
- Pollen Robotics Reachy Mini SDK：`https://github.com/pollen-robotics/reachy_mini`
- Hugging Face speech-to-speech：`https://github.com/huggingface/speech-to-speech`
- ODAS：`https://github.com/introlab/odas`
- PortAudio：`https://www.portaudio.com/`
- python-sounddevice：`https://python-sounddevice.readthedocs.io/`
- miniaudio：`https://miniaud.io/`
- pyannote.audio：`https://github.com/pyannote/pyannote-audio`
- Asteroid：`https://github.com/asteroid-team/asteroid`
- pyroomacoustics：`https://github.com/LCAV/pyroomacoustics`

## ROS4HRI

- Standard overview: <https://ros4hri.github.io/standard.html>
- REP-155 specification: <https://www.ros.org/reps/rep-0155.html>
- Interactive social robots tutorial: <https://ros4hri.github.io/tutorials/interactive-social-robots/zero-to-llm.html>

本项目采用的约束：

- HRI 感知 topic 放在 `/humans/` 下。
- `Person` 是长期身份；`Face`、`Body`、`Voice` 是临时观测 ID。
- 标准 TF frame 包括 `face_<faceID>`、`gaze_<faceID>`、`body_<bodyID>`、`voice_<voiceID>`。
- ROS4HRI 教程里的 attention/chat/control 方框适合作为架构参考，但不全是可直接复用的开源包；本仓库实现自己的可插拔 attention container，并消费 ROS4HRI 兼容输入。
