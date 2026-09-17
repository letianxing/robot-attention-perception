# 开源模型与训练策略

第一版采用“开源权重 + 现场标定 + 保守概率门控”。只有验收场景出现稳定、可复现且无法通过标定解决的错误，才微调模型。

| 能力 | 系统 | 第一版开源方案 | 训练判断 |
| --- | --- | --- | --- |
| 人/脸/物体 | `vision-detection` | YOLOv8n + YuNet，MediaPipe 可选 | 先不训练 |
| 姿态/手势/注视 | `vision-detection` | MediaPipe；L2CS-Net 插件 | 安装角度产生系统偏差时再微调 |
| 深度/点云 | `vision-detection` | RealSense 对齐深度；Mac 单目低置信代理 | 不训练，做外参标定 |
| DOA/跟踪/波束 | `voice-detection` | C++ 基线 + ODAS 插件；HARK 对照 | 不训练，标定阵列几何/传递函数 |
| AEC/降噪 | `voice-detection` | WebRTC AEC3 目标实现 | 不训练，标定播放延迟 |
| 重叠语音分离 | 4090 voice worker | Asteroid/SpeechBrain，按 overlap 触发 | 指标不足才微调目标说话人提取 |
| ASR | 4090；Mac smoke | FunASR/SenseVoice/Whisper；Mac Vosk/whisper.cpp | 先热词，指标不足再微调 |
| 环境声音 | voice 低频插件 | PANNs/AudioSet | 机器人事件类别不足再训分类头 |
| Active speaker | attention AV Binder | Light-ASD/TalkNet/SyncNet adapter | 先校准；必要时用机器人视角数据微调 |
| 声纹/身份 | voice -> PersonManager | SpeechBrain ECAPA-TDNN | 先阈值校准；必要时做远场域适配 |
| AVSR | 4090 可选教师 | AV-HuBERT 离线参考 | 中文视觉纠错确有收益时再微调 |
| 注意力/共同来源 | attention | C++ S1-S5 + Bayesian causal inference | 不先训练 |

## 不进入实时主链路

- SoundSpaces 用于离线生成 RIR、仿真和 sim-to-real 测试，不是机器人运行时组件。
- AV-HuBERT 仓库已归档、模型大、依赖旧且以英语数据为主，只作为离线教师候选。
- DINOv2、SAM2、GroundingDINO、CLIP 用于低频语义查询，不占用 20-50 Hz 人/声注意力循环。
- MAVA、iCub/GASP、CTCNet 和心理物理论文用于设计状态机、损失和评测，不作为强运行依赖。
- HARK 与 ODAS 是互换插件，不在同一流上重复运行。ODAS/odas_ros 为 GPLv3，产品化时保持独立进程边界并做许可证确认。

## 真需要训练时收集什么

1. Sipeed 原始 8ch 48 kHz、每人方位、目标干净参考和机器人播放参考。
2. RealSense RGB、对齐深度、嘴部 ROI、face/body/voice transient ID。
3. 同步人脸与声纹注册样本，覆盖 0.5/1.5/3/6 米、不同方向、噪声和机器人运动。
4. 中文 ASR 原话、边界、噪声条件和最终转写。

训练优先级仅在指标失败后启用：active-speaker 概率校准 > 远场声纹域适配 > 目标说话人提取 > 中文 AVSR。不从零训练基础视觉、ASR 或分离网络。

## 来源

- HARK: https://hark.jp/
- ODAS ROS2: https://github.com/introlab/odas_ros/tree/ros2
- SoundSpaces: https://github.com/facebookresearch/sound-spaces
- AV-HuBERT: https://github.com/facebookresearch/av_hubert
- SpeechBrain: https://github.com/speechbrain/speechbrain
- RealSense ROS: https://github.com/realsenseai/realsense-ros
- ROS4HRI: https://ros4hri.github.io/standard.html
