# ASR 测试边界映射

这份文档把语音识别测试场景映射到系统日志字段。原则：ASR 只证明“听到的语音转成了什么文字”，不证明“机器人该不该听、该不该执行、该怎么回答”。

## 统一日志字段

```json
{
  "scenario_id": "quiet_natural_sentence",
  "utterance_expected": "测试者实际原句，保留停顿、重复、改口",
  "environment": {
    "place": "office",
    "distance_m": 1.5,
    "relative_direction_deg": 0,
    "background_noise": "air_conditioner",
    "occlusion": "none",
    "robot_state": "speaking|moving|idle"
  },
  "speaking_style": {
    "speed": "normal|fast|slow",
    "volume_dba": 62,
    "accent": "普通话/粤语口音/东北口音等",
    "language_mix": "zh-CN,en",
    "non_speech": ["laugh", "cough"]
  },
  "asr_result": {
    "track_id": "a_owner",
    "text": "最终识别文字",
    "language": "zh-CN",
    "clarity": 0.82,
    "is_final": true,
    "started_ms": 760,
    "ended_ms": 980,
    "emitted_ms": 1250
  },
  "diff_notes": {
    "prefix_missing": false,
    "suffix_missing": false,
    "missing_keywords": ["不"],
    "extra_words": [],
    "number_changes": [],
    "name_changes": [],
    "merged_other_speaker": false,
    "captured_background_speech": false
  }
}
```

## 场景覆盖关系

| 用户测试场景 | 系统必须提供的观测 |
| --- | --- |
| 基础语句、短词、长句 | `SpeechTranscript.text/is_final/started_ms/ended_ms/emitted_ms` |
| 人名、地点、产品词汇 | 原文、最终转写、名称差异记录，不在 ASR 中判断身份 |
| 数字、日期、单位 | 最终转写和差异记录，中文数字/阿拉伯数字归一由测试团队处理 |
| 中英文混合 | `language` 可选；文字必须保留中英文片段 |
| 否定、多重否定、转述、时间、条件 | 只记录关键词是否存在和顺序，不做意图判断 |
| 语速、音量、口音、咳嗽、停顿、改口 | 记录 `speaking_style`，保留口头语和改口内容 |
| 距离、方向、遮挡 | 记录环境条件和转写结果；DOA 正确性属于声音方向测试 |
| 噪声、音乐、强噪声 | 记录是否漏字、错字、重复、混入背景内容 |
| 机器人播放声音、用户插话 | 记录 self-echo 是否进入转写，用户插话文字是否缺句首 |
| 多人说话 | 能分离时每个 track 一条转写；不能分离时记录混合转写 |
| 长时间、网络异常、重启 | 记录空转、重复、迟到、断网恢复后的转写时间 |

## 不放进 ASR 结论的内容

- 这句话是不是在对机器人说。
- 这句话属于停止、跟随、召回、暂停、恢复还是否定。
- 说话人是谁、是否有权限。
- 多人同时说话时应该回应谁。
- 机器人是否回答、停止、转头或移动。
- 突然声音是否引起惊跳。

这些由注意力融合、意图路由、身份识别、权限判断、对话决策和运动控制分别测试。

## 对声学系统的直接要求

- ASR 输出必须按 `track_id` 绑定声学流，哪怕当前只有混合流，也要使用稳定 id。
- 每条最终转写必须包含开始、结束、输出时间，支持计算 end-to-final latency。
- 多人场景如果无法分离，不能伪造多条转写；只能记录混合结果。
- 机器人自播报期间必须记录 self-echo probability 或 AEC 状态，用于解释是否把机器人自己的声音写进转写。
- 轻声、喊话、噪声、回声场景必须保留 `clarity` 或等价质量字段；如果当前 ASR 后端不给质量分，就由声学前端提供 SNR/clarity proxy。

