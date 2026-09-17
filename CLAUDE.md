# robot-attention-perception：实现交接（2026-09-16）

## 工作规则与实际部署
- 用户授权跨六项目协作；大量未提交实现不可擅自清理或回滚。不创建子代理，除非用户明确要求。
- 修改前检查并停止本机受管服务，不擅自打开摄像头/麦克风或播放。远端162的Qwen为常驻服务，不能随本机stop_all误停。
- 机器人名称“小圆”。用户要求自然多人对话、旁人记忆、视听身份、注意力约束打断、主动搭话。声纹/视觉冲突优先可靠声纹归属发言，不据此改写冲突人脸。
- 本机启动：cd ~/Golands/biomimetic-brain-test && bash scripts/start_all.sh；停止：bash scripts/stop_all.sh；状态：python3 scripts/mac_stack.py status。
- Python：~/Golands/robot-attention-perception/.venv-mac/bin/python。MediaPipe固定0.10.32，不盲目升级。
- 端口：Vision8080、Voice8090、Attention8092、Brain8094、Memory8788、ROS9090；慢脑http://172.16.60.162:18050/v1/chat/completions，model=qwen3.5-9b。
- 所有基准和单元测试仅说明对应样本/程序行为，不能描述为现场识别准确率达标。
- 不在代码/文档保存SSH密码。完整历史见voice-detection/AGENTS.md下方历史记录；本节当前状态优先于过期历史。
- /Applications/声音/Mac全系统启动与三项验收.txt保持“一键启动／分别启动”两块，详细验收独立保存。

## 本项目已实现
- InteractionFusion统一裁决视觉聚焦、听觉叫名、当前发言、回复目标保持、环境对话和反射；上游ROS/local仅候选，不凭过期listen放行。
- finalize_gate按同句时间/声源证据决定最终ASR；单纯看人不等于向机器人发言。叫名可在镜头无人/普通音量触发；句中提及不触发。
- 进入持续注视约800ms；保持阈值宽松、短丢失650ms保留目标不放行、明确转开300ms释放；语音结束继承目标。
- 连续视觉/听觉归一化竞争、实际dt、习惯化/恢复/返回抑制、有限可靠跨模态增益、独立两路焦点、变化去重。人物与已跟踪物体参加竞争。
- session_attention.py订阅Brain /brain/state：同会话活跃交流增强目标，1200ms过期或完成/取消/冲突撤销。feedback和额外处理份额已有计算接口，但没有额外感知执行器，显式executable=false。
- Person保留owner角色；可靠声纹优先归属发言，冲突记identity_conflict/visual_person_id，已移除唯一可见脸被声音自动改名的捷径。独立声纹池更新worker，避免阻塞融合。
- GroupConversation不限逻辑参与人数，明确邀请开启120s群体会话；向人称呼拒绝机器人门控，明确群体问句可参与，听话人不确定保持unknown。
- turn_taking.py附和/叫名/明确停止/正常让话分级，VAP不独立授权；确认信息进入Brain。
- audio/visual startle与orient；固定脸框变化与物体大面积突然出现确认，独立对话许可。真实双闪脉冲绑定、有限双唇音口型冲突实验。
- 8092实时摄像头、表情/手势、ASR/VAD、TTS、切换设备、历史记忆分页/回溯、模型上下文、主动等待、注意力分布、状态转换、注意力输入源勾选与算法(.so)切换、交流意愿状态。
- .run/attention-decisions.jsonl异步有界数值日志，10MB*3轮换，不录原始声音图像/向量。

## 关键文件、验证及限制
- robot_attention_perception/{interaction_fusion,visual_focus,conversation,group_conversation,turn_taking,attention_competition,competition_adapter,attention_sources,attention_plugin,linguistic_context,session_attention,reflex_attention,live_crossmodal,registration,live_runtime}.py；web/live_*。
- 算法插件：include/robot_attention_perception/attention_plugin_abi.h、src/plugins/av_memory_language_v1.cpp、plugins/libattention_*.so、tests/cpp_attention_plugin_test.cpp、scripts/build_attention_plugins.sh、scripts/verify_attention_algorithm.py。
- 最近Attention132测试通过；tests/test_daily_interactions.py为模拟传感器->Brain/mock TTS联动，不代表现场准确率。
- 完整人际addressee模型、完整麦格克音素融合、额外ROI调度/实际反馈/实体动作未完成。不要把分配建议称为已执行。

详细历史与最新追加记录请同时阅读本项目AGENTS.md，冲突以用户最新指令为准。

## 统一注意力竞争（第五版，2026-09-16 最新）
理论链已全部落地：Itti-Koch 显著性 → Biased Competition → Reynolds-Heeger 除性归一化（v2，归一化池跨模态耦合 κ=0.35）
→ Selective Tuning（v2 方位抑制环 + workspace soft-WTA 压制非赢家）→ Habituation/IOR/Hysteresis → Global Workspace → Brain。
唯一没做的是层级 Selective Tuning：候选空间是扁平的 person/track/object，没有真正的 part-whole 层级，硬套是空架子。
输入源五个（8092 可勾选，存 .run/attention-config.json）：audio_visual、memory_context、linguistic_context、
cross_session_memory、internal_state（默认关闭，无生产者）。关闭=消融该通道证据，不是关硬件或安全反射。
算法为独立 .so（编译 bash scripts/build_attention_plugins.sh）：av_memory_language_v1（默认，自上而下相加）、
av_memory_language_v2（乘性注意场 + 跨模态池 + 抑制环）、builtin_audio_visual_v1（仅视听基线）。
v1/v2 跑同一套行为用例全过；现场优劣未比较，所以默认仍是 v1。缺 .so 不阻断启动，回退基线并报错。
Global Workspace（global_workspace.py）是统一订阅点也是认知层竞争：soft-WTA(share=a²/Σa²) + 有限容量 4 +
urgency≥0.7 抢占；非感知候选（记忆、惊跳）在这里竞争，因为它们没有模态。
入口 GET /api/workspace、?since=N 回放、/api/workspace/stream (SSE)、ROS /attention/workspace。订阅者落后丢帧并计数。
双向：Memory→Attention 用 memory_candidates.py（保守：只提未完成的事、每 5 分钟一次、人在场、安静 3 秒、
同一件事只提一次，ATTENTION_MEMORY_EVENTS=0 可关）；Brain→Attention 用 attention_memory.goals 作 top-down bias
（awaiting_answer/holding_floor/deferred_turn/greet_owner，只偏置不放行）。
记忆通道整体乘以识别置信度；角色（主人/陌生人）不进交流意愿层，只留在第一层 importance 影响「看哪里」。
模型：声纹 ERes2NetV2（192 维）；人脸走商用路线，默认 sface(Apache-2.0) + 质量门槛 + 5 帧模板均值，
arcface(buffalo_l) 已下载但权重仅限非商业研究，--face-backend arcface 可切；身份识别 300ms 节流。
Attention 164、Brain 62、Vision 23、C++ 4 个测试目标、页面 DOM 通过；
scripts/verify_attention_algorithm.py 为不开硬件的端到端离线检查。
权重与阈值是工程先验，未做现场 ROC 标定；论文数字不能当本机准确率。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。

## 可插拔注意力（第四版历史，2026-09-16）
输入源五个（8092 可勾选，存 .run/attention-config.json）：audio_visual、memory_context、linguistic_context、
cross_session_memory、internal_state（默认关闭，无生产者）。关闭=消融该通道证据，不是关硬件或安全反射。
算法为独立 .so（C ABI 见 include/robot_attention_perception/attention_plugin_abi.h，编译 bash scripts/build_attention_plugins.sh）：
av_memory_language_v1（默认，自上而下相加）、av_memory_language_v2（Reynolds-Heeger 乘性注意场 + Selective Tuning
方位抑制环，交流意愿层与 v1 共享 attention_common.hpp）、builtin_audio_visual_v1（内置仅视听基线）。
v1/v2 跑同一套行为用例且全过；两者优劣未经现场对比，所以默认仍是 v1。缺 .so 不阻断启动，回退基线并报错。
统一订阅点 global_workspace.py：每个感知周期广播一份内容（cycle/focus/target/engagement/coalition/sources/
algorithm/provenance）。入口 GET /api/workspace、/api/workspace?since=N 回放、/api/workspace/stream (SSE)、
ROS /attention/workspace。订阅者落后会丢帧并计数，不阻塞感知循环。Brain 现有 brain_input 通路未改。
结构：自下而上（视听显著度、声学突变、句首唤醒词）+ 自上而下（工作记忆目标、熟悉度先验）的归一化竞争，
其上再加一层按人的序贯证据累积决定「是不是在跟我交流」。内部状态将来接 arousal/motivation 与第二层通道。
记忆通道整体乘以识别置信度；角色（主人/陌生人）不进交流意愿层，只留在第一层 importance 影响「看哪里」。
模型已替换：声纹 CampPlus -> ERes2NetV2（192 维，3 秒语音 14.1->53.9ms，句末计算）；
人脸新增可切换后端，默认仍是 sface（Apache-2.0），arcface(buffalo_l w600k_r50) 已下载但权重仅限非商业研究，
需确认用途后用 --face-backend arcface 启用；身份识别加 300ms 节流。旧声纹/人脸档案已按指示清空，
备份在 ~/Golands/.identity-backup-*。阈值未在真人数据上标定。
Attention 151、Brain 62、Vision 16、C++ 4 个测试目标、页面 DOM 通过；
scripts/verify_attention_algorithm.py 为不开硬件的端到端离线检查（九个场景，含三组消融/对照）。
权重与阈值是工程先验，未做现场 ROC 标定；论文数字不能当本机准确率。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。

## 可插拔注意力（第三版历史，2026-09-16）
注意力输入源与算法都已解耦，8092 面板「注意力输入源与算法」可勾选来源、切换算法，选择存 .run/attention-config.json。
来源五个：audio_visual、memory_context（本会话工作记忆）、linguistic_context（当前语境文本）、
cross_session_memory（跨会话熟悉度，后台查 hri-memory-service，快循环只读缓存）、internal_state（默认关闭，无生产者）。
算法为独立 .so，C ABI 见 include/robot_attention_perception/attention_plugin_abi.h，编译 bash scripts/build_attention_plugins.sh：
av_memory_language_v1（默认，视听+记忆+语境，含按人累积的交流意愿层）、builtin_audio_visual_v1（内置仅视听对照基线）。
缺少 .so 不阻断启动，自动回退基线并在面板显示加载错误与「无交流意愿层」。
结构上是 自下而上（视听显著度、声学突变、句首唤醒词）+ 自上而下（工作记忆目标、熟悉度先验）的归一化竞争，
其上再加一层按人的序贯证据累积决定「是不是在跟我交流」。内部状态将来作为 arousal/motivation 接第一层、连续通道接第二层。
记忆通道整体乘以识别置信度：身份不可靠时那份历史不能替当前这个人说话；角色（主人/陌生人）不进入交流意愿层，
只保留在第一层 importance 影响「看哪里」，权限与内容仍归 Brain。
无声邀请、转头回答、自言自语后转向、称呼他人、熟人回访这些情况不再有各自的代码分支，由同一次融合更新产生。
Attention 143 项、Brain 62 项、C++ attention_plugin_test 9 项、页面 DOM 通过；
scripts/verify_attention_algorithm.py 为不开硬件的端到端离线检查（九个场景，含三组消融/对照）。
权重是工程先验，未做现场 ROC 标定；论文数字不能当本机准确率。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。

## 可插拔注意力（第二版历史，2026-09-16）
注意力输入源与算法都已解耦，8092 面板"注意力输入源与算法"可勾选来源、切换算法，选择存 .run/attention-config.json。
来源：audio_visual、memory_context、linguistic_context（当前语境文本，新增）、internal_state（默认关闭，无生产者）。
算法为独立 .so，C ABI 见 include/robot_attention_perception/attention_plugin_abi.h，编译 bash scripts/build_attention_plugins.sh：
av_memory_language_v1（默认，视听+记忆+语境，含按人累积的交流意愿层）、builtin_audio_visual_v1（内置仅视听对照基线）。
缺少 .so 不阻断启动，自动回退基线并在面板显示加载错误与"无交流意愿层"。
无声邀请、转头回答、自言自语后转向、称呼他人这些情况不再有各自的代码分支，由同一次融合更新产生。
Attention 132 项、Brain 66 项、C++ attention_plugin_test、页面 DOM 通过；scripts/verify_attention_algorithm.py 为不开硬件的端到端离线检查。
权重是工程先验，未做现场 ROC 标定；论文数字不能当本机准确率。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。

## 可插拔注意力输入（第一版历史，2026-09-16）
新增attention_sources.py。ATTENTION_SOURCES默认audio_visual,memory_context；可选internal_state默认关闭。聚合同会话Brain工作记忆摘要和真实内部状态，来源启停/缺失/过期显式化；不授予说话权、不创造未观测对象。ROS /brain/internal_state schema_version1为本适配入口，未改pacific-rim IDL。去掉无输入时常量motivation=.2；数学gain1是无调制，不是假装测得平静。Attention101项和DOM通过。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。没有内部状态也可试用第一版，但现场可靠性未验收，不能搬用论文准确率。

## 会话切换交接（2026-09-16，最新）
新AI先读 `/Users/letianxing/Golands/voice-detection/SESSION_HANDOFF.md`，再读本项目当前摘要。
本轮已核查：本机全部感知/Brain服务停止，162远端Qwen健康运行。最新Attention101/Brain60/DOM测试通过，不等于现场准确率。没有提交git，保护所有未提交修改和用户数据。
