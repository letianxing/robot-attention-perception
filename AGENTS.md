# robot-attention-perception：当前实现与交接（2026-09-16）

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
- 8092实时摄像头、表情/手势、ASR/VAD、TTS、切换设备、历史记忆分页/回溯、模型上下文、主动等待、注意力分布、状态转换。
- .run/attention-decisions.jsonl异步有界数值日志，10MB*3轮换，不录原始声音图像/向量。

## 关键文件、验证及限制
- robot_attention_perception/{interaction_fusion,visual_focus,conversation,group_conversation,turn_taking,attention_competition,competition_adapter,session_attention,reflex_attention,live_crossmodal,registration,live_runtime}.py；web/live_*。
- 最近Attention97测试通过；tests/test_daily_interactions.py为模拟传感器->Brain/mock TTS联动，不代表现场准确率。
- 完整人际addressee模型、完整麦格克音素融合、额外ROI调度/实际反馈/实体动作未完成。不要把分配建议称为已执行。

## 可插拔注意力输入（2026-09-16）
新增attention_sources.py。ATTENTION_SOURCES默认audio_visual,memory_context；可选internal_state默认关闭。聚合同会话Brain工作记忆摘要和真实内部状态，来源启停/缺失/过期显式化；不授予说话权、不创造未观测对象。ROS /brain/internal_state schema_version1为本适配入口，未改pacific-rim IDL。去掉无输入时常量motivation=.2；数学gain1是无调制，不是假装测得平静。Attention101项和DOM通过。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。没有内部状态也可试用第一版，但现场可靠性未验收，不能搬用论文准确率。

## 会话切换交接（2026-09-16，最新）
新AI先读 `/Users/letianxing/Golands/voice-detection/SESSION_HANDOFF.md`，再读本项目当前摘要。
本轮已核查：本机全部感知/Brain服务停止，162远端Qwen健康运行。最新Attention101/Brain60/DOM测试通过，不等于现场准确率。没有提交git，保护所有未提交修改和用户数据。

## 持续对视可以发起邀请 + 打断更难触发（2026-09-16，最新）
用户反馈「我盯着它很久不说话，它也不主动询问」。查 .run/attention-decisions.jsonl 的现场记录：
那一轮（17:59:37–18:01:34）这个人的 gaze≥0.65 连续 114.7 秒，engagement 一直是 0.514–0.517、
reasons 只有 gaze_engaged，从没到过 INVITED。这不是 bug，是之前写死的设计：
交流意愿层里纯注视的渐近值就在阈值以下，只有「本会话说过话」或「跨会话熟人」才可能被邀请。
现在按用户要求改：新增「持续对视」证据项（attention_common.hpp）。
  依据：双人对视平均舒适时长约 3.3 秒（Binetti et al. 2016），超过之后就不再是普通看一眼，而是一种请求。
  所以 3.3 秒才开始计分，8 秒饱和，权重 1.2，乘以识别置信度——一个忽有忽无的检测不算「在盯着谁」。
  邀请只发一次：跨过阈值后保持 2 秒（让下游看到的是状态而不是一帧），之后压制到这个人把视线移开再看回来。
  ATTENTION_REASON_SUSTAINED_GAZE 是 ABI 新增的枚举值（1<<16），没有改结构体布局，旧插件照常加载。
  旧用例 test_sustained_gaze_alone_does_not_invite 断言的正是被推翻的那条规则，已改写为
  「看两秒不算邀请、连续盯着最终会邀请一次、识别不可靠的人永远不会」。
打断（用户反馈「很容易被打断」）：
  turn_taking 的确认窗口 120/200ms -> 200/350ms。依据是人类话轮转换间隔普遍在 200ms 量级
  （Stivers et al. 2009），几十毫秒的重叠更可能是应声或口误而不是要接话；显式「停一下」「小圆」「小心」
  仍然是 80–120ms 的短窗口，那些没有歧义。
  视觉打断（嘴动+注视、还没出词）原来一帧就成立——机器人说话时人本来就会看着它、有表情反应。
  现在要求同一个人连续 400ms，且麦克风的自回声概率 <0.3（不是在听自己）。计时按人记，不按当前焦点记。
视觉逼近判据同时改了（reflex_attention.py）：原来是包围盒面积比 ≥2.5、≥0.18，和镜头有关、和快慢无关，
而且写死 face_confidence≥0.85 比检测器自身的 0.80 还严。现在按角尺寸和角扩张速度：≥10°、≥57 deg/s
（小鼠逃跑反应集中在 10–40°、57–320 deg/s，Yilmaz & Meister 2013），水平视场角 ATTENTION_CAMERA_HFOV_DEG 可配。
新增用例：走过来（1 秒角度翻倍）不算威胁。论文是头顶圆盘对小鼠，这里只当量级参考，没在本机相机标定过。
验证：attention 183、C++ 4 个目标（v1/v2 同一套用例）、离线九场景、brain 77、voice 84、vision 27 全过。

## 身份连续性：同一个 person_id 就是同一个人（2026-09-16，最新）
配合 vision 侧的 GuestGallery（见 vision-detection/AGENTS.md）。PersonManager 原来只按
face_id/body_id 的概率和方位做关联，5 秒没见到就过期，于是视觉一换 id 就是新的陌生人。
改了两点：
1) _match_visual 先看 person_id：已有轨迹的 person_id 和这次观测相同，就是同一个人，
   不再看方位和框。前提是这个 id 真的代表人——_carries_identity() 排除 anonymous_ 和 vision_face_track，
   后者是检测器对重叠框的记账，框不重叠之后它什么都不保证。这是和视觉服务的命名约定，写在函数注释里。
2) 过期窗口按 id 类型分开：带真实身份（名字或 guest）的轨迹留 60 秒（identity_retention_ms），
   只靠几何维持的仍然是 5 秒。人走开一会儿再回来是同一个人，而不是新来的。
新增用例：长间隔后同一个 person_id 不再新建人、不同 id 仍是不同人、没有身份的观测仍然只能靠几何
（8 秒后确实是新人，这是诚实的答案不是 bug）。原 test_expires_temporary_person_after_retention_window
的夹具 id 改成 vision_face_track_0001，保留它原本要测的「几何过期」语义。
验证：attention 186、vision 31、voice 84、brain 77、C++ 4 目标、离线九场景全过。

## 对话已经开着的时候，每句话不必重新自证（2026-09-17）
配合 voice-detection 的声纹修复。现场四次失败里有三次卡在 finalize_gate：
  "小圆，明天是。" 放行 → 紧接着的 "星期几？" 被拒（no_contemporaneous_dialogue_permission）
  机器人刚问完话，主人靠在椅子上回答 "我想出去玩" → 拒
  主人站在镜头前问天气 → 拒（声纹 0.0，视线不在正中）
新增 _conversation_floor()，三条短时通路，任何一条成立就放行，都会把理由写进 gate：
1) answer_to_our_question_unverified_voice：25 秒内机器人对某人说过话（RESPONDING/active_reply_target），
   期间没有别人占过话轮，且这句的声纹没有指认另一个登记人 → 归给那个人。
2) within_name_call_window：12 秒内出现过点名（live 侧 focus_origin='direct_call'，
   finalize 侧 explicit_final_directed_call）→ 放行。叫了名字之后就该进入听的状态，这不需要摄像头。
3) speech_from_the_person_we_are_attending_to：最近 1.5 秒里只有一个人、且一直是 VISUAL_FOCUS/LISTENING/RESPONDING
   → 他说的话就是对我说的。
三条都不是"认出了声音"，识别结果照旧是 unverified，conversation.py 里 identity_resolution 会标成
answer_continuity_unverified，下游能区分"认出来了"和"按对话推断的"。
别人插话、超时、声纹指认了另一个登记人，三条都立刻失效。
新增用例：AnswerContinuityTest 4 条、NameCallWindowTest 2 条。
另外加了"小袁"这个同音写法——现场 ASR 就是这么写的。
验证：attention 192、voice 89、brain 77、vision 31、离线九场景全过。
