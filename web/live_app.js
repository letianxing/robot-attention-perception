const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = (v, digits = 2) => v === null || v === undefined || v === '' || !Number.isFinite(Number(v)) ? '--' : Number(v).toFixed(digits);
const timeText = ms => ms ? new Date(Number(ms)).toLocaleTimeString('zh-CN', {hour12:false}) + '.' + String(Math.floor(Number(ms) % 1000)).padStart(3,'0') : '--';
const elapsed = (end, start) => end && start ? `${Math.max(0, Math.round(end-start))} ms` : '--';
let latestState = {}, latestBrain = null, devices = {}, devicesSynced = false, ttsBusy = false, brainError = '', signalHistory = [];
const health = (id, ok) => $(id).classList.toggle('ok', Boolean(ok));
const interactionLabels={IDLE:'无目标',ACQUIRING:'确认注视',VISUAL_FOCUS:'视觉关注，等待开口',AUDIO_FOCUS:'被点名，关注声源',LISTENING:'正在听取对我的发言',RESPONDING:'保持回话目标',ORIENTING:'事件定向',AMBIENT_SPEECH:'环境对话，仅记录',INVITED:'无声邀请，可开口接话'};
const engagementLabels={IDLE:'无交流意愿',OBSERVING:'在观察，未达交流阈值',ENGAGED:'正在与我交流',INVITED:'无声邀请我接话',EXPECTED_ANSWER:'在回答我刚才的问题'};
const reasonLabels={gaze_engaged:'注视与朝向',lip_audio_sync:'唇动与声音同步',directed_call:'句首点名',memory_participant:'近期对话参与者',expected_answer:'我还在等他回答',topic_continuation:'延续同一话题',addresses_other:'称呼的是别人',backchannel:'附和',silent_invitation:'持续注视但未开口',dialogue_target:'正在进行的对话对象',internal_state:'内部状态',sound_novelty:'声音突变',familiar_across_sessions:'以往会话聊过'};
let attentionConfig=null;
const row = (title, detail) => `<div class="row"><strong>${esc(title)}</strong><span>${esc(detail)}</span></div>`;
function retainScroll(element, content) { const top = element.scrollTop; element.innerHTML = content; if (top > 0) element.scrollTop = top; }
async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {...options, cache:'no-store', signal: options.signal || AbortSignal.timeout(2500)});
  const data = await response.json();
  if (!response.ok || (data.error && !('s2s' in data))) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function refreshMicrophones() {
  const selected = $('microphone').value, required = Number($('profile').selectedOptions[0]?.dataset.channels || 1);
  $('microphone').innerHTML = (devices.microphones || []).filter(d => Number(d.max_input_channels || 0) >= required).map(d => `<option value="${esc(d.index)}">${esc(d.name)} · ${d.max_input_channels}ch</option>`).join('') || '<option value="">未检测到符合通道要求的麦克风</option>';
  if ([...$('microphone').options].some(o=>o.value===selected)) $('microphone').value=selected;
}
async function configure() {
  const config = await fetchJSON('/api/config');
  $('historySession').value = config.session_id;
  loadHistory(true);
  $('session').textContent = `${config.session_id} · 摄像头 → ASR / VAD → 注意力 → Brain / TTS`;
  $('cameraStream').src = config.vision_stream_url;
  for (const name of ['vision','voice','brain']) $(name+'Link').href=config[name+'_url'];
  $('cameraStream').onerror = () => { $('targetOverlay').textContent='画面连接中断'; setTimeout(()=>{$('cameraStream').src=config.vision_stream_url+'?t='+Date.now();},3000); };
  devices = await fetchJSON('/api/devices');
  $('camera').innerHTML = (devices.cameras || []).map(d=>`<option value="${esc(d.source_type)}:${esc(d.index)}">${esc(d.name)}</option>`).join('') || '<option value="opencv:0">Camera 0</option>';
  $('profile').innerHTML = (devices.profiles || []).map(d=>`<option value="${esc(d.id)}" data-channels="${d.input_channels}">${esc(d.label)} · ${d.input_channels}ch</option>`).join('');
  refreshMicrophones(); $('profile').onchange=refreshMicrophones;
}
function render(state) {
  latestState = state;
  const diagnostics=state.diagnostics || {}, voice=state.voice || {}, vision=state.vision?.state || state.vision || {}, attention=state.attention || {};
  const vad=voice.last_vad || {}, transcript=voice.last_transcript || {}, partial=voice.last_streaming_transcript || {}, track=voice.last_track || {};
  const frameMs=vision.stamp_ms || (vision.stamp ? vision.stamp*1000 : 0), frameAge=frameMs ? Date.now()-frameMs : null;
  health('visionHealth', diagnostics.vision_ok && frameAge !== null && frameAge < 2000);
  health('voiceHealth', diagnostics.voice_ok && voice.running); health('rosHealth', diagnostics.rosbridge_connected); health('memoryHealth', diagnostics.memory_ok);
  $('captureStatus').textContent=`${voice.running ? '麦克风采集中' : '麦克风未采集'} · ${voice.input_device_info?.name || '--'}`;
  $('targetOverlay').textContent=attention.target_id && attention.target_id!=='none' ? `当前目标 ${attention.target_id}${attention.focus_status==='held_briefly'?' · 短暂跟踪保持':''}` : '当前无注意力目标';
  const focus=state.reflex_attention, spatial=focus?.spatial || {};
  $('reflexFocus').textContent=focus?.active ? `惊跳 / 定向注意力：${focus.source} → ${focus.target_id} · ${spatial.position_valid ? '3D '+JSON.stringify(spatial.position_xyz_m)+' m @ '+spatial.frame_id : spatial.azimuth_deg!=null ? '方位 '+number(spatial.azimuth_deg,1)+'°（无有效 3D 坐标）' : '方位未知'}；不自动放行对话` : '惊跳 / 定向注意力：无';
  if(focus?.active) $('targetOverlay').textContent='反射注意力 → '+focus.target_id;
  $('transportOverlay').textContent=`${diagnostics.attention_transport || 'offline'} · 融合 ${number(diagnostics.fusion_cycle_ms,1)} ms`;
  $('visionAge').textContent=frameAge===null ? '尚无画面数据' : `${Math.max(0,Math.round(frameAge))} ms 前`;
  $('visionBackend').textContent=vision.emotion_backend || '视觉原始值';
  const people=vision.people || [];
  retainScroll($('visionPeople'), people.map(p=>row(`${p.person_id} · ${p.role || 'unknown'}`,
    `情绪 ${p.emotion_label || 'unknown'}（${p.emotion_valid ? '有效' : '无效/待确认'}）· 分数 ${number(p.emotion_confidence)} · 原始 ${number(p.emotion_raw)} · valence ${number(p.emotion_valence)} · arousal ${number(p.emotion_arousal)}\n手势 ${p.gesture || 'none'} ${number(p.gesture_score)} · 注视 ${number(p.gaze_score)} · 唇动 ${p.lip_motion_valid === false ? '未知 / 待确认' : p.lip_motion ? '是' : '否'} · ${p.lip_backend || 'unknown'} · 开合 ${number(p.mouth_open_ratio)}`)).join('') || '<div class="empty">尚无人物观测，情绪值不作判断</div>');
  const gestures=(vision.gestures || []).filter(g=>!String(g.gesture || '').endsWith('_detected'));
  $('gestures').textContent='原始手势事件：'+(gestures.map(g=>`${g.gesture} ${number(g.score)}`).join(' / ') || '无');
  $('vad').textContent=!voice.running ? '未采集' : vad.active ? 'VAD 活跃' : 'VAD 静音'; $('vad').classList.toggle('active',Boolean(voice.running && vad.active));
  $('partialText').textContent=partial.text || '等待语音…'; $('partialTime').textContent=timeText(partial.emitted_ms);
  $('partialState').textContent=partial.is_final ? '已收尾' : Date.now()-(partial.emitted_ms || 0)>1500 ? '最近一次 partial' : 'partial';
  $('finalText').textContent=transcript.text || '--'; $('finalTime').textContent=timeText(transcript.emitted_ms);
  const speaker=transcript.speaker || voice.last_speaker || {};
  $('speaker').textContent=`最终句说话人：${speaker.speaker_id || 'unknown'} · 相似度 ${number(speaker.similarity)} · 声纹样本 ${speaker.pool_size || 1} · 候选间隔 ${number(speaker.match_margin)}`;
  $('asrBackend').textContent=voice.asr_backend || '--';
  const metrics=[['RMS dBFS',number(vad.rms_dbfs,1)],['SNR dB',number(vad.snr_db,1)],['VAD 概率',number(vad.probability)],['声源角度',track.azimuth_deg==null?'未知':number(track.azimuth_deg,1)+'°'],['噪声底 dBFS',number(vad.noise_floor_dbfs,1)],['自身回声',number(track.self_echo_probability)],['partial 延迟',partial.first_result_latency_ms==null?'--':Math.round(partial.first_result_latency_ms)+' ms'],['final 延迟',elapsed(transcript.emitted_ms,transcript.ended_ms)]];
  $('audioMetrics').innerHTML=metrics.map(([name,value])=>`<div class="metric"><span>${esc(name)}</span><strong>${esc(value)}</strong></div>`).join('');
  const observed=(state.utterances || []).find(t=>t.utterance_id===transcript.utterance_id), gate=observed?.attention;
  $('utteranceGate').textContent=!gate ? '最近一句：等待注意力标注' : gate.listen && gate.addressed_to_robot ? '最近一句：注意力放行，可进入回复' : '最近一句：未放行，只记录记忆';
  $('utteranceGate').parentElement.classList.toggle('allowed',Boolean(gate?.listen && gate?.addressed_to_robot));
  $('gateDetail').textContent=gate ? `listen=${Boolean(gate.listen)} · addressed=${Boolean(gate.addressed_to_robot)} · confidence=${number(gate.confidence)} · ${gate.identity_conflict ? "声纹/人脸冲突：优先声纹；不绑定冲突人脸 · " : ""}${(gate.reasons || []).join(' / ')}` : '--';
  const engagement=attention.engagement || {};
  const engagementReasons=(engagement.reasons || []).map(r=>reasonLabels[r] || r).join(' / ');
  $('engagement').textContent=engagement.unavailable_reason ? '交流意愿：当前算法没有这一层' :
    !engagement.state ? '交流意愿：--' :
    `交流意愿：${engagementLabels[engagement.state] || engagement.state}${engagement.person_id?' · '+engagement.person_id:''} · 置信 ${number(engagement.confidence)}${engagementReasons?' · '+engagementReasons:''}`;
  renderWorkspace();
  const distribution=attention.attention_distribution || {};
  $('attentionDistribution').innerHTML=Object.entries(distribution.sources || {}).map(([id,s])=>row(id,`${s.enabled?'已启用':'未启用'} · ${s.available?'有有效输入':'当前不可用'} · ${s.reason}`)).join('')
    +Object.entries(distribution.familiarity || {}).map(([id,f])=>row('熟悉度 '+id,`跨会话 ${f.prior_sessions} 次会话 / ${f.prior_turns} 条记录 · 权重 ${number(f.familiarity)}`)).join('')+Object.entries(distribution.candidates || {}).filter(([,v])=>v.valid).map(([id,v])=>row(id,`${v.modality} · 强度 ${number(v.a)} · 习惯化 ${number(v.h)} · 返回抑制 ${number(v.r)} · 焦点 ${distribution.focus?.[v.modality]===id?'是':'否'}`)).join('') || '<div class="empty">暂无有效候选</div>';
  $('liveAttention').textContent=`状态 ${interactionLabels[attention.interaction_phase] || attention.interaction_phase || '--'} · 实时注意力：${attention.target_id || 'none'} · listen=${Boolean(attention.listen)} · addressed=${Boolean(attention.addressed_to_robot)} · ${number(attention.confidence)}（与最终句门控分别显示）`;
  const hypotheses=state.person_hypotheses || [], selected=$('temporaryPerson').value;
  $('temporaryPerson').innerHTML=hypotheses.map(p=>`<option value="${esc(p.temporary_id)}">${esc(p.temporary_id)} → ${esc(p.resolved_person_id)}</option>`).join('');
  if(hypotheses.some(p=>p.temporary_id===selected)) $('temporaryPerson').value=selected;
  $('behavior').textContent=state.active_behaviors?.[0]?.kind || '身份关联稳定';
  if(!devicesSynced && devices.profiles && voice.input_device_info) {
    if(voice.profile) $('profile').value=voice.profile; refreshMicrophones(); $('microphone').value=String(voice.input_device_info.index); devicesSynced=true;
  }
  $('notice').textContent=[diagnostics.vision_error, diagnostics.voice_error, voice.error, voice.asr_error, voice.audio_scene?.acoustic?.direction_warning].filter(Boolean).join('\n');
  updateRegistration(state); renderBrain(); renderCocktail(state.cocktail || [], voice); renderPerformance(vision,voice,diagnostics); recordSignal(vad,voice); renderRaw();
}
const phases={thinking:'准备回复',synthesizing:'合成语音',speaking:'正在播放',completed:'已播完',interrupted:'已打断',muted:'TTS 已关闭',stopped:'已停止',error:'回复失败'};
const eventLabels={gaze_invitation_started:"转向邀请加入讨论",yield_requested:"等待短句边界让话",backchannel_observed:"收到附和，继续说话",proactive_deferred:"延后主动问候",user_turn_deferred:"惊跳后待回答",attention_changed:"注意力状态变化",startle_attention:'惊跳唤起注意力',reflex_started:'惊跳短叹',echo_rejected:'排除自身回声',first_token:'首个文本',phrase_ready:'分句就绪',heard:'收到最终转写',gate_accepted:'门控放行',gate_rejected:'门控拒绝',proactive_started:'主动搭话',reply_ready:'回复已生成',playback_started:'开始播放',playback_stopped:'停止播放',interrupted:'触发打断',turn_finished:'本轮结束',error:'错误'};
function renderBrain() {
  const brain=latestBrain, turn=brain?.current_turn, s2s=brain?.s2s || {}, playing=Boolean(latestState.voice?.playback?.speaking || s2s.speaking);
  health('brainHealth',Boolean(brain && !brainError));
  $('replyStatus').textContent=brainError ? 'Brain 未连接' : playing ? (s2s.generating ? '正在播放 · 文本生成中' : '正在播放') : phases[turn?.status] || (brain ? '等待输入' : '连接中');
  $('replyStatus').className='badge'+(playing?' speaking':brainError || turn?.status==='error'?' error':'');
  if(!ttsBusy) $('ttsEnabled').checked=Boolean(brain?.tts?.enabled); $('ttsEnabled').disabled=ttsBusy || !brain;
  $('ttsVoice').textContent=brain ? `${brain.tts?.voice || '--'} · ${brain.llm?.model || '--'}` : '--';
  $('interruptCount').textContent=`打断 ${s2s.interruptions || 0} 次`;
  $('replyInput').textContent=turn ? turn.gaze_invitation ? '转向机器人但未说话：结合刚才讨论接话' : turn.reflex ? (turn.reflex.source==='vision'?'视觉':'声音')+'惊跳：'+(turn.reflex.reason || turn.reflex.kind) : turn.proactive ? '主人出现触发的主动搭话' : turn.user_text || '--' : '--';
  $('replyPath').textContent=turn?.path || '--'; $('replyText').textContent=turn?.assistant_text || (s2s.thinking ? '正在生成回复…' : '等待门控后的输入…');
  $('timings').innerHTML=[['记忆检索',elapsed(turn?.memory_finished_ms,turn?.memory_started_ms)],['模型首字',elapsed(turn?.first_token_ms,turn?.llm_request_ms)],['首个文本',elapsed(turn?.first_token_ms,turn?.received_ms)],['首段准备',elapsed(turn?.first_chunk_ready_ms,turn?.received_ms)],['首段合成 / 启动',elapsed(turn?.playback_started_ms,turn?.first_chunk_ready_ms)],['句末 → 播放确认',elapsed(turn?.playback_started_ms,turn?.asr_ended_ms)]].map(([name,value])=>`<span>${esc(name)}<strong>${esc(value)}</strong></span>`).join('');
  const initiative=Object.entries(brain?.initiative || {}), waiting=initiative.find(([,v])=>v.locked);
  $('proactiveStatus').textContent=waiting ? `主动搭话：${waiting[0]} · ${waiting[1].already_interacted ? '本次已交谈，不重复问候' : '等待开口 / 安静 '+Math.ceil(waiting[1].wait_remaining_ms/1000)+' 秒'}` : '主动搭话：尚未锁定已确认的主人';
  $('playingChunk').textContent=playing && turn?.current_chunk ? '正在播报这段：'+turn.current_chunk : '';
  if(!ttsBusy){$('ttsRate').value=brain?.tts?.rate || 210;$('ttsRateValue').textContent=$('ttsRate').value;}
  $('replyMemoryTitle').textContent=`本次上下文：历史记忆 ${turn?.memories?.length || 0} 条 · 刚才对话 ${turn?.recent_conversation?.length || 0} 条`;
  $('replyMemories').innerHTML=(turn?.recent_conversation || []).map(m=>row('刚才 · '+(m.person_id || 'unknown'),m.text)).join('')+(turn?.memories || []).map(m=>row(m.person_id || 'unknown',m.text)).join('') || '<div class="empty">无参考记忆</div>';
  $('replyEvidence').textContent=turn ? `${turn.person_id} · 输入 ${timeText(turn.asr_emitted_ms || turn.received_ms)} · 播放 ${timeText(turn.playback_started_ms)} · 结束 ${timeText(turn.playback_ended_ms)}` : '--';
  $('replyError').textContent=brainError || s2s.error || brain?.error || '';
  $('memoryStatus').textContent=brain ? `记忆待写入 ${brain.memory?.pending || 0}${brain.memory?.error ? ' · '+brain.memory.error : ''}` : '记忆：--';
  retainScroll($('timeline'),(brain?.timeline || []).slice(0,60).map(event=>`<div class="timeline-row"><time>${timeText(event.stamp_ms)}</time><strong>${esc(eventLabels[event.kind] || event.kind)}</strong><span>${esc(event.text || interactionLabels[event.reason] || event.reason || event.error || phases[event.status] || event.status || '')}<small>${esc(event.person_id || '')} · ${esc(event.turn_id || '')}</small></span></div>`).join('') || '<div class="empty">等待回话事件（仅显示 Brain 当前进程记录）</div>');
}
const identityLabels={voice_enrollment:'声纹（已登记）',face_recognition:'人脸',voice_cluster:'声纹（自动登记）',av_association:'视听关联',unresolved:'未确定',answer_continuity_unverified:'按对话推断'};
function voiceprint(digest){
  if(!digest || !digest.length) return '<div class="voiceprint silent">暂无向量</div>';
  const peak=Math.max(...digest.map(v=>Math.abs(Number(v)||0)),1e-6);
  return `<div class="voiceprint" title="${esc(digest.slice(0,8).map(v=>Number(v).toFixed(3)).join(' '))}…（192 维均值池化到 ${digest.length} 段）">`+
    digest.map(v=>`<i style="height:${Math.max(2,Math.round(Math.abs(Number(v)||0)/peak*22))}px;opacity:${Number(v)<0?.45:1}"></i>`).join('')+'</div>';
}
function renderCocktail(rows, voice){
  const registry=voice.voice_registry || {}, decision=voice.speaker_decision || {};
  $('cocktailStatus').textContent=rows.length
    ? `${rows.length} 个声音 · 已登记 ${Object.values(registry).filter(v=>v.role==='owner'||v.role==='known').length} 人 · 自动登记 ${Object.values(registry).filter(v=>v.role==='stranger').length} 个`
    + (decision.score!==undefined ? ` · 最近判定 ${esc(decision.speaker_id||'unknown')} 余弦 ${number(decision.score)} z ${number(decision.z,1)}（${esc(decision.path||'')}）` : '')
    : '等待语音';
  retainScroll($('cocktailRows'), rows.map(person=>{
    const entry=registry[person.person_id]||{};
    const pool=entry.pool_size!==undefined ? `声纹池 ${entry.pool_size} 段 / ${entry.anchor_groups||1} 组` : '未登记声纹';
    const said=person.utterances.length ? person.utterances.map(u=>
      `<span class="${u.addressed?'answered':u.held?'held':''}">${timeText(u.stamp_ms).slice(0,8)} ${esc(u.text)}${u.addressed?' · 对我说':u.held?' · 先记下':''}</span>`).join('')
      : '<span>这段时间只听到声音，没有成句</span>';
    return `<div class="cocktail-row${person.attended?' attended':''}">
      ${voiceprint(person.digest)}
      <div class="cocktail-person"><strong>${esc(person.display_name)}</strong>
        <span>${esc(identityLabels[person.identity_source]||person.identity_source||'仅声纹聚类')} · 置信 ${number(person.identity_confidence)}</span>
        <span>${esc(pool)}${person.reference_available?' · 有参考音':''}${person.attended?' · 正在听他':''}</span></div>
      <div class="cocktail-said">${said}</div></div>`;
  }).join('') || '<div class="empty">房间里还没有人说话</div>');
}
function renderPerformance(vision,voice,diagnostics){
  const music=voice.audio_scene?.music || {}, entries=[];
  for(const [name,value] of Object.entries(vision.model_timings || {}))entries.push([name,`${number(value.ms,1)} ms`,`${Math.max(0,Date.now()-value.stamp_ms)} ms 前`]);
  entries.push(['视觉帧',`${number(vision.frame_processing_ms,1)} ms`,`${number(vision.inference_fps,1)} fps 实测`]);
  entries.push(['ASR 解码',`${number(voice.asr_decode_ms,1)} ms`,'100 ms 音频批次']);
  const bio=voice.audio_scene?.acoustic?.bio || {},vap=voice.audio_scene?.turn_prediction || {};
  entries.push(['VAP 轮次预测',`${number(vap.inference_ms,1)} ms`,vap.ready ? `${vap.device || 'cpu'} · ${vap.valid?'有效':'等待新音频'}` : '加载中 / 不可用']);
  entries.push(['F0',bio.f0_hz ? `${number(bio.f0_hz,1)} Hz`:'--',`有声置信 ${number(bio.voicing_confidence)}`]);
  entries.push(['声学显著度',number(bio.salience_score),`谱变化 ${number(bio.spectral_flux)}`]);
  entries.push(['声纹',`${number(voice.last_transcript?.speaker_embedding_ms,1)} ms`,'最近一次最终句']);
  for(const [name,ms] of Object.entries(music.model_timings_ms || {}))entries.push([name,`${number(ms,1)} ms`,`${number((music.window_ms || 0)/1000,1)} s 音频窗口`]);
  entries.push(['注意力',`${number(diagnostics.fusion_cycle_ms,1)} ms`,latestBrain?.input_transport || '连接中']);
  entries.push(['LLM 首文本',elapsed(latestBrain?.current_turn?.first_token_ms,latestBrain?.current_turn?.received_ms),'包含记忆检索']);
  $('performance').innerHTML=entries.map(([name,value,detail])=>`<div class="metric"><span>${esc(name)}</span><strong>${esc(value)}</strong><span>${esc(detail)}</span></div>`).join('');
}
function recordSignal(vad,voice) {
  const stamp=Date.now(); signalHistory.push({stamp,rms:voice.running ? vad.rms_dbfs : null,vad:Boolean(voice.running && vad.active),tts:Boolean(voice.playback?.speaking || latestBrain?.s2s?.speaking)});
  signalHistory=signalHistory.filter(p=>stamp-p.stamp<=30000);
  const canvas=$('signalPlot'), ctx=canvas.getContext('2d'), width=canvas.width,height=canvas.height;
  ctx.clearRect(0,0,width,height); ctx.fillStyle='#f5f8f6';ctx.fillRect(0,0,width,height);
  ctx.strokeStyle='#dae6de';ctx.fillStyle='#829387';ctx.font='11px sans-serif';
  for(const db of [-80,-40,0]){const y=height-10-(db+80)/80*(height-28);ctx.beginPath();ctx.moveTo(35,y);ctx.lineTo(width,y);ctx.stroke();ctx.fillText(String(db),3,y+3);}
  const x=p=>35+(p.stamp-(stamp-30000))/30000*(width-35);
  for(let i=1;i<signalHistory.length;i++){const p=signalHistory[i],previous=signalHistory[i-1];if(p.vad){ctx.fillStyle='#d6eddf';ctx.fillRect(x(previous),18,Math.max(1,x(p)-x(previous)),height-28);}if(p.tts){ctx.fillStyle='#e9ad51';ctx.fillRect(x(previous),2,Math.max(1,x(p)-x(previous)),8);}}
  ctx.strokeStyle='#25704d';ctx.lineWidth=1.7;ctx.beginPath();let begun=false;
  for(const p of signalHistory){if(p.rms==null){begun=false;continue;}const y=height-10-(Math.max(-80,Math.min(0,p.rms))+80)/80*(height-28);if(!begun)ctx.moveTo(x(p),y);else ctx.lineTo(x(p),y);begun=true;}ctx.stroke();
}
function renderRaw(){const v=latestState.voice || {}; $('rawSnapshot').textContent=JSON.stringify({vision:latestState.vision,voice:{running:v.running,last_vad:v.last_vad,last_track:v.last_track,last_transcript:v.last_transcript,last_streaming_transcript:v.last_streaming_transcript,playback:v.playback},attention:latestState.attention,interruption_candidate:latestState.interruption_candidate,reflex_attention:latestState.reflex_attention,crossmodal_events:latestState.crossmodal_events,interaction_transitions:latestState.interaction_transitions,brain_turn:latestBrain?.current_turn},null,2);}
async function pollPerception(){try{render(await fetchJSON('/api/state'));}catch(error){latestState={};$('vad').textContent='连接中断';$('captureStatus').textContent='采集状态未知';renderBrain();$('notice').textContent='感知服务连接失败：'+error.message;for(const id of ['visionHealth','voiceHealth','rosHealth'])health(id,false);}finally{setTimeout(pollPerception,200);}}
async function pollBrain(){try{latestBrain=await fetchJSON('/api/brain-state');brainError='';}catch(error){brainError=error.message;latestBrain=null;}finally{renderBrain();setTimeout(pollBrain,250);}}
async function command(path,target='all'){if(path==='/api/start' && target!=='camera' && !$('microphone').value)throw new Error('未检测到匹配的麦克风，请检查USB连接；不会回退到内置麦克风。');const [source,index]=($('camera').value || 'opencv:0').split(':',2);return fetchJSON(path,{method:'POST',signal:AbortSignal.timeout(30000),headers:{'content-type':'application/json'},body:JSON.stringify({target,camera_index:index,camera_source_type:source,profile:$('profile').value || 'mac_builtin',device:$('microphone').value || 'default'})});}
for(const name of ['start','stop']) $(name).onclick=async()=>{try{await command('/api/'+name);devicesSynced=false;}catch(error){$('notice').textContent=error.message;}};
for(const [id,target] of [['cameraApply','camera'],['microphoneApply','microphone']]) $(id).onclick=async()=>{ $(id).disabled=true; try{await command('/api/start',target);devicesSynced=false;}catch(error){$('notice').textContent=error.message;}finally{$(id).disabled=false;}};
$('registerPerson').onclick=async()=>{try{await fetchJSON('/api/register-person',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({temporary_id:$('temporaryPerson').value,persistent_id:$('persistentPerson').value})});}catch(error){$('notice').textContent=error.message;}};
$('ttsEnabled').onchange=async()=>{ttsBusy=true;try{await fetchJSON('/api/brain-tts',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({enabled:$('ttsEnabled').checked,voice:latestBrain?.tts?.voice || 'Tingting',rate:Number($('ttsRate').value)})});}catch(error){$('replyError').textContent=error.message;}finally{ttsBusy=false;}};
$('ttsRate').oninput=()=>{$('ttsRateValue').textContent=$('ttsRate').value;};
$('ttsRate').onchange=()=>{$('ttsEnabled').onchange();};
function renderAttentionConfig(config){
  attentionConfig=config;
  $('attentionSources').innerHTML=(config.sources || []).map(source=>
    `<label class="row"><span><input type="checkbox" data-source="${esc(source.id)}"${source.enabled?' checked':''}> <strong>${esc(source.display_name)}</strong></span><span>${esc(source.description)}</span></label>`).join('');
  const algorithms=config.algorithm?.algorithms || [];
  $('attentionAlgorithm').innerHTML=algorithms.map(a=>`<option value="${esc(a.id)}"${a.active?' selected':''}>${esc(a.display_name)}${a.kind==='shared_library'?'（.so）':'（内置）'}</option>`).join('') || '<option value="">未发现可用算法</option>';
  const active=algorithms.find(a=>a.active);
  const errors=Object.entries(config.algorithm?.load_errors || {}).map(([name,message])=>`${name}: ${message}`);
  $('attentionAlgorithmDetail').textContent=[active?.summary,
    active?`需要来源：${(active.required_sources||[]).join('、')||'无'}；可选：${(active.optional_sources||[]).join('、')||'无'}`:'',
    active?.calibrated?'':'未标定：不能把论文或单元测试结果当作现场准确率。',
    active?.limits?.length?'限制：'+active.limits.join('；'):'',
    errors.length?'加载失败：'+errors.join('；'):''].filter(Boolean).join(' ｜ ');
  $('attentionAddressee').innerHTML=(config.addressee || []).map(term=>
    `<label class="row"><span><strong>${esc(term.label)}</strong></span>`
    +`<span><input type="number" step="0.05" min="0" max="2" data-addressee="${esc(term.id)}" value="${term.weight}"></span></label>`
  ).join('') || '<div class="empty">无</div>';
  const last=config.addressee_last;
  const verdict={answer:'立刻回复',hold:'记下来但先不回',ignore:'当作旁人说话'}[last?.decision]||(last?.granted?'放行':'未放行');
  $('addresseeLast').textContent=last
    ? `最近一句「${last.text}」→ ${verdict} · 得分 ${number(last.score)}（回复线 ${number(last.answer_threshold)} / 记住线 ${number(last.threshold)}）`
      +` · 命中 ${Object.entries(last.terms||{}).map(([k,v])=>`${k} ${number(v)}`).join('、')||'无'}`
      +`${last.person_id?' · 归属 '+esc(last.person_id):''}`
    : '最近一句的判定：还没有完整句子';
  $('attentionConfigStatus').textContent=(config.messages || []).join('；') || config.note || '';
}
let workspace=null;
async function pollWorkspace(){
  for(;;){
    try{workspace=await fetchJSON('/api/workspace');}catch(error){workspace={error:error.message};}
    await new Promise(r=>setTimeout(r,500));
  }
}
function renderWorkspace(){
  if(!workspace){$('workspaceSummary').textContent='等待第一轮广播…';return;}
  if(workspace.error){$('workspaceSummary').textContent='读取失败：'+workspace.error;return;}
  if(!workspace.cycle){$('workspaceSummary').textContent=workspace.reason || '尚未广播';return;}
  const cap=workspace.capacity || {}, st=workspace.status || {}, prov=workspace.provenance || {};
  const memory=prov.memory_events || {};
  $('workspaceSummary').textContent=`周期 ${workspace.cycle} · 容量 ${cap.admitted ?? '--'}/${cap.limit ?? '--'}`
    +` · 被压制 ${cap.suppressed ?? 0} · 订阅者 ${st.subscribers ?? 0} · 丢帧 ${st.dropped_cycles ?? 0}`
    +` · 记忆候选 ${memory.enabled===false?'已关闭':(memory.reason || '--')}`;
  $('workspaceCoalition').innerHTML=(workspace.coalition || []).map(item=>row(
    `${item.admitted?'✓':'✗'} ${item.candidate_id}`,
    `${item.kind}${item.person_id?' · '+item.person_id:''} · 份额 ${number(item.share)} · 强度 ${number(item.activation)}`
    +`${item.urgency?' · 紧急 '+number(item.urgency):''} · ${item.admission}`
    +`${item.summary?' · '+item.summary:''}${(item.reasons||[]).length?' · '+item.reasons.map(r=>reasonLabels[r]||r).join('/'):''}`
  )).join('') || '<div class="empty">本轮没有候选</div>';
}
async function loadAttentionConfig(){
  try{renderAttentionConfig(await fetchJSON('/api/attention-config'));}
  catch(error){$('attentionConfigStatus').textContent='读取注意力配置失败：'+error.message;}
}
$('attentionApply').onclick=async()=>{
  $('attentionApply').disabled=true;
  const sources=[...$('attentionSources').querySelectorAll('input[data-source]')].filter(input=>input.checked).map(input=>input.dataset.source);
  const addressee_weights={};
  for(const input of $('attentionAddressee').querySelectorAll('input[data-addressee]')){
    const value=Number(input.value);
    if(Number.isFinite(value))addressee_weights[input.dataset.addressee]=value;
  }
  try{
    renderAttentionConfig(await fetchJSON('/api/attention-config',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify({sources,algorithm:$('attentionAlgorithm').value,addressee_weights})}));
  }catch(error){$('attentionConfigStatus').textContent='应用失败：'+error.message;}
  finally{$('attentionApply').disabled=false;}
};
configure().catch(error=>{$('notice').textContent=error.message;});loadAttentionConfig();pollWorkspace();pollPerception();pollBrain();

let historyOffset=0, historySnapshot=null, historyTotal=0, historyBusy=false, historySequence=0;
async function loadHistory(reset=false) {
  if(reset){historyOffset=0;historySnapshot=null;}
  const session=$('historySession').value.trim();if(!session)return;
  const sequence=++historySequence;historyBusy=true;
  const params=new URLSearchParams({session,q:$('historyQuery').value.trim(),person:$('historyPerson').value.trim(),kind:$('historyKind').value,offset:String(historyOffset)});
  if(historySnapshot!==null)params.set('as_of_ms',String(historySnapshot));
  try {
    const data=await fetchJSON('/api/memory-search?'+params);
    if(sequence!==historySequence)return;
    historyTotal=data.total ?? (data.hits || []).length;historySnapshot=data.as_of_ms ?? null;
    const records=(data.hits || []).map(h=>h.record);
    $('historyStatus').textContent=`共 ${historyTotal} 条 · ${historyTotal ? historyOffset+1 : 0}–${Math.min(historyOffset+records.length,historyTotal)} · ${new Date().toLocaleTimeString('zh-CN')} 更新`;
    retainScroll($('historyRecords'),records.map(record=>{
      const details=record.structured || {}, input=record.kind==='heard_utterance', gate=details.attention || {};
      const person=record.entity?.person_id || 'unknown';
      const addressed=gate.addressed_to_robot===true;
      const label=details.addressee==='group'?'向群体提问（含机器人）':String(details.addressee || '').startsWith('person:')?'对 '+details.addressee.slice(7)+' 说':record.kind==='robot_echo'?'自身回声（未作为用户记忆 / 输入）':input ? addressed?'对机器人说':'非机器人 / 听话人未确定' : details.interrupted?'机器人回复（被打断）':details.playback_completed?'机器人回复（已播完）':'机器人回复（未完整播放）';
      return `<article class="memory-record"><div class="memory-meta"><time>${esc(new Date(record.observed_at_ms).toLocaleString('zh-CN',{hour12:false}))}</time><strong>${esc(input ? person : '机器人 → '+person)}</strong><span class="memory-tag ${addressed?'addressed':''}">${esc(label)}</span></div><p>${esc(record.text || '')}</p><button class="trace-button secondary" data-turn="${esc(details.timing?.turn_id || details.utterance_id || record.trace_id || '')}">回溯这轮</button><details><summary>查看该条证据 / 原始记录</summary><pre>${esc(JSON.stringify(record,null,2))}</pre></details></article>`;
    }).join('') || '<div class="empty">该会话或筛选条件下还没有记录</div>');
    $('historyPrev').disabled=historyOffset===0;$('historyNext').disabled=historyOffset+25>=historyTotal;
  } catch(error) {if(sequence===historySequence)$('historyStatus').textContent='记忆查询失败：'+error.message;}
  finally {if(sequence===historySequence)historyBusy=false;}
}
$('historyForm').onsubmit=event=>{event.preventDefault();loadHistory(true);};
$('historyPrev').onclick=()=>{historyOffset=Math.max(0,historyOffset-25);loadHistory();};
$('historyNext').onclick=()=>{historyOffset+=25;loadHistory();};
$('historyForm').oninput=()=>{$('historyAuto').checked=false;};
$('historyRecords').addEventListener('toggle',event=>{if(event.target.open)$('historyAuto').checked=false;},true);
setInterval(()=>{if(!historyBusy && historyOffset===0 && $('historyAuto').checked)loadHistory(true);},2000);

$('historyRecords').addEventListener('click',async event=>{
  const button=event.target.closest('[data-turn]');if(!button)return;
  $('historyAuto').checked=false;$('historicalTrace').open=true;
  try{const data=await fetchJSON('/api/memory-trace?'+new URLSearchParams({turn:button.dataset.turn,session:$('historySession').value}));
    $('historyTraceRecords').innerHTML=(data.hits || []).map(h=>h.record.structured).sort((a,b)=>a.stamp_ms-b.stamp_ms || a.id-b.id).map(e=>`<div class="timeline-row"><time>${timeText(e.stamp_ms)}</time><strong>${esc(eventLabels[e.kind] || e.kind)}</strong><span>${esc(e.text || e.reason || e.error || e.status || '')}</span></div>`).join('') || '<div class="empty">该轮没有事件链（旧版本记录仅保留原始内容）</div>';
  }catch(error){$('historyTraceRecords').textContent=error.message;}
});


let registration=null;
async function registrationTTS(enabled){
  return fetchJSON('/api/brain-tts',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({enabled,voice:latestBrain?.tts?.voice || 'Tingting',rate:Number($('ttsRate').value)})});
}
async function endRegistration(message){
  const session=registration;registration=null;
  $('registrationStart').disabled=false;$('registrationCancel').disabled=true;
  $('registrationStatus').textContent=message;
  if(session?.restoreTTS)try{await registrationTTS(true);}catch(error){$('registrationStatus').textContent+='；请在TTS区重新开启声音：'+error.message;}
}
$('registrationStart').onclick=async()=>{
  const name=$('registrationName').value.trim();
  if(!name){$('registrationStatus').textContent='先填一个称呼，方便以后认出你。';return;}
  if(!latestState.voice?.running){$('registrationStatus').textContent='请先在上方启动麦克风和摄像头采集。';return;}
  if(!latestBrain){$('registrationStatus').textContent='请等待Brain连接，以便注册时暂停机器人说话。';return;}
  registration={name,role:$('registrationRole').value,started:Date.now(),last:'',busy:true,restoreTTS:Boolean(latestBrain.tts?.enabled)};
  $('registrationStart').disabled=true;$('registrationCancel').disabled=false;
  try{await registrationTTS(false);if(registration){registration.busy=false;registration.started=Date.now();const active=registration;setTimeout(()=>{if(registration===active && !active.busy)endRegistration('等待超时，准备好后可重新开始。');},90000);$('registrationStatus').textContent='正在听，请面对镜头自然说一段话……';}}
  catch(error){await endRegistration('暂时无法开始：'+error.message);}
};
$('registrationCancel').onclick=()=>endRegistration('已取消注册。');
async function updateRegistration(state){
  const session=registration;if(!session || session.busy)return;
  if(Date.now()-session.started>90000){await endRegistration('等待超时，准备好后可重新开始。');return;}
  const vision=state.vision?.state || state.vision || {}, voice=state.voice || {}, t=voice.last_transcript || {};
  const faces=(vision.people || []).filter(p=>p.face_visible || Number(p.face_confidence)>=.75);
  if(faces.length!==1){$('registrationStatus').textContent='请只让注册者一人面对镜头。';return;}
  if(!t.utterance_id || t.started_ms<session.started || t.utterance_id===session.last)return;
  session.last=t.utterance_id;
  if(t.ended_ms-t.started_ms<2000 || (t.text || '').trim().length<6){$('registrationStatus').textContent='听到了，再自然说一句稍长的话（至少2秒）。';return;}
  session.busy=true;$('registrationCancel').disabled=true;$('registrationStatus').textContent='听到：“'+t.text+'”。正在保存人脸和声纹……';
  try{
    const result=await fetchJSON('/api/guided-registration',{method:'POST',signal:AbortSignal.timeout(15000),headers:{'content-type':'application/json'},body:JSON.stringify({name:session.name,role:session.role,utterance_id:t.utterance_id})});
    if(registration!==session)return;
    if(result.success){await endRegistration(result.message+(result.already_registered?'':'。下次直接与机器人说话即可。'));}
    else{session.busy=false;$('registrationCancel').disabled=false;$('registrationStatus').textContent=result.message+'；请再说一句。';}
  }catch(error){if(registration===session){session.busy=false;$('registrationCancel').disabled=false;$('registrationStatus').textContent=error.message+'；请再说一句。';}}
}
