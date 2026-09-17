const fs=require('fs'), vm=require('vm'), assert=require('assert');
const path=require('path');
const html=fs.readFileSync(path.join(__dirname,'../web/live_index.html'),'utf8');
const nodes={};
class Element {
  constructor(id){this.id=id;this.value='';this.checked=false;this.disabled=false;this.textContent='';this.scrollTop=0;this.style={};this.options=[];this.selectedOptions=[{dataset:{channels:'1'}}];this.classList={toggle(){}};this.parentElement={classList:{toggle(){}}};this.width=1200;this.height=100;}
  set innerHTML(text){this.html=text;this.options=[];} get innerHTML(){return this.html || '';}
  addEventListener(){} getContext(){return new Proxy({}, {get:()=>()=>{}});}
}
for(const [,id] of html.matchAll(/id="([^"]+)"/g)){assert(!nodes[id],'duplicate id '+id);nodes[id]=new Element(id);}
const fixture={stamp_ms:Date.now(),vision:{state:{stamp:Date.now()/1000,emotion_backend:'ferplus',people:[{person_id:'person1',role:'owner',emotion_label:'happy',emotion_valid:false,gesture:'none'}]}},voice:{running:true,asr_backend:'sherpa',last_vad:{active:false,rms_dbfs:-55},last_track:{voice_activity:true},last_transcript:{utterance_id:'u',text:'<img src=x onerror=alert(1)>',speaker:{speaker_id:'person1'}}},utterances:[{utterance_id:'u',attention:{listen:false,addressed_to_robot:false}}],diagnostics:{vision_ok:true,voice_ok:true,rosbridge_connected:true},
cocktail:[{key:'天行',display_name:'天行',person_id:'天行',registered:true,identity_source:'face_recognition',identity_confidence:.82,digest:[.1,-.2,.3],attended:true,reference_available:true,last_heard_ms:Date.now(),utterances:[{text:'<b>今天天气怎么样</b>',stamp_ms:Date.now(),addressed:true,held:false}]},
 {key:'stranger_1',display_name:'陌生人1',person_id:'stranger_1',registered:true,identity_source:'voice_cluster',identity_confidence:.4,digest:[],attended:false,last_heard_ms:Date.now(),utterances:[{text:'下午几点走',stamp_ms:Date.now(),addressed:false,held:true}]}]};
const brain={tts:{enabled:true,rate:210},s2s:{speaking:true},current_turn:{turn_id:'old',user_text:'真正触发回复的句子',assistant_text:'第一句。',current_chunk:'第一句。',status:'speaking',memories:[]},timeline:[{id:1,kind:"playback_started",stamp_ms:Date.now(),turn_id:"old",text:"第一句。"}]};
let urls=[];
const context={console,URLSearchParams,AbortSignal,Date,JSON,Number,String,Boolean,Math,document:{getElementById(id){assert(nodes[id],'missing DOM element '+id);return nodes[id];}},setTimeout(){},setInterval(){},fetch:async(url)=>{
  urls.push(url);
  let data=url==='/api/config'?{session_id:'test',vision_stream_url:'http://localhost/stream',vision_url:'http://localhost:8080',voice_url:'http://localhost:8090',brain_url:'http://localhost:8094'}:url==='/api/devices'?{profiles:[],cameras:[],microphones:[]}:url==='/api/state'?fixture:url==='/api/brain-state'?brain:{hits:[{record:{kind:'heard_utterance',text:'<img src=x onerror=alert(1)>',observed_at_ms:1000,entity:{person_id:'person1'},structured:{attention:{addressed_to_robot:false}}}}],total:50,as_of_ms:12345};
  return {ok:true,json:async()=>data};
}};
vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(__dirname,'../web/live_app.js'),'utf8'),context);
(async()=>{
  await new Promise(setImmediate);
  assert.equal(nodes.vad.textContent,'VAD 静音');
  assert(nodes.timeline.innerHTML.includes('开始播放'));
  assert(nodes.timeline.innerHTML.includes('第一句。'));
  assert(nodes.cameraApply && nodes.microphoneApply);
  assert.equal(nodes.replyInput.textContent,'真正触发回复的句子');
  assert(nodes.visionPeople.innerHTML.includes('无效/待确认'));
  // 鸡尾酒会一行一个声音：声纹条、名字、以及这个人说的话
  assert(nodes.cocktailRows.innerHTML.includes('天行')&&nodes.cocktailRows.innerHTML.includes('陌生人1'));
  assert(nodes.cocktailRows.innerHTML.includes('voiceprint')&&nodes.cocktailRows.innerHTML.includes('暂无向量'));
  assert(nodes.cocktailRows.innerHTML.includes('对我说')&&nodes.cocktailRows.innerHTML.includes('先记下'));
  assert(nodes.cocktailRows.innerHTML.includes('&lt;b&gt;')&&!nodes.cocktailRows.innerHTML.includes('<b>'));
  assert(nodes.cocktailStatus.textContent.includes('2 个声音'));
  assert(nodes.historyRecords.innerHTML.includes('&lt;img'));
  assert(!nodes.historyRecords.innerHTML.includes('<img'));
  vm.runInContext('historyOffset=25; loadHistory();',context);
  await new Promise(setImmediate);
  assert(urls.some(url=>url.includes('offset=25')&&url.includes('as_of_ms=12345')));
  console.log('dashboard DOM, raw VAD, turn binding, escaping and pagination checks passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
