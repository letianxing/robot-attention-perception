"""Load swappable attention algorithms through the C ABI.

The host owns perception, identity and permission. A plugin only ranks evidence
it was handed and reports a belief; a missing or broken library is reported as
an error, never silently replaced by a neutral result.
"""
import ctypes,json,os,threading
from pathlib import Path

ABI_VERSION=1
ID_MAX=64
MODALITY_MAX=16
STATE_MAX=32

REASON_BITS=[(1<<0,'gaze_engaged'),(1<<1,'lip_audio_sync'),(1<<2,'directed_call'),(1<<3,'memory_participant'),
             (1<<4,'expected_answer'),(1<<5,'topic_continuation'),(1<<6,'addresses_other'),(1<<7,'backchannel'),
             (1<<8,'silent_invitation'),(1<<9,'dialogue_target'),(1<<10,'internal_state'),(1<<11,'sound_novelty'),
             (1<<12,'familiar_across_sessions'),(1<<13,'gesture_invite'),(1<<14,'gesture_reject'),(1<<15,'open_exchange'),
             (1<<16,'sustained_gaze')]

EXTRA_MEMORY_FAMILIARITY='memory_familiarity'
EXTRA_AZIMUTH_DEG='azimuth_deg'
EXTRA_AV_SYNC_ONSET='av_sync_onset'
EXTRA_GESTURE_INVITE='gesture_invite'
EXTRA_GESTURE_REJECT='gesture_reject'
EXTRA_RETURNED='returned_after_absence'
EXTRA_OPEN_EXCHANGE='memory_open_exchange'


def reasons_from_mask(mask):
    return [name for bit,name in REASON_BITS if mask & bit]


class KeyValue(ctypes.Structure):
    _fields_=[('key',ctypes.c_char*ID_MAX),('value',ctypes.c_double)]


class CandidateIn(ctypes.Structure):
    _fields_=[('candidate_id',ctypes.c_char*ID_MAX),('modality',ctypes.c_char*MODALITY_MAX),('person_id',ctypes.c_char*ID_MAX),
              ('stamp_ms',ctypes.c_int64),('ttl_ms',ctypes.c_int32),
              ('salience',ctypes.c_double),('goal',ctypes.c_double),('surprise',ctypes.c_double),
              ('observability',ctypes.c_double),('uncertainty',ctypes.c_double),('importance',ctypes.c_double),
              ('gaze',ctypes.c_double),('body_facing',ctypes.c_double),('lip_sync',ctypes.c_double),
              ('voice_activity',ctypes.c_double),('identity_confidence',ctypes.c_double),
              ('memory_participant',ctypes.c_double),('memory_recency',ctypes.c_double),
              ('memory_expected_answer',ctypes.c_double),('memory_dialogue_target',ctypes.c_double),
              ('lang_directed_call',ctypes.c_double),('lang_question',ctypes.c_double),
              ('lang_answer_continuation',ctypes.c_double),('lang_topic_continuation',ctypes.c_double),
              ('lang_addresses_other',ctypes.c_double),('lang_backchannel',ctypes.c_double),
              ('motivation',ctypes.c_double),
              ('link_id',ctypes.c_char*ID_MAX),('link_confidence',ctypes.c_double),
              ('change_id',ctypes.c_char*ID_MAX),('event_id',ctypes.c_char*ID_MAX),
              ('extra_count',ctypes.c_int32),('extra',ctypes.POINTER(KeyValue))]


class FrameIn(ctypes.Structure):
    _fields_=[('abi_version',ctypes.c_int32),('stamp_ms',ctypes.c_int64),('arousal_gain',ctypes.c_double),
              ('source_audio_visual',ctypes.c_int32),('source_memory_context',ctypes.c_int32),
              ('source_linguistic_context',ctypes.c_int32),('source_internal_state',ctypes.c_int32),
              ('robot_speaking',ctypes.c_int32),('reflex_active',ctypes.c_int32),
              ('candidate_count',ctypes.c_int32),('candidates',ctypes.POINTER(CandidateIn)),
              ('extra_count',ctypes.c_int32),('extra',ctypes.POINTER(KeyValue))]


class CandidateOut(ctypes.Structure):
    _fields_=[('candidate_id',ctypes.c_char*ID_MAX),('modality',ctypes.c_char*MODALITY_MAX),
              ('activation',ctypes.c_double),('habituation',ctypes.c_double),('inhibition',ctypes.c_double),
              ('change',ctypes.c_double),('normalized',ctypes.c_double),('input',ctypes.c_double),
              ('goal',ctypes.c_double),('uncertainty',ctypes.c_double),('learning',ctypes.c_double),
              ('engagement',ctypes.c_double),('addressee_robot',ctypes.c_double),
              ('valid',ctypes.c_int32),('is_focus',ctypes.c_int32),('reason_mask',ctypes.c_uint32)]


class FrameOut(ctypes.Structure):
    _fields_=[('abi_version',ctypes.c_int32),('stamp_ms',ctypes.c_int64),
              ('candidate_count',ctypes.c_int32),('candidates',ctypes.POINTER(CandidateOut)),
              ('visual_focus',ctypes.c_char*ID_MAX),('audio_focus',ctypes.c_char*ID_MAX),
              ('engagement_state',ctypes.c_char*STATE_MAX),('engaged_person',ctypes.c_char*ID_MAX),
              ('engagement_confidence',ctypes.c_double),('engagement_reason_mask',ctypes.c_uint32),
              ('residual_visual',ctypes.c_double),('residual_audio',ctypes.c_double),
              ('transition_sequence',ctypes.c_int32)]


_STRUCT_ORDER=(FrameIn,CandidateIn,FrameOut,CandidateOut,KeyValue)


def _text(value,limit):
    return str(value or '').encode('utf-8')[:limit-1]


class SharedLibraryAlgorithm:
    """One loaded .so plus one algorithm instance. Not internally reentrant."""

    kind='shared_library'

    def __init__(self,path):
        self.path=str(path)
        self.library=ctypes.CDLL(self.path)
        self.library.attention_plugin_abi_version.restype=ctypes.c_int32
        abi=int(self.library.attention_plugin_abi_version())
        if abi!=ABI_VERSION:
            raise RuntimeError(f'{Path(self.path).name} reports ABI {abi}, host speaks {ABI_VERSION}')
        self.library.attention_plugin_struct_size.restype=ctypes.c_int32
        self.library.attention_plugin_struct_size.argtypes=[ctypes.c_int32]
        for index,structure in enumerate(_STRUCT_ORDER):
            plugin_size=int(self.library.attention_plugin_struct_size(index))
            if plugin_size!=ctypes.sizeof(structure):
                raise RuntimeError(f'{structure.__name__} is {ctypes.sizeof(structure)} bytes here and '
                                   f'{plugin_size} in {Path(self.path).name}')
        self.library.attention_plugin_manifest.restype=ctypes.c_char_p
        self.manifest=json.loads(self.library.attention_plugin_manifest().decode('utf-8'))
        self.id=str(self.manifest.get('id') or Path(self.path).stem)
        self.library.attention_plugin_create.restype=ctypes.c_void_p
        self.library.attention_plugin_destroy.argtypes=[ctypes.c_void_p]
        self.library.attention_plugin_update.argtypes=[ctypes.c_void_p,ctypes.POINTER(FrameIn)]
        self.library.attention_plugin_update.restype=ctypes.POINTER(FrameOut)
        self.handle=ctypes.c_void_p(self.library.attention_plugin_create())
        if not self.handle:
            raise RuntimeError(f'{self.id} refused to create an instance')

    def close(self):
        if self.handle:
            self.library.attention_plugin_destroy(self.handle)
            self.handle=None

    def reset(self):
        """Start from nothing when this algorithm is switched in.

        An instance that last ran minutes ago still holds that much habituation,
        return inhibition and engagement. Handing it the current frame would
        compare now against a situation that no longer exists, so switching
        gives you the algorithm, not its memory of an earlier session.
        """
        self.close()
        self.handle=ctypes.c_void_p(self.library.attention_plugin_create())
        if not self.handle:
            raise RuntimeError(f'{self.id} refused to create an instance')

    def update(self,stamp_ms,candidates,sources,arousal_gain=1.,robot_speaking=False,reflex_active=False):
        if not self.handle:
            raise RuntimeError(f'{self.id} is closed')
        array=(CandidateIn*len(candidates))()
        # The plugin only borrows these buffers, so they must outlive the call.
        extras=[]
        for slot,item in zip(array,candidates):
            extra=item.get('extra') or {}
            if extra:
                buffer=(KeyValue*len(extra))()
                for cell,(key,value) in zip(buffer,extra.items()):
                    cell.key=_text(key,ID_MAX);cell.value=float(value)
                extras.append(buffer)
                slot.extra=buffer;slot.extra_count=len(extra)
            slot.candidate_id=_text(item.get('candidate_id'),ID_MAX)
            slot.modality=_text(item.get('modality'),MODALITY_MAX)
            slot.person_id=_text(item.get('person_id'),ID_MAX)
            slot.stamp_ms=int(item.get('stamp_ms') or 0)
            slot.ttl_ms=int(item.get('ttl_ms') or 600)
            slot.link_id=_text(item.get('link_id'),ID_MAX)
            slot.change_id=_text(item.get('change_id'),ID_MAX)
            slot.event_id=_text(item.get('event_id'),ID_MAX)
            for name in ('salience','goal','surprise','observability','uncertainty','importance','gaze','body_facing',
                         'lip_sync','voice_activity','identity_confidence','memory_participant','memory_recency',
                         'memory_expected_answer','memory_dialogue_target','lang_directed_call','lang_question',
                         'lang_answer_continuation','lang_topic_continuation','lang_addresses_other','lang_backchannel',
                         'motivation','link_confidence'):
                setattr(slot,name,float(item.get(name) or 0.))
        frame=FrameIn(abi_version=ABI_VERSION,stamp_ms=int(stamp_ms),arousal_gain=float(arousal_gain),
                      source_audio_visual=int(bool(sources.get('audio_visual'))),
                      source_memory_context=int(bool(sources.get('memory_context'))),
                      source_linguistic_context=int(bool(sources.get('linguistic_context'))),
                      source_internal_state=int(bool(sources.get('internal_state'))),
                      robot_speaking=int(bool(robot_speaking)),reflex_active=int(bool(reflex_active)),
                      candidate_count=len(candidates),candidates=array)
        result=self.library.attention_plugin_update(self.handle,ctypes.byref(frame))
        assert extras is not None # holds the borrowed key/value buffers past the call
        if not result:
            return None
        out=result.contents
        rows={}
        for index in range(out.candidate_count):
            row=out.candidates[index]
            rows[row.candidate_id.decode('utf-8','replace')]={
                'modality':row.modality.decode('utf-8','replace'),'a':row.activation,'h':row.habituation,
                'r':row.inhibition,'c':row.change,'normalized':row.normalized,'input':row.input,
                'goal':row.goal,'uncertainty':row.uncertainty,'learning':row.learning,
                'engagement':row.engagement,'addressee_robot':row.addressee_robot,
                'valid':bool(row.valid),'focus':bool(row.is_focus),'reasons':reasons_from_mask(row.reason_mask)}
        return {'stamp_ms':out.stamp_ms,'model':self.id,'calibrated':bool(self.manifest.get('calibrated')),
                'focus':{'visual':out.visual_focus.decode('utf-8','replace') or None,
                         'audio':out.audio_focus.decode('utf-8','replace') or None},
                'candidates':rows,
                'residual':{'visual':max(0.,out.residual_visual),'audio':max(0.,out.residual_audio)},
                'engagement':{'state':out.engagement_state.decode('utf-8','replace'),
                              'person_id':out.engaged_person.decode('utf-8','replace') or None,
                              'confidence':out.engagement_confidence,
                              'reasons':reasons_from_mask(out.engagement_reason_mask)},
                'transition_sequence':out.transition_sequence}


class BuiltinAudioVisualAlgorithm:
    """Pure-Python audio-visual competition, kept as the comparison baseline.

    It has no engagement layer, so selecting it is an explicit ablation: the
    console must show that memory and language evidence stop reaching attention.
    """

    kind='builtin'
    id='builtin_audio_visual_v1'
    manifest={'id':'builtin_audio_visual_v1','version':'1.0.0','abi':ABI_VERSION,
              'display_name':'仅视听 连续竞争 v1（对照基线）',
              'summary':'原有归一化竞争，只消费视听证据；不产生交流意愿状态。',
              'required_sources':['audio_visual'],'optional_sources':['internal_state'],
              'outputs':['attention_distribution'],'engagement_states':[],'calibrated':False,
              'limits':['无交流意愿层：无声邀请与转头回答不会被识别','权重为工程先验，未做现场标定']}

    def __init__(self):
        from .attention_competition import AttentionCompetition,Candidate
        self._candidate=Candidate
        self.engine=AttentionCompetition()

    def close(self):
        return None

    def reset(self):
        from .attention_competition import AttentionCompetition
        self.engine=AttentionCompetition()

    def update(self,stamp_ms,candidates,sources,arousal_gain=1.,robot_speaking=False,reflex_active=False):
        built=[]
        for item in candidates:
            links=()
            if sources.get('audio_visual') and item.get('link_id'):
                links=((str(item['link_id']),float(item.get('link_confidence') or 0.)),)
            built.append(self._candidate(
                key=str(item.get('candidate_id') or ''),modality=str(item.get('modality') or ''),
                stamp_ms=int(item.get('stamp_ms') or 0),salience=float(item.get('salience') or 0.),
                goal=float(item.get('goal') or 0.),importance=float(item.get('importance') or 0.),
                uncertainty=float(item.get('uncertainty') or 0.),observability=float(item.get('observability') or 0.),
                motivation=float(item.get('motivation') or 0.) if sources.get('internal_state') else 0.,
                surprise=float(item.get('surprise') or 0.),change_id=str(item.get('change_id') or ''),
                links=links,ttl_ms=int(item.get('ttl_ms') or 600)))
        result=self.engine.update(int(stamp_ms),built,arousal=arousal_gain)
        rows={key:dict(value,engagement=0.,addressee_robot=0.,focus=result['focus'].get(value['modality'])==key,
                       reasons=[]) for key,value in result['candidates'].items()}
        return {'stamp_ms':result['stamp_ms'],'model':self.id,'calibrated':False,'focus':result['focus'],
                'candidates':rows,'residual':result['residual'],
                'engagement':{'state':'','person_id':None,'confidence':0.,'reasons':[],
                              'unavailable_reason':'selected_algorithm_has_no_engagement_layer'},
                'transition_sequence':result['transition_sequence']}


def default_plugin_dir():
    return Path(os.environ.get('ATTENTION_PLUGIN_DIR') or Path(__file__).resolve().parents[1]/'plugins')


class AlgorithmRegistry:
    """Discovers algorithms and swaps the active one without restarting."""

    def __init__(self,plugin_dir=None,preferred=None):
        self.plugin_dir=Path(plugin_dir) if plugin_dir else default_plugin_dir()
        self.lock=threading.RLock()
        self.errors={}
        self.active=None
        self._builtin=BuiltinAudioVisualAlgorithm()
        self.available={self._builtin.id:{'id':self._builtin.id,'kind':'builtin','path':'',
                                          'manifest':self._builtin.manifest}}
        self._discover()
        self.requested=preferred or os.environ.get('ATTENTION_ALGORITHM') or 'av_memory_language_v1'
        if not self.select(self.requested):
            self.errors['requested_algorithm']=f'{self.requested} 未加载，已回退到内置基线'
            self.select(self._builtin.id)

    def _discover(self):
        self.libraries={}
        if not self.plugin_dir.is_dir():
            self.errors['plugin_dir']=f'{self.plugin_dir} 不存在，未编译注意力算法库'
            return
        for path in sorted(self.plugin_dir.glob('libattention_*.so')):
            try:
                algorithm=SharedLibraryAlgorithm(path)
            except Exception as exc:
                self.errors[path.name]=str(exc)
                continue
            self.libraries[algorithm.id]=algorithm
            self.available[algorithm.id]={'id':algorithm.id,'kind':'shared_library','path':str(path),
                                          'manifest':algorithm.manifest}

    def select(self,algorithm_id):
        with self.lock:
            algorithm_id=str(algorithm_id or '')
            algorithm=self._builtin if algorithm_id==self._builtin.id else self.libraries.get(algorithm_id)
            if algorithm is None:
                return False
            if algorithm is not self.active:
                algorithm.reset()
            self.active=algorithm
            return True

    def update(self,*args,**kwargs):
        with self.lock:
            active=self.active
        return active.update(*args,**kwargs) if active else None

    def status(self):
        with self.lock:
            active=self.active
            return {'active':active.id if active else '','plugin_dir':str(self.plugin_dir),
                    'requested':getattr(self,'requested',''),
                    'algorithms':[{'id':entry['id'],'kind':entry['kind'],'path':entry['path'],
                                   'display_name':entry['manifest'].get('display_name',entry['id']),
                                   'summary':entry['manifest'].get('summary',''),
                                   'required_sources':entry['manifest'].get('required_sources',[]),
                                   'optional_sources':entry['manifest'].get('optional_sources',[]),
                                   'calibrated':bool(entry['manifest'].get('calibrated')),
                                   'limits':entry['manifest'].get('limits',[]),
                                   'references':entry['manifest'].get('references',[]),
                                   'active':bool(active and entry['id']==active.id)}
                                  for entry in sorted(self.available.values(),key=lambda item:item['id'])],
                    'load_errors':dict(self.errors)}

    def close(self):
        with self.lock:
            for algorithm in self.libraries.values():
                algorithm.close()
            self.libraries={}
            self.active=None
