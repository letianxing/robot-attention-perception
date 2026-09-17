#!/usr/bin/env python3
"""Drive the real source adapters, the real .so and the real fusion offline.

No camera, microphone, model server or network is touched: synthetic perception
payloads are fed through the same code path the live runtime uses, so this shows
the wiring works end to end. It says nothing about field accuracy.

    .venv-mac/bin/python scripts/verify_attention_algorithm.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_attention_perception.attention_sources import AttentionSources
from robot_attention_perception.familiarity import FamiliarityIndex
from robot_attention_perception.interaction_fusion import InteractionFusion
from robot_attention_perception.types import AcousticTrack, VisionPerson

STEP_MS = 50


def face(stamp, person, gaze=.9, facing=.9, lips=False, identity=.8):
    return VisionPerson(person, stamp, face_id='f-' + person, face_visible=True, face_confidence=.9,
                        gaze_score=gaze, body_facing_score=facing, lip_motion=lips, identity_confidence=identity)


def track(stamp, person, track_id='t1', similarity=.9):
    return AcousticTrack(track_id, stamp, voice_activity=True, speech_probability=.9, clarity=.8,
                         speaker_label=person, speaker_similarity=similarity)


def memory_index(prior_sessions, session_id='verify'):
    """A stand-in for hri-memory-service so nothing here needs a running server."""
    def search(person_id):
        return {'hits': [{'record': {'entity': {'person_id': person_id},
                                     'scope': {'session_id': f'past-{index}'},
                                     'observed_at_ms': index}} for index in range(prior_sessions)]}
    return FamiliarityIndex('http://unused', 'subject', 'robot', session_id, search=search)


def brain_state(stamp, participants=(), expected=None, current=None):
    return {'session_id': 'verify', 'stamp_ms': stamp, 's2s': {},
            'current_turn': current or {},
            'attention_memory': {
                'recent_participants': [{'person_id': person, 'last_heard_ms': stamp - 4000} for person in participants],
                'expected_answer': expected}}


def final_transcript(stamp, text, person, track_id='t1', similarity=.9, utterance_id='u-1'):
    """One utterance keeps one id, so a phasic language effect fires once."""
    return {'last_transcript': {'is_final': True, 'utterance_id': utterance_id, 'emitted_ms': stamp, 'text': text,
                                'track_id': track_id, 'speaker': {'speaker_id': person, 'similarity': similarity}}}


def run(label, frames, sources=('audio_visual', 'memory_context', 'linguistic_context'), familiarity=None):
    """frames: list of (people, tracks, voice, brain) for consecutive 50 ms ticks."""
    adapters = AttentionSources('verify', enabled=list(sources), familiarity=familiarity)
    fusion = InteractionFusion()
    stamp = 1_000_000
    state = {}
    for people, tracks, voice, brain in frames:
        if brain is not None:
            adapters.receive_brain(brain(stamp))
        live_people = [item(stamp) for item in people]
        live_tracks = [item(stamp) for item in tracks]
        payload = voice(stamp) if voice else {}
        context = adapters.context(stamp, live_people, live_tracks, payload)
        state = fusion.update(stamp, {'algorithm': 'verify'}, live_people, live_tracks, payload, None, context=context)
        stamp += STEP_MS
    engagement = state.get('engagement') or {}
    print(f"{label:<44} phase={state.get('interaction_phase'):<14} addressed={str(bool(state.get('addressed_to_robot'))):<5} "
          f"engagement={engagement.get('state') or '--':<16} conf={engagement.get('confidence', 0):.3f} "
          f"reasons={','.join(engagement.get('reasons') or []) or '--'}")
    return state, engagement


def hold(count, people=(), tracks=(), voice=None, brain=None):
    return [(people, tracks, voice, brain)] * count


def main():
    failures = []

    def expect(condition, message):
        if not condition:
            failures.append(message)

    algorithm = InteractionFusion().competition.registry.status()
    print('active algorithm:', algorithm['active'], '| plugin dir:', algorithm['plugin_dir'])
    if algorithm['load_errors']:
        print('load errors:', algorithm['load_errors'])
    print()

    owner_gaze = [lambda stamp: face(stamp, 'owner')]
    memory_of_discussion = lambda stamp: brain_state(stamp, participants=('owner', 'guest'))

    state, engagement = run('两人讨论后，其中一人静静看向机器人',
                            hold(60, people=owner_gaze, brain=memory_of_discussion))
    expect(state['interaction_phase'] == 'INVITED' and not state['addressed_to_robot'],
           '参与者的持续注视应进入 INVITED 且不算对机器人说话')

    state, engagement = run('同样证据，拔掉记忆来源（消融）',
                            hold(60, people=owner_gaze, brain=memory_of_discussion),
                            sources=('audio_visual', 'linguistic_context'))
    expect(state['interaction_phase'] != 'INVITED', '关闭记忆来源后不应仍然产生邀请')

    state, engagement = run('陌生人只是长时间盯着看',
                            hold(80, people=[lambda stamp: face(stamp, 'guest')], brain=None))
    expect(state['interaction_phase'] != 'INVITED', '没有记忆语境的注视不应被当作邀请')

    answer_memory = lambda stamp: brain_state(stamp, participants=('owner',),
                                              expected={'person_id': 'owner', 'expires_ms': stamp + 8000,
                                                        'question': '周六去公园好不好？'})
    state, engagement = run('机器人问完后，本人转头回答（无人脸）',
                            hold(8, tracks=[lambda stamp: track(stamp, 'owner')],
                                 voice=lambda stamp: final_transcript(stamp, '好啊周六去公园', 'owner'),
                                 brain=answer_memory))
    expect(state['addressed_to_robot'] and 'expected_answer_to_robot_question' in state['reasons'],
           '对机器人问题的回答应在注意力层放行')

    state, engagement = run('同一场景，但这句话是在叫另一个人',
                            hold(20, tracks=[lambda stamp: track(stamp, 'owner')],
                                 voice=lambda stamp: final_transcript(stamp, '小林你觉得呢', 'owner'),
                                 brain=lambda stamp: brain_state(stamp, participants=('owner', '小林'))))
    expect(not state['addressed_to_robot'], '称呼他人的话不应被当作对机器人说')

    familiar_sources = ('audio_visual', 'memory_context', 'linguistic_context', 'cross_session_memory')
    index = memory_index(prior_sessions=6)
    # Warm the cache at the same clock the scenarios run on.
    index.request(1_000_000, ['owner'])
    while index.status(1_000_000)['pending']:
        time.sleep(0.01)
    state, engagement = run('以往会话常聊的人回来，静静看向机器人',
                            hold(80, people=owner_gaze, brain=None),
                            sources=familiar_sources, familiarity=index)
    expect(state['interaction_phase'] == 'INVITED', '跨会话熟悉的人应可发起无声邀请')
    expect('familiar_across_sessions' in (engagement.get('reasons') or []), '应记录跨会话熟悉度这条依据')

    state, engagement = run('同一个人，但只看了 1 秒',
                            hold(20, people=owner_gaze, brain=None),
                            sources=familiar_sources, familiarity=index)
    expect(state['interaction_phase'] != 'INVITED', '熟悉度应比当前会话参与者更慢，需要更长注视')

    state, engagement = run('声纹匹配很弱的人，蹭别人的待答问题',
                            hold(10, tracks=[lambda stamp: track(stamp, 'owner', similarity=.5)],
                                 voice=lambda stamp: final_transcript(stamp, '好啊周六去公园', 'owner', similarity=.5),
                                 brain=answer_memory))
    expect(not state['addressed_to_robot'], '识别不可靠时不能继承对别人提的问题')

    state, engagement = run('走过来看着机器人直接说话（无历史、无点名）',
                            hold(6, people=[lambda stamp: face(stamp, 'guest', lips=True)],
                                 tracks=[lambda stamp: track(stamp, 'guest')],
                                 voice=lambda stamp: final_transcript(stamp, '现在几点了', 'guest'),
                                 brain=None))
    expect(state['addressed_to_robot'], '走过来看着机器人说话应被放行')
    expect(engagement.get('state') == 'ENGAGED', '交流意愿层自己就应达到 ENGAGED，不依赖离散快通道')

    state, engagement = run('镜头外普通音量直接叫名字',
                            hold(6, tracks=[lambda stamp: track(stamp, 'owner')],
                                 voice=lambda stamp: final_transcript(stamp, '小圆，现在几点', 'owner'),
                                 brain=None))
    expect(state['addressed_to_robot'], '句首点名应始终放行')

    print()
    if failures:
        for item in failures:
            print('FAILED:', item)
        return 1
    print('全部离线检查通过。这只验证了合成输入下的接线与阈值行为，不代表现场识别准确率。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
