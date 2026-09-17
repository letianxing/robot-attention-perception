"""Explicit shared-conversation membership and conservative addressee evidence.

Membership alone never authorizes speech. Unknown addressees stay unknown.
"""
import re
from collections import deque

class GroupConversation:
    def __init__(self):
        self.members={};self.active_until=0;self.last_turns=deque(maxlen=30)

    def annotate(self,turn):
        stamp=turn['stamp_ms'];text=turn['transcript']['text'].strip()
        person=turn.get('person_id','unknown');gate=turn['attention']
        self.members={k:v for k,v in self.members.items() if stamp-v<120000}
        valid_person=person not in {'','unknown','robot',None}
        if valid_person:self.members[person]=stamp
        addressed=bool(gate.get('listen') and gate.get('addressed_to_robot'))
        if addressed and re.search(r'(一起聊|加入.*讨论|我们三个|一起讨论)',text):
            self.active_until=stamp+120000
        if addressed and re.search(r'(退出讨论|先别参与|你先别说)',text):self.active_until=0
        active=stamp<self.active_until
        named=[p for p in self.members if p!=person and not p.startswith(('stranger_','anonymous_','vision_'))
               and re.match(r'^'+re.escape(p)+r'[，,：:\s]',text)]
        target='robot' if addressed else 'unknown'
        confidence=float(gate.get('confidence') or 0) if addressed else 0.
        reason='attention_gate' if addressed else 'insufficient_addressee_evidence'
        if named and not re.match(r'^(?:你好[，, ]*)?小圆',text):
            target='person:'+named[0];confidence=.9;reason='explicit_human_address'
            gate.update(listen=False,addressed_to_robot=False,reasons=list(gate.get('reasons',[]))+[reason])
        group_question=bool(re.search(r'(大家|你们|各位).*(怎么看|觉得|认为|同意|意见|建议|好吗|吗|呢|？|\?)',text))
        if active and valid_person and group_question and target=='unknown' and turn.get('source')!='robot_echo' and float(turn['transcript'].get('self_echo_probability') or 0)<.4:
            target='group';confidence=.8;reason='explicit_group_question_in_joined_conversation'
            gate.update(listen=True,addressed_to_robot=True,confidence=max(.8,float(gate.get('confidence') or 0)),reasons=list(gate.get('reasons',[]))+[reason])
        prior=[t for t in self.last_turns if 0<=stamp-t['stamp_ms']<=15000]
        overlap=any(turn['transcript'].get('started_ms',stamp)<t['ended_ms'] and t['person_id']!=person for t in prior)
        turn.update(addressee=target,addressee_confidence=confidence,addressee_reason=reason,
                    group_context={'active':active,'participants':list(self.members)+(['robot'] if active else []),
                                   'floor_speaker':person,'overlapping_turns':overlap,'joined_until_ms':self.active_until if active else None})
        self.last_turns.append({'person_id':person,'stamp_ms':stamp,'ended_ms':turn['transcript'].get('ended_ms',stamp)})
        return turn
