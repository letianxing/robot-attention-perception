"""Time-continuous, per-modality normalized competition inspired by PRD §3.

Engineering parameters, not calibrated biological constants. No speech or action
permission is granted here. All updates read the previous state simultaneously.
"""
from dataclasses import dataclass
from collections import deque
import math


def unit(value):
    value=float(value)
    return min(1.,max(0.,value)) if math.isfinite(value) else 0.


@dataclass
class Candidate:
    key: str
    modality: str
    stamp_ms: int
    salience: float=0.
    goal: float=0.
    importance: float=0.
    uncertainty: float=0.
    observability: float=0.
    motivation: float=0.
    surprise: float=0.
    change_id: str=''
    links: tuple=()
    ttl_ms: int=600


class AttentionCompetition:
    def __init__(self):
        self.state={};self.focus={'visual':None,'audio':None};self.stamp=None
        self.changes=deque(maxlen=2048);self.feedback_seen=deque(maxlen=2048)
        self.feedback={};self.sequence=0

    def observe_result(self,event_id,key,problem,condition,completed,new_information):
        if not completed or event_id in self.feedback_seen:return False
        self.feedback_seen.append(event_id)
        counts=self.feedback.setdefault((key,problem,condition),[1,1])
        counts[0]+=int(bool(new_information));counts[1]+=int(not new_information)
        return True

    def learning(self,key,problem='identity',condition='normal'):
        a,b=self.feedback.get((key,problem,condition),(1,1));return a/(a+b)

    def update(self,stamp,candidates,arousal=1.):
        if self.stamp is not None and stamp<=self.stamp:return self.snapshot()
        dt=0. if self.stamp is None else (stamp-self.stamp)/1000
        self.stamp=stamp;gain=min(2.,max(.5,float(arousal)))
        current={c.key:c for c in candidates if c.modality in self.focus and 0<=stamp-c.stamp_ms<c.ttl_ms}
        previous={key:dict(value) for key,value in self.state.items()}
        for key,c in current.items():
            previous.setdefault(key,{'a':0.,'h':0.,'r':0.,'c':0.,'modality':c.modality,'last_seen':stamp})
        activations={};new_changes=set()
        for key,c in current.items():
            old=previous[key];change=old['c']
            if c.change_id and c.change_id not in self.changes:
                change=1.;new_changes.add(c.change_id)
            cross=sum(unit(conf)*previous.get(other,{}).get('a',0.) for other,conf in c.links
                      if other in current and current[other].modality!=c.modality)
            learning=self.learning(key)*unit(c.observability)
            motive=unit(c.motivation)*unit(c.uncertainty)*learning
            e=gain*(unit(c.salience)+.6*unit(c.surprise))*(1-old['h']*(1-change))*(1+.2*min(1,cross))+.9*unit(c.goal)+.6*motive+.3*unit(c.importance)
            u=max(0.,e+.2*old['a']-.3*old['r']*(1-unit(c.goal))*(1-change))
            activations[key]=(u,change,e,learning)
        self.changes.extend(sorted(new_changes))
        denominators={m:.3**2+sum(v[0]**2 for k,v in activations.items() if current[k].modality==m) for m in self.focus}
        alpha=-math.expm1(-dt/(.3/gain))
        updated={}
        for key,old in previous.items():
            c=current.get(key);u,change,e,learning=activations.get(key,(0.,old['c'],0.,0.))
            z=u*u/denominators[old['modality']] if c else 0.
            a=old['a']+alpha*(z-old['a'])
            # Exact solution of exposure/recovery/change ODE, bounded in [0,1].
            exposure=1. if c else 0.;up=exposure*(1-change)/10
            down=(1-exposure)/30+change/.5;rate=up+down
            equilibrium=up/rate if rate else 0.
            h=equilibrium+(old['h']-equilibrium)*math.exp(-rate*dt) if rate else old['h']
            updated[key]={'a':a,'h':unit(h),'r':old['r']*math.exp(-dt/1.5),'c':change*math.exp(-dt),
                          'modality':old['modality'],'last_seen':stamp if c else old['last_seen'],'valid':bool(c),
                          'input':e,'normalized':z,'learning':learning,'uncertainty':unit(c.uncertainty) if c else 0.,
                          'goal':unit(c.goal) if c else 0.}
        for modality in self.focus:
            ranked=sorted(((v['a'],k) for k,v in updated.items() if v['modality']==modality and v['valid']),reverse=True)
            old_focus=self.focus[modality];winner=ranked[0] if ranked else (0.,None);runner=ranked[1][0] if len(ranked)>1 else 0.
            held=old_focus in current and updated[old_focus]['a']>=.15
            focus=old_focus if held else None
            if winner[0]>=.25 and winner[0]-runner>=.08 and (not held or winner[1]==old_focus or winner[0]-updated[old_focus]['a']>=.08):focus=winner[1]
            if focus!=old_focus:
                self.sequence+=1
                if old_focus in updated:updated[old_focus]['r']=1.
            self.focus[modality]=focus
        self.state={k:v for k,v in updated.items() if stamp-v['last_seen']<60000}
        return self.snapshot()

    def snapshot(self):
        return {'stamp_ms':self.stamp,'model':'normalized_competition_prd_v1','calibrated':False,'focus':dict(self.focus),
                'candidates':{k:{name:round(v[name],5) if isinstance(v[name],float) else v[name] for name in v} for k,v in self.state.items()},
                'residual':{m:max(0.,1-sum(v['a'] for v in self.state.values() if v['modality']==m)) for m in self.focus},
                'transition_sequence':self.sequence}
