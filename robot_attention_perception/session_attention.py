"""Read-only top-down support from real active dialogue, never new permission."""
import threading

class SessionAttention:
    def __init__(self,session):
        self.session=session;self.lock=threading.Lock();self.latest=None
    def receive(self,payload):
        if payload.get('session_id')!=self.session:return
        with self.lock:self.latest=dict(payload)
    def context(self,stamp,people):
        with self.lock:state=self.latest
        if not state or not 0<=stamp-int(state.get('stamp_ms') or 0)<1200:return {'goals':[],'reason':'brain_state_unavailable'}
        turn=state.get('current_turn') or {};s2s=state.get('s2s') or {}
        active=turn.get('status') in {'thinking','synthesizing','speaking'} and any(s2s.get(k) for k in ['thinking','generating','speaking'])
        if not active or turn.get('reflex'):return {'goals':[],'reason':'no_active_dialogue'}
        gate=turn.get('attention') or {};person=turn.get('person_id')
        # A voice/face identity conflict does not redirect the face goal to the voice ID.
        if gate.get('identity_conflict'):return {'goals':[],'reason':'identity_conflict'}
        visible=next((p for p in people if p.person_id==person and p.face_visible and 0<=stamp-p.stamp_ms<600),None)
        if visible is None:return {'goals':[],'reason':'partner_not_visible'}
        return {'goals':[{'candidate_id':'visual:'+person,'stamp_ms':state['stamp_ms'],'confirmed':True,'strength':.85,'purpose':'active_dialogue','turn_id':turn.get('turn_id')}],
                'reason':'active_dialogue'}
