"""Short-lived orienting attention, separate from permission to answer speech."""
from collections import deque
import math
import os


# Looming is judged in degrees per second, not in how much of the frame the box
# grew, because the same box growth means a different approach depending on the
# lens. In mice, flight is induced by stimuli subtending 10-40 degrees and
# expanding at 57-320 deg/s; slower expansion produces freezing rather than
# flight, and the canonical looming stimulus (Yilmaz & Meister 2013) goes from 2
# to 20 degrees in 250 ms, about 72 deg/s. Those numbers come from overhead dark
# discs shown to mice, so they are an order-of-magnitude guide here, not a
# calibration — hence both are tunable and neither has been checked on this
# camera against real approaches.
CAMERA_HFOV_DEG = float(os.environ.get('ATTENTION_CAMERA_HFOV_DEG', '60'))
LOOM_MIN_ANGLE_DEG = float(os.environ.get('ATTENTION_LOOM_MIN_DEG', '10'))
LOOM_MIN_SPEED_DEG_S = float(os.environ.get('ATTENTION_LOOM_MIN_DEG_PER_S', '57'))
# Matches the detector's own operating point; requiring more than the detector
# reports is what silently disabled the visual paths once before.
FACE_CONFIDENCE = float(os.environ.get('ATTENTION_FACE_CONFIDENCE', '0.80'))


def angular_size_deg(area_ratio):
    """Rough angular width of a box covering this fraction of the frame."""
    return math.sqrt(max(0., float(area_ratio or 0.))) * CAMERA_HFOV_DEG


def spatial_source(source, item):
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    xyz = item.get('position_xyz_m')
    valid_xyz = (item.get('position_valid') is not False and isinstance(xyz, (list, tuple))
                 and len(xyz) == 3 and all(finite(v) for v in xyz) and bool(item.get('frame_id')))
    azimuth = item.get('direction_deg', item.get('azimuth_deg'))
    if item.get('direction_valid') is False or not finite(azimuth):
        azimuth = None
    distance = item.get('distance_m')
    if not finite(distance) or distance <= 0:
        distance = None
    return {'frame_id': item.get('frame_id') or ('robot_audio' if source == 'audio' else 'camera'),
            'stamp_ms': item.get('stamp_ms'), 'azimuth_deg': azimuth,
            'elevation_deg': item.get('elevation_deg') if finite(item.get('elevation_deg')) else None,
            'distance_m': distance, 'depth_source': item.get('depth_source','none'),
            'distance_confidence': item.get('distance_confidence'),
            'position_xyz_m': list(xyz) if valid_xyz else None, 'position_valid': bool(valid_xyz),
            'mode': '3d' if valid_xyz else 'bearing_with_depth_estimate' if azimuth is not None and distance is not None else 'bearing_only' if azimuth is not None else 'unknown'}



class ReflexAttention:
    def __init__(self, hold_ms=1800):
        self.hold_ms = hold_ms
        self.seen = deque(maxlen=512)
        self.focus = None
        self.visual_previous = {}
        self.last_visual = {}
        self.started_ms = None
        self.object_tracks = {}
        self.object_sequence = 0
        self.last_visual_event = -100000

    def update(self, stamp, voice, vision):
        raw = vision.get('state') if isinstance(vision.get('state'),dict) else vision
        candidates = []
        for event in (voice.get('audio_scene') or {}).get('events', []):
            if not event.get('id') or event.get('kind') not in {'startle','orient'} or event.get('id') in self.seen or not 0 <= stamp-event.get('stamp_ms',0) < 1000:
                continue
            self.seen.append(event['id'])
            spatial = spatial_source('audio', event)
            candidates.append(dict(event, spatial=spatial, source='audio', target_id='sound:'+event['id'], azimuth_deg=spatial['azimuth_deg']))
        if self.started_ms is None:self.started_ms=stamp
        sources=list((raw or {}).get('people', []))
        used=set()
        self.object_tracks={k:v for k,v in self.object_tracks.items() if stamp-v['stamp_ms']<1000}
        for obj in (raw or {}).get('objects',[]):
            center=obj.get('center') or [0,0]
            if obj.get('confidence',0)<.85 or not 0<=stamp-obj.get('stamp_ms',0)<500:continue
            matches=[(sum((a-b)**2 for a,b in zip(center,v['center'])),k) for k,v in self.object_tracks.items() if k not in used and v['label']==obj['label']]
            key=min(matches)[1] if matches and min(matches)[0]<.04 else None
            if key is None:
                self.object_sequence+=1;key='object:'+str(self.object_sequence)
            self.object_tracks[key]=dict(obj);used.add(key)
            sources.append(dict(obj,person_id=key,face_confidence=obj['confidence'],is_object=True))
        for person in sources:
            key = str(person.get('person_id') or '')
            area = float(person.get('reflex_area_ratio',person.get('bbox_area_ratio')) or 0)
            previous = self.visual_previous.get(key)
            frame_ms = int(person.get('stamp_ms') or float((raw or {}).get('stamp') or stamp/1000)*1000)
            if previous and frame_ms > previous['stamp_ms']:
                dt = frame_ms-previous['stamp_ms']
                # A confirmed, rapid expansion of the same visible source; no trigger on first detection.
                angle, was = angular_size_deg(area), angular_size_deg(previous['area'])
                expansion = (angle-was)/dt*1000 if dt else 0.
                approach = (100 <= dt <= 600 and angle >= LOOM_MIN_ANGLE_DEG
                            and expansion >= LOOM_MIN_SPEED_DEG_S)
                center=person.get('center') or [0,0]
                appeared = (person.get('is_object') and previous.get('new_large') and 100<=dt<=400 and area>=.3
                            and .25<=center[0]<=.75 and .2<=center[1]<=.8)
                if (approach or appeared) and person.get('face_confidence',0)>=FACE_CONFIDENCE and stamp-self.last_visual_event>=8000:
                    event = {'id': f'visual-approach:{key}:{frame_ms}', 'kind':'startle', 'source':'vision',
                             'target_id':key if person.get('is_object') else 'person:'+key, 'stamp_ms':stamp, 'azimuth_deg':person.get('azimuth_deg'),
                             'evidence':{'before_area':previous['area'],'area':area,'interval_ms':dt,
                                         'angle_deg':round(angle,1),'expansion_deg_per_s':round(expansion,1)},
                             'reason':'sudden_large_object' if appeared else 'rapid_visual_approach', 'spatial':spatial_source('vision',dict(person, stamp_ms=frame_ms))}
                    self.last_visual[key]=stamp
                    self.last_visual_event=stamp
                    candidates.append(event)
                if dt >= 400:
                    self.visual_previous[key]={'stamp_ms':frame_ms,'area':area}
            elif not previous:
                self.visual_previous[key]={'stamp_ms':frame_ms,'area':area,'new_large':bool(person.get('is_object') and area>=.3 and stamp-self.started_ms>=2000)}
        self.visual_previous={k:v for k,v in self.visual_previous.items() if stamp-v['stamp_ms'] < 3000}
        if candidates:
            event = max(candidates,key=lambda item:(item['kind']=='startle',item['stamp_ms']))
            self.focus = dict(event, active=True, expires_ms=stamp+self.hold_ms, listen=False, addressed_to_robot=False)
        if self.focus and stamp > self.focus['expires_ms']:
            self.focus = dict(self.focus, active=False)
        return self.focus
