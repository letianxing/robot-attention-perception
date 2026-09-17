import unittest
from dataclasses import replace

from robot_attention_perception.person_manager import (
    IdentityEvidence,
    PersonManagerConfig,
    ProbabilisticPersonManager,
)
from robot_attention_perception.types import AcousticTrack, VisionPerson


def person(stamp_ms, face_id="face_1", body_id="body_1", azimuth=5.0):
    return VisionPerson(
        person_id="raw_person_0",
        stamp_ms=stamp_ms,
        face_id=face_id,
        body_id=body_id,
        azimuth_deg=azimuth,
        face_visible=True,
        face_confidence=0.95,
        gaze_score=0.8,
    )


def voice(stamp_ms, track_id="voice_1", azimuth=6.0):
    return AcousticTrack(
        track_id=track_id,
        stamp_ms=stamp_ms,
        voice_activity=True,
        speech_probability=0.9,
        clarity=0.8,
        azimuth_deg=azimuth,
    )


class PersonManagerTest(unittest.TestCase):
    def test_builds_stable_temporary_person_and_binds_voice_probabilistically(self):
        manager = ProbabilisticPersonManager()
        output = []
        for index in range(5):
            stamp = 1000 + index * 50
            output = manager.update([person(stamp)], [voice(stamp)], stamp)

        self.assertEqual(output[0].person_id, "anonymous_0001")
        self.assertEqual(output[0].voice_id, "voice_1")
        self.assertEqual(len(manager.tracks), 1)

    def test_reidentifies_changed_observation_ids_within_retention_window(self):
        manager = ProbabilisticPersonManager()
        first = manager.update([person(1000)], [], 1000)[0]
        manager.update([], [], 1800)
        returned = manager.update(
            [person(4000, face_id="face_1", body_id="body_2", azimuth=7.0)], [], 4000
        )[0]

        self.assertEqual(first.person_id, returned.person_id)

    def test_expires_temporary_person_after_retention_window(self):
        manager = ProbabilisticPersonManager()
        # A face_track id is the detector's bookkeeping for overlapping boxes, so
        # after six seconds away it vouches for nothing and this is a new person.
        box_track = lambda stamp: replace(person(stamp), person_id="vision_face_track_0001")
        first = manager.update([box_track(1000)], [], 1000)[0]
        second = manager.update([box_track(7000)], [], 7000)[0]

        self.assertNotEqual(first.person_id, second.person_id)

    def test_identity_evidence_is_smoothed_before_persistent_binding(self):
        manager = ProbabilisticPersonManager()
        manager.update([person(1000)], [], 1000)

        manager.ingest_identity_evidence(IdentityEvidence("face", "face_1", {"owner": 1.0}, 1010))
        self.assertEqual(manager.active_people(1010)[0].person_id, "anonymous_0001")
        for index in range(12):
            manager.ingest_identity_evidence(
                IdentityEvidence("face", "face_1", {"owner": 1.0}, 1020 + index)
            )

        self.assertEqual(manager.active_people(1100)[0].person_id, "owner")
        self.assertEqual(manager.active_people(1100)[0].role, "known")

    def test_sustained_ambiguous_voice_mapping_requests_clarification(self):
        manager = ProbabilisticPersonManager(
            PersonManagerConfig(ambiguity_hold_ms=100, clarification_cooldown_ms=1000)
        )
        behavior_seen = False
        for index in range(10):
            stamp = 1000 + index * 50
            manager.update(
                [person(stamp)],
                [voice(stamp, "voice_1"), voice(stamp, "voice_2")],
                stamp,
            )
            behavior_seen = behavior_seen or bool(manager.active_behaviors)

        self.assertTrue(behavior_seen)


if __name__ == "__main__":
    unittest.main()

class IdentityIsolationTests(unittest.TestCase):
    def test_offscreen_voice_cannot_rename_visible_owner_face(self):
        from dataclasses import replace
        manager=ProbabilisticPersonManager()
        face=replace(person(1000),person_id='owner',role='owner',identity_confidence=.9)
        other=replace(voice(1000),speaker_label='other',speaker_similarity=.99)
        result=manager.update([face],[other],1000)
        self.assertEqual(result[0].person_id,'owner')
        self.assertEqual(result[0].role,'owner')
        self.assertNotIn('other',next(iter(manager.tracks.values())).identity_probabilities)


class IdentityContinuityTest(unittest.TestCase):
    """One person in the room has to stay one person.

    The field session created anonymous_0001 through anonymous_0013 in two
    minutes for what was a single visitor: every time the face box lost its
    track, vision issued a new id and this manager made a new stranger out of it,
    so nothing they said ever accumulated against them.
    """

    def guest(self, stamp_ms, person_id="vision_guest_0001", azimuth=5.0):
        return VisionPerson(person_id=person_id, stamp_ms=stamp_ms, face_id=f"{person_id}_face_obs",
                            body_id=f"{person_id}_body_obs", azimuth_deg=azimuth, face_visible=True,
                            face_confidence=0.9, gaze_score=0.7)

    def test_the_same_person_id_after_a_long_gap_is_the_same_person(self):
        manager = ProbabilisticPersonManager()
        manager.update([self.guest(1000)], [], 1000)
        first = list(manager.tracks)
        created = [event for event in manager.events if event["kind"] == "temporary_person_created"]

        # Away long enough that box geometry could not vouch for anything, and
        # back from a different place in the frame.
        manager.update([self.guest(40000, azimuth=-40.0)], [], 40000)

        created += [event for event in manager.events if event["kind"] == "temporary_person_created"]
        self.assertEqual(list(manager.tracks), first)
        self.assertEqual(len(created), 1)

    def test_a_different_person_id_is_still_a_different_person(self):
        manager = ProbabilisticPersonManager()
        manager.update([self.guest(1000)], [], 1000)
        manager.update([self.guest(1050), self.guest(1050, person_id="vision_guest_0002", azimuth=-30.0)], [], 1050)

        self.assertEqual(len(manager.tracks), 2)

    def test_an_unnamed_observation_still_relies_on_geometry(self):
        manager = ProbabilisticPersonManager(PersonManagerConfig(reidentification_retention_ms=1000))
        manager.update([self.guest(1000, person_id="unknown")], [], 1000)
        created = [event for event in manager.events if event["kind"] == "temporary_person_created"]
        manager.update([self.guest(9000, person_id="unknown", azimuth=-40.0)], [], 9000)
        created += [event for event in manager.events if event["kind"] == "temporary_person_created"]

        # Nothing identifies this observation, so eight seconds later it is a new
        # person as far as anyone can tell. That is the honest answer, not a bug.
        self.assertEqual(len(created), 2)
