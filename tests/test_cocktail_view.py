"""In a noisy room, one row per voice — with the right sentences under it."""
import unittest

from robot_attention_perception.cocktail_view import cocktail_rows


def turn(person, text, stamp, cluster=None, source="voice_cluster", addressed=False, held=False):
    return {"person_id": person, "voice_cluster_id": cluster, "identity_source": source,
            "identity_confidence": .7, "stamp_ms": stamp, "source": "microphone",
            "transcript": {"text": text}, "addressee": "robot" if addressed else "non_robot_or_unknown",
            "held_for_attention": held, "addressee_detail": {"score": .5}}


class CocktailViewTest(unittest.TestCase):
    def voice(self):
        return {"voice_registry": {"天行": {"display_name": "天行", "role": "owner", "pool_size": 3,
                                            "anchor_groups": 2, "aliases": ["stranger_aa"]},
                                   "stranger_bb": {"display_name": "陌生人1", "role": "stranger",
                                                   "pool_size": 1, "anchor_groups": 1, "aliases": []}},
                "speakers": {"speakers": [
                    {"speaker_id": "天行", "person_id": "天行", "display_name": "天行",
                     "embedding_digest": [.2, -.1, .4], "last_heard_ms": 10000, "speech_ms": 3000,
                     "identity_confidence": .8, "azimuth_deg": -12, "reference_available": True},
                    {"speaker_id": "stranger_bb", "person_id": "stranger_bb", "display_name": "陌生人1",
                     "embedding_digest": [.1, .3, -.2], "last_heard_ms": 9000, "speech_ms": 2000,
                     "identity_confidence": .4, "azimuth_deg": 30, "reference_available": False}]}}

    def test_each_voice_gets_its_own_sentences(self):
        rows = cocktail_rows(10000, self.voice(),
                             [turn("天行", "今天天气怎么样", 9800, source="face_recognition", addressed=True),
                              turn("stranger_bb", "下午几点走", 9000, held=True)],
                             {"target_id": "person:天行", "listen": True})
        self.assertEqual([row["display_name"] for row in rows], ["天行", "陌生人1"])
        self.assertEqual([u["text"] for u in rows[0]["utterances"]], ["今天天气怎么样"])
        self.assertTrue(rows[0]["utterances"][0]["addressed"])
        self.assertTrue(rows[0]["attended"])
        self.assertEqual(rows[0]["identity_source"], "face_recognition")
        self.assertTrue(rows[1]["utterances"][0]["held"])
        self.assertFalse(rows[1]["attended"])
        self.assertEqual(len(rows[0]["digest"]), 3)

    def test_a_cluster_that_was_merged_shows_up_under_the_person(self):
        """The alias must not appear as a second voice in the room.

        Once a face has said that stranger_aa is 天行, the cluster is folded
        into his pool; two rows for one person on screen would be exactly the
        confusion the merge exists to remove.
        """
        rows = cocktail_rows(10000, self.voice(),
                             [turn("天行", "刚才那个先放着", 9600, cluster="stranger_aa")],
                             {"target_id": "none", "listen": False})
        self.assertEqual(len(rows), 2)
        owner = next(row for row in rows if row["display_name"] == "天行")
        self.assertEqual([u["text"] for u in owner["utterances"]], ["刚才那个先放着"])

    def test_our_own_playback_and_stale_talk_are_not_people_in_the_room(self):
        echo = dict(turn("天行", "上海今天晴", 9900), source="robot_echo")
        old = turn("stranger_bb", "很久以前说的", 10000 - 300000)
        rows = cocktail_rows(10000, self.voice(), [echo, old], None)
        self.assertTrue(all(not row["utterances"] for row in rows))
        self.assertEqual(len(rows), 2)   # still listed: the tracker heard them

    def test_somebody_off_camera_with_no_track_still_gets_a_row(self):
        rows = cocktail_rows(10000, {"voice_registry": {}, "speakers": {"speakers": []}},
                             [turn("stranger_cc", "谁在那边", 9500)], None)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["display_name"], "stranger_cc")
        self.assertEqual(rows[0]["digest"], [])
