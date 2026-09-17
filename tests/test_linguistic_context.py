import unittest

from robot_attention_perception.linguistic_context import LinguisticContext, addressed_human


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.context = LinguisticContext()

    def features(self, text, **kwargs):
        kwargs.setdefault('known_people', ('小天', '小林'))
        return self.context.features(5000, text, **kwargs)

    def test_a_name_call_is_only_at_the_start(self):
        self.assertEqual(self.features('小圆，今天几点了？')['lang_directed_call'], 1.)
        # 识别器把「小圆」写成同音字的概率不低，名字叫不动机器人比误触更糟
        for homophone in ('小园，今天几点了？', '小元，今天几点了？'):
            self.assertEqual(self.features(homophone)['lang_directed_call'], 1., homophone)
        mention = self.features('这个机器人挺好玩的')
        self.assertEqual(mention['lang_directed_call'], 0.)
        self.assertEqual(mention['mentions_robot_without_calling'], 1.)

    def test_a_named_human_is_evidence_against_the_robot(self):
        result = self.features('小天你觉得呢？', speaker_id='小林')
        self.assertEqual(result['lang_addresses_other'], 1.)
        self.assertEqual(result['addressed_human_id'], '小天')
        self.assertEqual(result['lang_topic_continuation'], 0.)
        # Speaking one's own name is not addressing somebody else.
        self.assertEqual(self.features('小天觉得可以', speaker_id='小天')['lang_addresses_other'], 0.)

    def test_unknown_speakers_leave_the_feature_at_zero_instead_of_guessing(self):
        self.assertEqual(self.context.features(5000, '小张你觉得呢？', '', ('小天',))['lang_addresses_other'], 0.)
        self.assertEqual(addressed_human('小张你觉得呢？', ()), (0., ''))

    def test_answer_shape_needs_the_question_or_an_answer_opening(self):
        self.assertGreater(self.features('是的，可以')['lang_answer_continuation'], 0.)
        self.assertEqual(self.features('那我先走了')['lang_answer_continuation'], 0.)
        self.assertGreater(self.features('周六去公园吧', open_question='你想周六去公园吗？')['lang_answer_continuation'], 0.)

    def test_questions_and_backchannels_are_labelled(self):
        self.assertEqual(self.features('现在几点？')['lang_question'], 1.)
        self.assertEqual(self.features('是这样吗')['lang_question'], 1.)
        self.assertEqual(self.features('嗯')['lang_backchannel'], 1.)
        self.assertEqual(self.features('嗯，我在想这件事')['lang_backchannel'], 0.)

    def test_topic_continuation_only_follows_exchanges_the_robot_was_part_of(self):
        self.context.observe([
            {'utterance_id': 'x', 'stamp_ms': 4000, 'transcript': {'text': '周六去公园好不好'},
             'attention': {'addressed_to_robot': True}},
            {'utterance_id': 'y', 'stamp_ms': 4200, 'transcript': {'text': '我想吃火锅'},
             'attention': {'addressed_to_robot': False}}])
        self.assertGreater(self.features('周六去公园吧')['lang_topic_continuation'], 0.)
        self.assertEqual(self.features('我想吃火锅')['lang_topic_continuation'], 0.)

    def test_an_empty_or_stale_utterance_produces_nothing(self):
        self.assertEqual(self.features(''), {})
        self.context.observe([{'utterance_id': 'old', 'stamp_ms': 0, 'transcript': {'text': '周六去公园好不好'},
                               'attention': {'addressed_to_robot': True}}])
        self.assertEqual(self.context.features(200000, '周六去公园吧', '', ())['lang_topic_continuation'], 0.)


if __name__ == '__main__':
    unittest.main()
