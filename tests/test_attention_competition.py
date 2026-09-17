import unittest,math
from robot_attention_perception.attention_competition import AttentionCompetition,Candidate

class CompetitionTests(unittest.TestCase):
    def test_order_independence_and_separate_modality_budgets(self):
        a,b=AttentionCompetition(),AttentionCompetition()
        for t in range(0,2001,100):
            c=[Candidate('person','visual',t,.8),Candidate('screen','visual',t,.7),Candidate('noise','audio',t,1.)]
            x=a.update(t,c);y=b.update(t,list(reversed(c)))
        self.assertEqual(x,y)
        for m in ['visual','audio']:
            self.assertLessEqual(sum(v['a'] for v in x['candidates'].values() if v['modality']==m),1.)
        self.assertGreater(x['candidates']['person']['a'],.1)
    def test_habituation_matches_example_and_rest_recovers(self):
        a=AttentionCompetition()
        for t in range(0,20001,100):a.update(t,[Candidate('tv','visual',t,.8)])
        self.assertAlmostEqual(a.state['tv']['h'],1-math.exp(-2),places=4)
        for t in range(20100,50001,100):a.update(t,[])
        self.assertAlmostEqual(a.state['tv']['h'],(1-math.exp(-2))*math.exp(-1),places=4)
        self.assertIsNone(a.focus['visual'])
    def test_change_dedup_feedback_and_stale_direction(self):
        a=AttentionCompetition();a.update(0,[Candidate('a','visual',0,1,change_id='event')])
        a.update(1000,[Candidate('a','visual',1000,1,change_id='event')])
        self.assertLess(a.state['a']['c'],1)
        self.assertTrue(a.observe_result('r','a','identity','normal',True,False))
        self.assertFalse(a.observe_result('r','a','identity','normal',True,False))
        self.assertFalse(a.observe_result('cancelled','a','identity','normal',False,False))
        self.assertAlmostEqual(a.learning('a'),1/3)
        a.update(2000,[Candidate('a','visual',0,1)])
        self.assertFalse(a.state['a']['valid']);self.assertIsNone(a.focus['visual'])
    def test_switch_margin_and_return_inhibition(self):
        a=AttentionCompetition()
        for t in range(0,2000,100):a.update(t,[Candidate('a','visual',t,.8,goal=1),Candidate('b','visual',t,.2)])
        self.assertEqual(a.focus['visual'],'a')
        for t in range(2000,4000,100):a.update(t,[Candidate('a','visual',t,.2),Candidate('b','visual',t,.8,goal=1)])
        self.assertEqual(a.focus['visual'],'b');self.assertGreater(a.state['a']['r'],0)
    def test_time_resolution_consistency(self):
        values=[]
        for step in [20,50,100]:
            a=AttentionCompetition()
            for t in range(0,5001,step):a.update(t,[Candidate('p','visual',t,.6,goal=.8)])
            values.append(a.state['p']['a'])
        self.assertLess(max(values)-min(values),.015)
