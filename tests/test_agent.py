import tempfile
import unittest
from pathlib import Path
from src.agent import Agent, ToolLayer, date_facts, normalize_numbers, supplied_refund_dates, required_handover
from src.retrieval import Retriever
from run import DEFAULT_DATASET


class FakeLLM:
    """No network: test Python gates independently of model compliance."""
    def generate(self, system, user):
        return {'reply':'Verified. How can I help with this order?', 'decision':'answer',
                'used_sources':['RET-2'], 'ticket':None}


class EscalationLLM:
    def generate(self, system, user):
        if 'Rewrite the draft' in system:
            import json
            ticket_id = json.loads(user)['ticket_id']
            return {'reply':'I created a human handover ticket: ' + ticket_id}
        return {'reply':'This needs human review.', 'decision':'escalate', 'used_sources':['ESC-1'],
                'ticket':{'summary':'Customer requests human support. Intent: human handover. No order or private details checked. Escalation reason: explicit human request.',
                          'reason':'human_request','intent':'support'}}


@unittest.skipUnless(Path(DEFAULT_DATASET).is_dir(), 'External dataset is required')
class AgentTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[3] / 'work' / 'test_runs'
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.retriever = Retriever(DEFAULT_DATASET, Path(self.tmp.name)/'index.json')
        self.tools = ToolLayer(DEFAULT_DATASET,self.tmp.name)
        self.agent = Agent(FakeLLM(),self.retriever,self.tools)

    def tearDown(self):
        self.tmp.cleanup()

    def test_unverified_lookup_discloses_no_order_details(self):
        result = self.agent.respond('Order 8978 2837')
        self.assertIsNone(self.agent.order)
        self.assertEqual(result['decision'],'clarify')
        self.assertNotIn('items',result['tool_calls'][0])
        self.assertNotIn('total_inr',result['tool_calls'][0])

    def test_two_failed_proofs_lock_order(self):
        self.agent.respond('89782837')
        self.agent.respond('0000')
        self.agent.respond('1111')
        result = self.agent.respond('4533')
        self.assertEqual(result['decision'],'refuse')
        self.assertIsNone(self.agent.order)

    def test_valid_proof_removed_from_history_and_tool_trace(self):
        self.agent.respond('89782837')
        result = self.agent.respond('4533')
        self.assertTrue(self.agent.order['verified'])
        self.assertNotIn('4533',str(self.agent.history))
        self.assertNotIn('4533',str(result))
        self.assertNotIn('delivery_pincode',self.agent.order)

    def test_verification_does_not_act_on_unrequested_background_damage(self):
        agent = Agent(EscalationLLM(),self.retriever,self.tools)
        agent.respond('Order number hai 89782837')
        result = agent.respond('Last four digits 4533 hain')
        self.assertEqual(result['decision'],'answer')
        self.assertEqual([c['name'] for c in result['tool_calls']],['lookup_order'])
        self.assertIn('IDV-2',{c['id'] for c in result['retrieved_chunks']})

    def test_damage_handover_threshold_is_read_from_policy_not_product_price(self):
        sources = self.retriever.search('I received ELE-0023 and it has a crack')
        sources += [c for c in self.retriever.chunks if c['id']=='ESC-1']
        reason = required_handover('I received ELE-0023 and it has a crack',sources)
        self.assertIn('more than 10000',reason)
        self.assertIn('45999',reason)

    def test_new_order_clears_old_verification(self):
        self.agent.respond('89782837')
        self.agent.respond('4533')
        self.agent.respond('18599253')
        self.assertIsNone(self.agent.order)

    def test_return_deadline_calendar_and_festival(self):
        order = {'delivered_on':'2026-10-12','placed_at':'2026-09-30T00:00',
                 'items':[{'sku':'A','title':'Shoes','return_category':'fashion'}]}
        self.assertEqual(date_facts(order,set())['return_deadlines'][0]['deadline'],'2026-10-27')
        order['placed_at'] = '2026-10-01T00:00'
        self.assertEqual(date_facts(order,set())['return_deadlines'][0]['deadline'],'2026-11-11')

    def test_business_days_exclude_sunday_and_holiday(self):
        order = {'cancellation_refund_initiated_on':'2026-10-17','cancellation_refund_method':'upi'}
        dates = date_facts(order,{'2026-10-19'})['refunds'][0]
        self.assertEqual(dates['earliest'],'2026-10-20')
        self.assertEqual(dates['latest'],'2026-10-22')

    def test_customer_supplied_refund_dates_need_no_order_lookup(self):
        facts = supplied_refund_dates('credit card wala refund 17 October 2026 ko shuru hua',{'2026-10-19'})
        self.assertEqual(facts['computed'][0]['earliest'],'2026-10-24')
        self.assertEqual(facts['computed'][0]['latest'],'2026-10-27')

    def test_retrieval_preserves_current_policy_and_ids(self):
        sources = self.retriever.search('electronics return window days')
        self.assertTrue(any(c['id']=='RET-2' and c['status']=='current' for c in sources))

    def test_broad_catalog_evidence_covers_categories_without_invented_products(self):
        import json
        from src.retrieval import read_csv
        sources = self.retriever.search('What kind of products do you sell?')
        summary = next(c for c in sources if c['id']=='products.csv:overview')
        categories = json.loads(summary['text'])['categories_and_example_products']
        rows = read_csv(Path(DEFAULT_DATASET)/'data/products.csv')
        self.assertEqual(set(categories),{r['return_category'] for r in rows})
        for category,examples in categories.items():
            names = {r['title'].split(',')[0].strip() for r in rows if r['return_category']==category}
            self.assertTrue(examples)
            self.assertTrue(set(examples)<=names)

    def test_exact_product_cannot_crowd_out_warranty_policy(self):
        sources = self.retriever.search('SKU ELE-0012 ZentroMart Assured claim kaise karein')
        self.assertIn('products.csv:ELE-0012',{c['id'] for c in sources})
        self.assertTrue(any(c['id']=='WAR-2' for c in sources))
        self.assertLessEqual(sum(c.get('doc_type')=='catalog' for c in sources),1)

    def test_spoken_order_digits(self):
        self.assertIn('8 9 7 8 2 8 3 7',normalize_numbers('eight nine seven eight two eight three seven'))

    def test_handover_creates_actual_ticket(self):
        agent = Agent(EscalationLLM(),self.retriever,self.tools)
        result = agent.respond('I want a human agent')
        self.assertEqual(result['decision'],'escalate')
        call = result['tool_calls'][0]
        self.assertTrue(call['created'])
        self.assertIn(call['ticket_id'],result['reply'])
        self.assertTrue((Path(self.tmp.name)/'tickets_created.csv').exists())

    def test_failed_ticket_does_not_claim_handover(self):
        agent = Agent(EscalationLLM(),self.retriever,self.tools)
        agent.tools.ticket = lambda **kwargs: {'created':False,'error':'test failure'}
        result = agent.respond('I want a human agent')
        self.assertIn('could not create',result['reply'])
        self.assertNotEqual(result['decision'],'escalate')


if __name__ == '__main__':
    unittest.main()
