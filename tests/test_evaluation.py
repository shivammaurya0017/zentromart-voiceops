import unittest
from evaluate import facts_score, canonical


class EvaluationTests(unittest.TestCase):
    def test_spoken_numbers_and_native_digits_are_accepted(self):
        self.assertIn('79',canonical('seventy-nine rupees'))
        self.assertIn('5 to 7',canonical('five to seven business days'))
        self.assertIn('5000',canonical('५,००० rupees'))
    def test_number_in_sku_is_not_a_warranty_duration(self):
        row = {'must_include':'12','expected_decision':'answer'}
        result = {'reply':'ELE-0012 is covered.','decision':'answer', 'retrieved_chunks':[]}
        self.assertEqual(facts_score(row,result)['required_fact_hits'],0)
        result['reply'] = 'ELE-0012 has 12 months of cover.'
        self.assertEqual(facts_score(row,result)['required_fact_hits'],1)

    def test_expected_tool_names_are_compared_without_verification_secrets(self):
        row = {'expected_decision':'clarify','expected_tool_calls':'lookup_order(order_id=89782837)'}
        result = {'reply':'Please verify.','decision':'clarify','retrieved_chunks':[],
                  'tool_calls':[{'name':'lookup_order','order_id':'89782837'}]}
        self.assertTrue(facts_score(row,result)['tool_names_match'])
