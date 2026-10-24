import json
import unittest
from unittest.mock import patch, MagicMock
from src.api import quota_hint
from src.api import request, Gemini, APIError


class QuotaDiagnosticsTests(unittest.TestCase):
    def test_only_allowlisted_quota_metadata_is_reported(self):
        body = json.dumps({'error':{'message':'SECRET_KEY_DO_NOT_PRINT', 'details':[
            {'violations':[{'quotaId':'EmbedContentRequestsPerMinute-FreeTier','quotaValue':'0',
                            'quotaDimensions':{'project':'private-project'}}]},
            {'retryDelay':'31s'}]}}).encode()
        hint = quota_hint(body)
        self.assertIn('limit 0',hint)
        self.assertIn('31s',hint)
        self.assertNotIn('SECRET',hint)
        self.assertNotIn('private-project',hint)

    def test_invalid_json_is_not_echoed(self):
        self.assertEqual(quota_hint(b'PRIVATE KEY INVALID JSON'), '')

    def test_requests_identify_the_app_to_the_gateway(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"ok":true}'
        with patch('src.api.urlopen',return_value=response) as opened:
            self.assertEqual(request('https://example.com',{},{}),{'ok':True})
        sent = opened.call_args.args[0]
        self.assertEqual(sent.get_header('User-agent'),'VoiceOps-Capstone/1.0')

    def test_unavailable_generation_model_uses_supported_alternative(self):
        client = Gemini('FAKE-TEST-KEY')
        with patch.object(client,'generate',side_effect=[APIError('API HTTP 503: unavailable'),{'reply':'connected'}]):
            selected = client.check_generation(['gemini-3.8-flash','gemini-3.5-flash-lite'])
        self.assertEqual(selected,'gemini-3.5-flash-lite')

    def test_quota_error_does_not_try_other_models(self):
        client = Gemini('FAKE-TEST-KEY')
        with patch.object(client,'generate',side_effect=APIError('API HTTP 429: quota')) as generated:
            with self.assertRaises(APIError):
                client.check_generation(['gemini-3.8-flash','gemini-3.5-flash-lite'])
            self.assertEqual(generated.call_count,1)
