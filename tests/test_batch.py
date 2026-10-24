import io
import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch
import run
from src.api import APIError


class FakeGemini:
    def __init__(self,*args,**kwargs):
        pass

    def embed(self,texts,*args):
        return [[0.0]*768 for _ in texts]

    def generate(self,*args):
        return {'reply':'Wallet refunds arrive within two hours after initiation.',
                'decision':'answer','used_sources':['REF-2'],'ticket':None}


class FakeGnani:
    def __init__(self,*args):
        pass

    def transcribe(self,audio,*args):
        if audio == b'FAIL':
            raise APIError('Synthetic STT failure for test')
        return 'How fast are wallet refunds?'

    def speak(self,*args):
        buffer = io.BytesIO()
        with wave.open(buffer,'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b'\0\0'*100)
        return buffer.getvalue()


@unittest.skipUnless(Path(run.DEFAULT_DATASET).is_dir(),'External dataset is required')
class BatchTests(unittest.TestCase):
    def test_failed_file_is_recorded_and_later_input_still_produces_text_audio_and_sources(self):
        scratch = Path(__file__).resolve().parents[3]/'work/test_runs'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            source,output = Path(temp)/'inputs',Path(temp)/'outputs'
            source.mkdir()
            (source/'01.wav').write_bytes(b'FAIL')
            (source/'02.wav').write_bytes(b'valid test input')
            argv = ['run.py','--input',str(source),'--output',str(output),'--lexical-preview']
            with patch.dict(os.environ,{'GEMINI_API_KEY':'FAKE','GNANI_API_KEY':'FAKE'}), \
                 patch('sys.argv',argv),patch('run.Gemini',FakeGemini),patch('run.Gnani',FakeGnani), \
                 patch('sys.stdout',new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as exited:
                    run.main()
            self.assertEqual(exited.exception.code,1)
            manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(len(manifest),2)
            self.assertIn('error',manifest[0])
            self.assertEqual(manifest[1]['audio_status'],'success')
            self.assertTrue((output/'02.wav.txt').is_file())
            with wave.open(str(output/'02.wav.reply.wav')) as audio:
                self.assertEqual(audio.getnchannels(),1)
            saved = json.loads((output/'02.wav.json').read_text(encoding='utf-8'))
            self.assertIn('REF-2',saved['used_sources'])
            self.assertTrue(saved['retrieved_chunks'])
