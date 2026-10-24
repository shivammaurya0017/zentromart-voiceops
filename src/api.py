"""Small standard-library clients. Secrets never enter URLs or exception text."""
import json
import mimetypes
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class APIError(RuntimeError):
    pass


def quota_hint(body):
    """Only expose numeric quota limits and standard identifiers, never error messages/keys."""
    import re
    try:
        details = json.loads(body).get('error', {}).get('details', [])
    except (ValueError, AttributeError):
        return ''
    hints = []
    for item in details:
        for violation in item.get('violations', []):
            name = violation.get('quotaId', '')
            limit = str(violation.get('quotaValue', ''))
            if name and re.fullmatch(r'[A-Za-z0-9_.\-/]{1,160}', name):
                hints.append(name + (' (limit ' + limit + ')' if limit.isdigit() else ''))
        delay = item.get('retryDelay', '')
        if isinstance(delay, str) and re.fullmatch(r'[0-9.]+s', delay):
            hints.append('Suggested retry after ' + delay)
    return '; '.join(dict.fromkeys(hints))


def request(url, headers, payload=None, raw=False, method=None):
    headers = {'User-Agent':'VoiceOps-Capstone/1.0', 'Accept':'application/json', **headers}
    if isinstance(payload, dict):
        payload = json.dumps(payload).encode()
        headers = {**headers, 'Content-Type': 'application/json'}
    for attempt in range(3):
        try:
            with urlopen(Request(url, data=payload, headers=headers, method=method), timeout=90) as r:
                data = r.read()
                return data if raw else json.loads(data)
        except HTTPError as e:
            status = e.code
            error_body = e.read()
            detail = quota_hint(error_body) if status == 429 else ''
            if status == 403 and b'1010' in error_body:
                raise APIError('The API gateway blocked this request before authentication (gateway 1010).') from None
            if status == 429 and '(limit 0)' in detail:
                raise APIError('API HTTP 429: This project has zero quota for the requested operation. ' + detail) from None
            if status in (429, 500, 502, 503) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            hint = {400: 'Request format or selected model is unsupported.',
                    401: 'API key was not accepted.', 403: 'API access is not enabled for this key/project.',
                    404: 'Selected model or endpoint is unavailable.',
                    429: 'Quota/rate limit reached. Wait or check your account quota.'}.get(status, 'Service request failed.')
            raise APIError(f'API HTTP {status}: {hint}' + (' ' + detail if detail else '')) from None
        except (URLError, TimeoutError):
            raise APIError('Could not reach the API. Check internet access and try again.') from None


class Gemini:
    base = 'https://generativelanguage.googleapis.com/v1beta/'

    def __init__(self, key, model='gemini-3.8-flash'):
        self.headers = {'x-goog-api-key': key}
        self.model = model.removeprefix('models/')
        self.embedding_model = 'gemini-embedding-2'

    def models(self):
        data = request(self.base + 'models?pageSize=1000', self.headers)
        return [m['name'].removeprefix('models/') for m in data.get('models', [])
                if 'generateContent' in m.get('supportedGenerationMethods', [])]

    def check_generation(self, available):
        """Try supported text models on unavailability, never bypass account/quota errors."""
        preferred = [self.model, 'gemini-3.5-flash-lite', 'gemini-3.6-flash', 'gemini-3.1-flash-lite']
        other = sorted((m for m in available if 'flash' in m and not any(
            excluded in m for excluded in ('image','tts','live','audio','robotics','computer','omni'))), reverse=True)
        choices = list(dict.fromkeys(m for m in preferred + other if m in available))[:4]
        if not choices:
            raise APIError('No suitable Flash text model is available for this account.')
        last_error = None
        for model in choices:
            self.model = model
            try:
                self.generate('Return JSON with a reply field.', 'Say connected.')
                return model
            except APIError as error:
                last_error = error
                if not any(code in str(error) for code in ('HTTP 404','HTTP 503')):
                    raise
        raise APIError('The available text models did not pass the connection check. ' + str(last_error)) from None

    def generate(self, system, user):
        data = request(self.base + 'models/' + self.model + ':generateContent', self.headers, {
            'systemInstruction': {'parts': [{'text': system}]},
            'contents': [{'role': 'user', 'parts': [{'text': user}]}],
            'generationConfig': {'temperature': 0, 'maxOutputTokens': 4096,
                                 'responseMimeType': 'application/json'}})
        try:
            text = ''.join(p.get('text', '') for p in data['candidates'][0]['content']['parts']
                           if not p.get('thought'))
            return json.loads(text.strip().removeprefix('```json').removesuffix('```').strip())
        except (KeyError, IndexError, ValueError):
            raise APIError('The model did not return a complete structured answer. Please retry.') from None

    def embed(self, texts, task='RETRIEVAL_DOCUMENT'):
        model = 'models/' + self.embedding_model
        out = []
        for start in range(0, len(texts), 8):
            batch = texts[start:start + 8]
            requests = []
            for text in batch:
                entry = {'model':model, 'content':{'parts':[{'text':text}]}, 'outputDimensionality':768}
                if self.embedding_model == 'gemini-embedding-001':
                    entry['taskType'] = task
                requests.append(entry)
            data = request(self.base + model + ':batchEmbedContents', self.headers, {
                'requests': requests})
            out.extend(e['values'] for e in data['embeddings'])
            if start + 8 < len(texts):
                time.sleep(2)
        return out


class Gnani:
    base = 'https://api.vachana.ai'

    def __init__(self, key):
        self.headers = {'X-API-Key-ID': key}

    def transcribe(self, audio, filename='recording.wav', language='en-IN'):
        boundary = 'VoiceOps' + uuid.uuid4().hex
        # Filename comes from our own file name, never interpolated into a shell.
        mime = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        safe_name = Path(filename).name.replace('"', '').replace('\r', '').replace('\n', '')
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="language_code"\r\n\r\n{language}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="format"\r\n\r\ntranscribe\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="audio_file"; filename="{safe_name}"\r\n'
                f'Content-Type: {mime}\r\n\r\n').encode() + audio + f'\r\n--{boundary}--\r\n'.encode()
        try:
            data = request(self.base + '/stt/v3', {**self.headers,
                           'Content-Type': 'multipart/form-data; boundary=' + boundary}, body)
        except APIError as e:
            raise APIError('Gnani listening (STT): ' + str(e)) from None
        if data.get('success') is False:
            raise APIError('Gnani could not transcribe this recording.')
        return data.get('transcript', '').strip()

    def speak(self, text, language='en-IN', voice='Kaveri'):
        try:
            return request(self.base + '/api/v1/tts/inference', self.headers, {
                'text': text, 'voice': voice, 'model': 'timbre-v2.5', 'language': language,
                'speed': 1.0, 'audio_config': {'sample_rate': 24000, 'num_channels': 1,
                'sample_width': 2, 'encoding': 'linear_pcm', 'container': 'wav'}}, raw=True)
        except APIError as e:
            raise APIError('Gnani speaking (TTS): ' + str(e)) from None
