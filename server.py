"""Local-only demo. Keys live in process memory, not saved to disk."""
import base64
import json
import os
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from src.api import Gemini, Gnani, APIError
from src.retrieval import Retriever
from src.agent import Agent, ToolLayer
from src.retrieval import read_csv
import importlib
import evaluate as evaluation
importlib.reload(evaluation)
facts_score = evaluation.facts_score
from run import DEFAULT_DATASET

ROOT = Path(__file__).resolve().parent
TOKEN = secrets.token_urlsafe(32)
STATE = {}
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Never log requests, transcripts or keys.

    def send(self, value, status=200, kind='application/json'):
        payload = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == '/':
            html = (ROOT / 'src/ui.html').read_text(encoding='utf-8').replace('__TOKEN__', TOKEN)
            self.send(html.encode(), kind='text/html; charset=utf-8')
        elif self.path == '/status':
            self.send({'configured':bool(STATE), 'dense':bool(STATE and STATE['retriever'].vectors),
                       'chunks':len(STATE['retriever'].chunks) if STATE else 0,
                       'model':STATE['client'].model if STATE else None})
        else:
            self.send({'error':'Not found'}, 404)

    def do_POST(self):
        if self.headers.get('X-VoiceOps-Token') != TOKEN:
            return self.send({'error':'Reload the local page before continuing.'}, 403)
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length > 25_000_000:
                return self.send({'error':'Recording too large; keep it below 60 seconds.'}, 413)
            data = json.loads(self.rfile.read(length))
            with LOCK:
                if self.path == '/configure':
                    dataset = data.get('dataset','').strip() or DEFAULT_DATASET
                    if not (Path(dataset) / 'kb').is_dir():
                        raise ValueError('Dataset folder not found. Use the extracted folder containing kb and data.')
                    client = Gemini(data['gemini'].strip(), data.get('model') or 'gemini-3.5-flash-lite')
                    models = client.models()  # Check credentials without exposing them.
                    generation_warning = None
                    try:
                        client.check_generation(models)
                    except APIError as exc:
                        # Voice can be tested independently of Gemini's transient availability.
                        generation_warning = 'Gemini text generation: ' + str(exc)
                    retriever = Retriever(dataset, ROOT / '.cache/index.json')
                    tools = ToolLayer(dataset, ROOT / '.runtime')
                    STATE.clear()
                    STATE.update(client=client, gnani=Gnani(data['gnani'].strip()), retriever=retriever, tools=tools)
                    STATE['agent'] = Agent(client, retriever, tools)
                    return self.send({'ok':True, 'model':client.model, 'chunks':len(retriever.chunks),
                                      'dense':bool(retriever.vectors), 'warning':generation_warning})
                if not STATE:
                    raise ValueError('Connect your keys first.')
                if self.path == '/index':
                    STATE['retriever'].build(STATE['client'])
                    return self.send({'ok':True, 'chunks':len(STATE['retriever'].chunks)})
                if self.path == '/speech-test':
                    audio = STATE['gnani'].speak('Hello. I am your ZentroMart AI support assistant.')
                    try:
                        transcript = STATE['gnani'].transcribe(audio)
                        listening = 'success' if transcript else 'empty_transcript'
                    except APIError as e:
                        listening = str(e)
                    return self.send({'speaking':'success', 'listening':listening,
                                      'audio':base64.b64encode(audio).decode()})
                if self.path == '/practice':
                    mode = data.get('mode','rag')
                    if mode not in ('baseline','rag','conversation'):
                        raise ValueError('Invalid evaluation mode')
                    conversation = mode == 'conversation'
                    filename = 'conversations.csv' if conversation else 'questions.csv'
                    rows = read_csv(STATE['retriever'].dataset / 'practice' / filename)
                    row = next((row for row in rows if (
                        (row['conversation_id'] == data['id'] and str(row['turn']) == str(data['turn']))
                        if conversation else (row['id'] == data['id']))), None)
                    if row is None:
                        raise ValueError('Practice case not found')
                    agents = STATE.setdefault('practice_agents',{})
                    identity = (mode, row['conversation_id'] if conversation else row['id'])
                    if not conversation or str(row['turn']) == '1' or identity not in agents:
                        agents[identity] = Agent(STATE['client'],STATE['retriever'],STATE['tools'],baseline=mode=='baseline')
                    agent = agents[identity]
                    message = row['customer_message'] if conversation else row['question']
                    result = agent.respond(message)
                    return self.send({'id':data['id'], 'turn':row.get('turn'), 'mode':mode,
                                      'model':STATE['client'].model,
                                      'result':result, 'checks':facts_score(row,result)})
                if self.path == '/reset':
                    STATE['agent'] = Agent(STATE['client'], STATE['retriever'], STATE['tools'])
                    return self.send({'ok':True})
                if self.path == '/chat':
                    result = STATE['agent'].respond(data['text'])
                elif self.path == '/voice':
                    audio = base64.b64decode(data['audio'], validate=True)
                    heard = STATE['gnani'].transcribe(audio, data.get('filename','recording.wav'), data.get('language','en-IN'))
                    result = STATE['agent'].respond(heard) if heard else {
                        'reply':"Sorry, I didn't catch that. Could you say it again?", 'decision':'clarify',
                        'used_sources':[], 'retrieved_chunks':[], 'tool_calls':[]}
                    # Voice transcripts can contain verification secrets; omit them from UI/log output.
                else:
                    return self.send({'error':'Not found'}, 404)
                if data.get('speak', True):
                    try:
                        result['audio'] = base64.b64encode(STATE['gnani'].speak(result['reply'],
                            data.get('language','en-IN'), data.get('voice','Kaveri'))).decode()
                    except APIError as e:
                        result['voice_error'] = str(e)
                return self.send(result)
        except (APIError, ValueError, KeyError) as e:
            # APIError is intentionally sanitized upstream; never serialize arbitrary exceptions.
            msg = str(e) if isinstance(e, APIError) else 'Check the selected settings and try again. The operation could not finish.'
            return self.send({'error':msg}, 400)
        except Exception:
            return self.send({'error':'Local operation failed. No secret values were logged. Please retry.'}, 500)


if __name__ == '__main__':
    address = ('127.0.0.1', int(os.getenv('VOICEOPS_PORT', '8768')))
    server = ThreadingHTTPServer(address, Handler)
    print(f'VoiceOps is running at http://{address[0]}:{address[1]}')
    print('Keys are entered in the local browser and held in memory. Close this window to stop.')
    threading.Timer(1, lambda: webbrowser.open(f'http://{address[0]}:{address[1]}')).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
