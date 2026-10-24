"""Batch evaluator entrypoint. No third-party Python dependencies required."""
import argparse
import getpass
import json
import os
from pathlib import Path
from src.api import Gemini, Gnani, APIError
from src.retrieval import Retriever
from src.agent import Agent, ToolLayer

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = r'D:\nlp capstone Dataset\Capstone_Project_Dataset_FTTT_2026'


def main():
    p = argparse.ArgumentParser(description='ZentroMart: audio folder -> grounded text, sources and Gnani speech')
    p.add_argument('--dataset', default=os.getenv('VOICEOPS_DATASET', DEFAULT_DATASET))
    p.add_argument('--input', required=True, help='Audio folder; subfolders are separate conversations')
    p.add_argument('--output', default='sample_outputs/batch')
    p.add_argument('--model', default=os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite'))
    p.add_argument('--language', default='en-IN')
    p.add_argument('--voice', default='Kaveri')
    p.add_argument('--conversation', action='store_true', help='Treat all root-level files as ordered turns in one conversation')
    p.add_argument('--lexical-preview', action='store_true', help='Skip embeddings for diagnostics only; not the final RAG system')
    args = p.parse_args()
    gemini = Gemini(os.getenv('GEMINI_API_KEY') or getpass.getpass('Gemini API key (hidden): '), args.model)
    gnani = Gnani(os.getenv('GNANI_API_KEY') or getpass.getpass('Gnani API key (hidden): '))
    retriever = Retriever(args.dataset, ROOT / '.cache/index.json')
    if not retriever.vectors and not args.lexical_preview:
        print('Building vector index (first run only)...')
        retriever.build(gemini)
    tools = ToolLayer(args.dataset, ROOT / '.runtime')
    source, output = Path(args.input), Path(args.output)
    if not source.is_dir():
        p.error('--input must be an existing folder')
    formats = {'.wav','.mp3','.ogg','.flac','.aac','.m4a','.webm'}
    files = sorted(f for f in source.rglob('*') if f.suffix.lower() in formats and f.is_file())
    agents, records = {}, []
    for f in files:
        rel = f.relative_to(source)
        session = str(rel.parent) if rel.parent != Path('.') else ('root' if args.conversation else str(rel))
        agent = agents.setdefault(session, Agent(gemini, retriever, tools))
        dest = output / rel.parent
        dest.mkdir(parents=True, exist_ok=True)
        # Include the input extension to avoid collisions between turn.wav and turn.mp3.
        base = dest / f.name
        record = {'input':str(rel), 'session':session}
        try:
            heard = gnani.transcribe(f.read_bytes(), f.name, args.language)
            result = (agent.respond(heard) if heard else {'reply':"Sorry, I didn't catch that. Could you say it again?",
                'decision':'clarify','used_sources':[], 'retrieved_chunks':[], 'tool_calls':[]})
            record.update(result)
            record['audio_status'] = 'pending'
            Path(str(base) + '.txt').write_text(result['reply'], encoding='utf-8')
            try:
                Path(str(base) + '.reply.wav').write_bytes(gnani.speak(result['reply'], args.language, args.voice))
                record['audio_status'] = 'success'
            except APIError as e:
                record['audio_status'], record['audio_error'] = 'failed', str(e)
        except (APIError, ValueError) as e:
            record['error'] = str(e)
        Path(str(base) + '.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        records.append(record)
        print(f'{rel}: ' + ('FAILED' if record.get('error') or record.get('audio_status') == 'failed' else 'OK'))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    if not files:
        print('No audio files found.')
    raise SystemExit(1 if not files or any(r.get('error') or r.get('audio_status') == 'failed' for r in records) else 0)


if __name__ == '__main__':
    main()
