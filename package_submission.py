"""Build a clean final ZIP from an allowlist; never include datasets or key files."""
import argparse
import re
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',default='shivam_maurya_voiceops.zip')
    args = parser.parse_args()
    if Path(args.name).name != args.name or not args.name.endswith('.zip'):
        parser.error('--name must be a simple ZIP filename')
    for required in ('report.pdf','demo.mp4'):
        if not (root/required).is_file():
            parser.error(f'{required} is missing. Finish the report/demo before making the final ZIP.')
    files = [root/name for name in ('README.md','.env.example','.gitignore','requirements.txt','run.py','server.py',
             'evaluate.py','START_VOICEOPS.cmd','DEMO_SCRIPT.md','report.pdf','demo.mp4')]
    files += [p for p in (root/'src').rglob('*') if p.suffix in ('.py','.html')]
    files += [p for p in (root/'tests').glob('*.py')]
    files += [p for p in (root/'sample_outputs').rglob('*') if p.is_file() and p.suffix in ('.json','.txt','.wav')]
    files = [p for p in files if p.is_file()]
    for file in files:
        if file.suffix in ('.md','.py','.html','.txt','.json') or file.name == '.env.example':
            text = file.read_text(encoding='utf-8-sig')
            if re.search(r'AIza[A-Za-z0-9_-]{35}',text):
                parser.error('A possible Gemini API key was detected. Packaging stopped; no value was printed.')
    target = root/args.name
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as bundle:
        for file in files:
            bundle.write(file,file.relative_to(root))
    print(f'Created {target.name} with {len(files)} allowlisted files. Dataset and private settings excluded.')


if __name__ == '__main__':
    main()
