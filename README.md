# ZentroMart VoiceOps

Repository: https://github.com/shivammaurya0017/zentromart-voiceops

An educational customer-support agent using Gemini, section-aware hybrid RAG,
the supplied identity-gated order tools, and **Gnani-only STT/TTS**.

## Current status

Working implementation with a successful live microphone-to-Gnani-STT-to-agent-to-
Gnani-TTS test. All 52 baseline questions, 52 grounded questions and 96 conversation
turns completed. Initial full-run decision matches were 18/52 for the baseline,
48/52 for grounded RAG and 91/96 for conversations. All 25 local checks passed.
These are heuristic practice measurements, not official scores. A separate
post-fix seven-question recheck matched six decisions; original runs are preserved.
Some conversation replies still omit policy details or use the wrong decision
label. Live batch-CLI audio testing, final report, demo and publication remain pending.

## Requirements

- Python 3.10 or newer. No pip packages are needed.
- Gemini API key and your own Gnani key with STT and TTS scopes.
- The original extracted dataset, obtained separately; do not commit/package it.
- Internet connectivity for Gemini and Gnani API requests.

## Start the local demo

### For reviewers running on another computer

1. Download this repository using **Code > Download ZIP**, then extract it.
2. Install Python 3.10+ if needed. Open the extracted project folder in VS Code
   (or a terminal); no Live Server extension is needed.
3. Run `python server.py` in the terminal. On Windows, `py server.py` is another
   option when Python is installed through the Python launcher.
4. Open the printed localhost URL in Chrome or Edge. Keep the terminal running.
5. Enter your own Gemini and Gnani API keys. Set **Dataset folder** to the local
   extracted original dataset folder containing `kb`, `data`, `tools` and `practice`.
   The example D: path belongs to the developer and must be changed on your device.
6. Click **Connect keys**, then **Build document index** on the first run. Wait
   for **Semantic index ready**. Allow microphone access when recording.

The repository contains no credentials or original dataset. API access, available
quota and internet connectivity are required. GitHub hosts the source; it does
not run the Python backend. A normal fresh visit loads the current UI; hard
refresh is not a setup requirement. Responses explicitly disable browser caching.

### Local launch

On Windows, double-click `START_VOICEOPS.cmd` (uses installed Python or the local
Codex bundled Python when available). Otherwise run:

```sh
python server.py
```

Open http://127.0.0.1:8768. Paste each key into its password box, confirm the dataset
folder and select **Connect keys**, then **Build document index**. Ask a question
or record a short voice message. Keys remain in local process memory, are sent to
their respective providers for authentication, and are not saved by the demo.
The demo does not open private key files or load `.env` automatically. Anyone who
can inspect the running local process could access its memory. Close the server
after use. The demo is intended for one local user, not internet hosting.

The model is configurable; gemini-3.5-flash-lite was used in the current live test.
Connect checks actual text generation and can try supported text alternatives on
404/503 errors. Quota and authentication failures are not bypassed by model switching.
Gnani API requests explicitly identify this client as VoiceOps-Capstone/1.0; this
avoids the gateway 1010 rejection observed with Python's default User-Agent.

The index stores 768-dimensional Gemini text embeddings in a local JSON vector
index, searched by cosine similarity. It combines dense ranks with BM25 using
reciprocal-rank fusion. Chunk metadata preserves section IDs, status, effective
date and authority. Relevant policy docs are expanded to include nearby exceptions.
Superseded returns policy remains available only for the historical exception.
Public catalog-wide category summaries are derived from all product rows so broad
store questions do not depend on two individual search hits. This adds evidence
at retrieval time without rebuilding the existing vectors. It does not establish
live stock availability. Additional store-question checks are saved separately
from the original full practice runs in `sample_outputs/store_basics.json`.
Before index creation the UI offers a clearly labelled lexical preview. Build the
dense index for the final system. Embedding errors do not silently switch modes.

## Reproducible audio-folder run

Set `GEMINI_API_KEY` and `GNANI_API_KEY` as environment variables, or allow the
program to request hidden input. Never put keys in the run command itself.

```sh
python run.py --dataset "/path/to/Capstone_Project_Dataset_FTTT_2026" --input "test_audio" --output "sample_outputs/batch"
```

One folder per conversation, with zero-padded filenames in turn order:

```text
test_audio/
  conversation_01/01.wav
  conversation_01/02.wav
  conversation_02/01.wav
```

Root-level files are independent questions by default. Use `--conversation` to
treat all root-level files as turns of one conversation. Each file produces reply
text, `.reply.wav` speech and JSON containing retrieved chunks, actually cited
section IDs, sanitized tool calls and decision. A manifest records every input,
including failures. A failed file does not prevent later files from running; the
command exits nonzero if any input fails. No transcript/proof values are persisted.
Use `--language hi-IN --voice Nalini` for Hindi. Use clips under 60 seconds.

## Verification and dates

The supplied `lookup_order` is imported from the external dataset. Python gates
order details, enforces at most one verification retry, and removes verification
values from history and ticket summaries. A new conversation clears verification.
Ticket creation uses the supplied tool with its output relocated to `.runtime/`.
The model cannot perform cancellations, approve refunds or issue compensation.
Only claim handover after `created=True`. Dataset date is **20 October 2026**.
Business days skip Sundays and the supplied holidays. Return windows use calendar
days and current/historical/festive policy applicability.

## Offline checks

```sh
python -m unittest discover -s tests -v
```

## Practice evaluation

```sh
python evaluate.py --dataset "/path/to/dataset" --mode baseline --output "sample_outputs/baseline.json"
python evaluate.py --dataset "/path/to/dataset" --mode rag --output "sample_outputs/rag.json"
python evaluate.py --dataset "/path/to/dataset" --mode conversation --output "sample_outputs/conversations.json"
```

Evaluation is a transparent heuristic screen (fact strings, decisions, evidence,
forbidden phrases), not a semantic or native-language judge. Review failures and
report that limitation. Outputs are generated by the agent, never edited by hand.
Practice labels are consumed only by the evaluator, never the agent or index.

## Submission checklist (not yet complete)

- Public GitHub repository URL and exact run command.
- Real Step 1 versus Step 2 evaluation results.
- Maximum four-page report PDF.
- Maximum two-minute working demo, including follow-up and handover/unknown.
- Sample outputs and clean submission ZIP.
- Exclude `.env`, API keys, private key files, dataset copy, virtual environments,
  generated tickets, and local caches. Do not zip the whole working folder blindly.

Once report.pdf and demo.mp4 are present, run `python package_submission.py`.
The packager uses an explicit allowlist and stops if either required artifact is missing.
Use DEMO_SCRIPT.md to record a genuine spoken run with a follow-up and handover.

API references: https://ai.google.dev/api/generate-content,
https://ai.google.dev/gemini-api/docs/embeddings,
https://docs.gnani.ai/api/introduction/quick-start.
