# Local Chat Memory & Recovery

A small Python terminal app that turns a pasted chatbot conversation into a compact, readable recovery note and opens it in Notepad. It runs locally, stores conversations and extracted points in SQLite, and does not send data to ChatGPT, Gemini, or another hosted service.

## Run on Windows

Use Python 3.10 or newer. From PowerShell in this folder:

```powershell
python chat_memory.py
```

Choose **1** to summarize the included sample chat, or **2** to paste a conversation. Paste the whole chat, enter `/done` on its own line, then provide a title. Keep speaker labels such as `You said:` and `ChatGPT said:` when available. The summary is saved to `output/chat_summary.md` and opened in Notepad. Ready-made conversations for a capstone project, study plan, and community book exchange are in `examples/showcase_*.txt` and can be pasted into option 2.

To run the sample flow directly:

```powershell
python scripts/demo_memory.py
```

The app uses only Python's standard library; there is no dependency installation step.

## What the summary contains

- Your goals, requirements, and preferences
- Decisions you explicitly confirmed
- Assistant ideas and guidance, clearly marked as unconfirmed
- Disagreements, rejected ideas, and unresolved points when detected
- Source turn numbers so you can check the original context

The extractor uses transparent text rules rather than a language model. It reduces a conversation to selected points, but it is not lossless compression and can miss nuance. Treat the note as a reviewable memory aid, not a replacement for the original conversation.

## Data and privacy

Conversation text and extracted memories are stored locally in `data/chat_memory.sqlite3`. Each run replaces `output/chat_summary.md` with the latest summary. Both locations are ignored by Git by default.

## Project layout

```text
chat_memory.py              Terminal app entry point
src/app/                    Import, extraction, SQLite storage, and summary output
examples/sample_chat.json   Prepared demo conversation
examples/showcase_*.txt     Mock conversations for live demonstrations
scripts/demo_memory.py      Direct sample-demo entry point
data/                       Local SQLite database (created as needed)
output/                     Generated Notepad summary (created as needed)
archive/research/           Earlier model research, datasets, reports, and tests
archive/generated/          Moved Python and test cache files
```

The app flow is: read the sample or pasted text → identify messages and speakers → extract and rank memory points → save messages and points in SQLite → write the Markdown recovery note → open it in Notepad.
