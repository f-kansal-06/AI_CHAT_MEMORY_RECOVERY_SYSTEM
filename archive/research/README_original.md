# Local Conversational Memory & Recovery

This repository includes a small terminal demonstration that summarizes a sample chat or a chat pasted by the user, then opens the result in Notepad. Processing is local: it does not call ChatGPT, Gemini, or any hosted API.

The original research pipeline remains under `src/data`, `src/models`, `src/baselines`, `src/eval`, and `scripts`. Its DistilBERT classifier predicts turn importance; it does not extract facts or establish whether a claim is true. The product layer keeps extraction and importance ranking separate.

## Run the lab demonstration

Use the project virtual environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python chat_memory.py
```

The menu has two main actions:

1. **Summarize the sample chat in Notepad** uses a prepared weekend-trip conversation.
2. **Paste my ChatGPT chat and summarize it** accepts a full transcript in the terminal.

For the paste option, copy the conversation, paste it into the terminal, then type `/done` on a new line and press Enter. Keep speaker headings such as `You said:` and `ChatGPT said:` when they are included; `User:`, `Assistant:`, and Gemini headings are also recognized. Give the chat a short title when prompted.

For a ChatGPT page paste that starts with `Loading older messages…` and has the question above a blank line followed by the answer, the app treats that question as the user message and the text below it as the assistant reply. It removes the loading line from the summary.

Both actions update `output/chat_summary.md` and open it in Notepad. The summary separates your goals and questions from assistant guidance and project ideas. It lists only explicit user decisions as decisions; if none were found, it says so. Source turn references are included. The local rule extractor is a simple baseline, so check the source references if a point looks wrong.

Start the menu with:

```powershell
python chat_memory.py
```

You can launch the prepared sample directly with:

```powershell
python scripts/demo_memory.py
```

The source chat and extracted points are stored in the local SQLite database at `data/chat_memory.sqlite3`. The output file is overwritten by the next summary. You can choose a different output path with `--output PATH`.

Supported formats:

- TXT transcripts with `User:` / `Assistant:` labels; unmarked paragraphs are treated as user messages.
- Pasted text also recognizes `You said:`, `ChatGPT said:`, `Gemini said:`, and common speaker names.
- JSON objects containing `messages`, `conversation`, or `turns`, or a JSON list of message objects.
- Single-conversation ChatGPT-style JSON exports with a `mapping` of message nodes.
- JSONL with one message object per line.

The default SQLite database is `data/chat_memory.sqlite3`. Override it with `--db PATH`. Re-importing identical message content returns the existing conversation instead of duplicating it. Deleting a conversation also deletes its messages, memories, and processing runs.

## Memory extraction and boundaries

The extractor is deterministic local rule-based NLP, with no generative model and no remote calls. It stores short fact, preference, requirement, decision, agreement, suggestion, rejected suggestion, uncertainty, unresolved question, disputed claim, conclusion, and next-action entries. Each entry records source message IDs, status, confidence, and importance. Exact duplicate entries link to all their source messages.

This approach is an initial, inspectable baseline. It does not perform general semantic entailment, reliably resolve pronouns, or detect implicit disagreements. It marks disagreement when the message uses an explicit disagreement cue. A suggestion is not automatically an agreed fact. Review extracted items against their sources before relying on a recovery brief.

The saved DistilBERT model belongs to the separate turn-importance research experiments. The simple chat-summary demo does not load it; it is not needed to summarize a chat and cannot extract facts by itself.

## Train and evaluate the local importance model

The tuning script uses a seeded randomized subset of the hyperparameter grid, validation PR-AUC for checkpoint selection, early stopping, class-weighted loss, and one held-out test evaluation after model selection. It saves each trial checkpoint, configuration, training curve, and a comparison report.

```powershell
python scripts/tune_memory_model.py --dataset custom --trials 5 --max-train-turns 2000
```

`custom` has weak heuristic labels, so the resulting metrics are exploratory and must not be presented as gold-standard generalization. `--dataset longmemeval` is available, but this workspace's prepared validation set has only 3 positive examples, too few for stable hyperparameter selection. Acquire the full benchmark and create reliable splits before drawing stronger conclusions.

The pretrained `distilbert-base-uncased` weights and tokenizer must already be available locally (or downloaded before running in an environment with network access). Training and CLI inference default to local-only loading. Missing weights cause a clear error; there is no silent random-weight fallback. CUDA is selected automatically when available; otherwise training uses CPU.

To cache the encoder and tokenizer before switching to offline mode, run once while model downloads are allowed:

```powershell
python -c "from transformers import AutoTokenizer, DistilBertModel; AutoTokenizer.from_pretrained('distilbert-base-uncased'); DistilBertModel.from_pretrained('distilbert-base-uncased')"
```

Tuning outputs are written under `results/tuning/`. The selected checkpoint path is recorded in `results/tuning/tuning_results.json`. The default model checkpoint is intentionally excluded from Git because it is large; keep a local copy or configure artifact storage before deployment.

### Current exploratory run

The five-trial run in this workspace selected trial 4 (learning rate `1e-5`, batch size 16, sequence length 256, dropout 0.1, weight decay 0.05, early stopped during a 3-epoch maximum). On the 598-turn custom test partition, DistilBERT reached PR-AUC 0.818, ROC-AUC 0.875, and F1 0.845 at a validation-selected threshold. TF-IDF on the same sampled training turns reached PR-AUC 0.853 and F1 0.846. At a 20% token budget, DistilBERT retained 47.4% of weakly labeled evidence turns. The model did not beat TF-IDF on PR-AUC; this result is preserved as measured.

These are heuristic custom labels, not gold facts or user judgments. The scores validate that training, checkpoint selection, test evaluation, and retention measurement execute; they do not establish that the extractor understands arbitrary conversation meaning.

## Setup

Dependencies are listed in `requirements.txt`. In an existing virtual environment:

```powershell
python -m pip install -r requirements.txt
```

No API keys are required. The app uses Python's built-in SQLite support.

## Tests

```powershell
python -m pytest -q tests/test_local_memory.py tests/test_pipeline.py
```

Tests cover TXT/JSON/JSONL ingestion, extraction stance and provenance, duplicate handling, SQLite persistence, search, recovery from stored memory, deletion, budget constraints, and the existing offline evaluation pipeline.

## Existing research assets and known data limits

Processed datasets, split files, historical experiment outputs, and reports are retained in the repository. Raw datasets are not present under `data/raw/`, so the preprocessing pipeline cannot be rerun from raw source data in this workspace. The available custom set contains 3,574 turns and 147 conversations according to existing project documentation; labels are heuristic. The prepared LongMemEval subset has 25 haystacks and only 3 positives in validation. Historical reports disagree with checked-in result JSON and should be treated as unverified until regenerated.

The simulated recovery evaluator now uses exact, namespaced aliases for supported evidence ID formats instead of substring matching. It remains an offline benchmark and is distinct from the SQLite-backed application recovery path.
