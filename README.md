# Local Conversational Memory & Recovery

This repository now includes a terminal application that imports conversation exports, extracts structured memory locally, stores source conversations and memories in SQLite, and reconstructs a continuation brief using the saved memory. It does not call ChatGPT, Gemini, or any hosted chatbot API.

The original research pipeline remains under `src/data`, `src/models`, `src/baselines`, `src/eval`, and `scripts`. Its DistilBERT classifier predicts turn importance; it does not extract facts or establish whether a claim is true. The product layer keeps extraction and importance ranking separate.

## Run the local application

Use the project virtual environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python chat_memory.py
```

Or run one import without opening the menu:

```powershell
python chat_memory.py --import-file examples/realistic_conversation.json
```

The menu supports file or manual import, analysis, browsing, recovery, search, size comparison, saved model evaluation, deletion, and exit. For manual input, choose **Import conversation file** and enter `manual`; use `User:` and `Assistant:` prefixes and finish with `/done`.

Supported formats:

- TXT transcripts with `User:` / `Assistant:` labels; unmarked paragraphs are treated as user messages.
- JSON objects containing `messages`, `conversation`, or `turns`, or a JSON list of message objects.
- Single-conversation ChatGPT-style JSON exports with a `mapping` of message nodes.
- JSONL with one message object per line.

The default SQLite database is `data/chat_memory.sqlite3`. Override it with `--db PATH`. Re-importing identical message content returns the existing conversation instead of duplicating it. Deleting a conversation also deletes its messages, memories, and processing runs.

Try the demo in the CLI:

1. Import `examples/realistic_conversation.json`.
2. Select the imported conversation and analyze it.
3. Review the stored memory, then recover it. The recovery renderer reads the memory table and source IDs; it does not reread transcript messages.
4. Search for `SQLite` and compare original versus retained token counts.

For a one-command end-to-end walkthrough (using a temporary database and the saved local classifier when available):

```powershell
python scripts/demo_memory.py
```

## Memory extraction and boundaries

The required extractor is deterministic local rule-based NLP, with no generative model and no remote calls. It stores short fact, preference, requirement, decision, agreement, suggestion, rejected suggestion, uncertainty, unresolved question, disputed claim, conclusion, and next-action entries. Each entry records source message IDs, status, confidence, and importance. Exact duplicate entries link to all their source messages. Likely polarity conflicts receive a shared review group.

This approach is an initial, inspectable baseline. It does not perform general semantic entailment, reliably resolve pronouns, or guarantee that an arbitrary disagreement is detected. A suggestion is not automatically an agreed fact. Review disputed and low-confidence items before relying on a recovery brief.

The saved importance model is optional. When `results/tuning/tuning_results.json` and its checkpoint are available, analysis combines the model's turn-salience score with the extractor's structural score. Otherwise, analysis reports and uses rule-based importance alone. The model score cannot convert a suggestion or uncertain statement into a confirmed fact.

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
