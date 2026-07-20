# Local cross-encoder deduplication experiment

Production remains `DEDUP_EXPERIMENT_MODE=existing_embedding_only`. The runtime below is optional and is not installed by `backend/requirements.txt`.

## Install

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend/requirements-cross-encoder.txt
```

## Cache exact model revisions

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; models=[('BAAI/bge-reranker-base','2cfc18c9415c912f9d8155881c133215df768a70'),('BAAI/bge-reranker-v2-m3','953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e')]; [snapshot_download(repo_id=model,revision=revision,cache_dir='data/model-cache') for model,revision in models]"
```

Model downloads are the only network-dependent step. They use no transcript or evaluation evidence.

## Evaluate offline

One model:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe backend/scripts/evaluate_dedup_cross_encoder.py --models BAAI/bge-reranker-base --cache-dir data/model-cache --output-dir data/outputs/dedup-cross-encoder
```

All models:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe backend/scripts/evaluate_dedup_cross_encoder.py --models all --cache-dir data/model-cache --output-dir data/outputs/dedup-cross-encoder
```

The evaluator writes `evaluation.json` and `evaluation.md`. It does not start the API or read persisted transcripts.

## Tests

Unit tests, with no model loading:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/app/tests/test_cross_encoder_experiment.py backend/app/tests/test_signal_deduplicator.py backend/app/tests/test_dedup_evaluation.py
```

Opt-in real cached-model CPU integration test:

```powershell
$env:RUN_CROSS_ENCODER_INTEGRATION='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe -m pytest backend/app/tests/test_cross_encoder_experiment.py
```

Full backend suite:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/app/tests
```

Normal application imports do not import Sentence Transformers, Transformers, or PyTorch. `trust_remote_code` remains false. CPU is the default device.
