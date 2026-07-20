# Experimental Signal Importance Estimation

Production remains `IMPORTANCE_MODE=importance_disabled`. The subsystem is not
called by `pipeline_runner.py` and never persists importance values.

## Dataset and baselines

```powershell
.\.venv\Scripts\python.exe -m backend.app.services.importance_evaluation validate
.\.venv\Scripts\python.exe -m backend.app.services.importance_evaluation baselines --output-dir data/outputs/importance
```

## Opt-in development evaluation

```powershell
.\.venv\Scripts\python.exe -m backend.app.services.importance_evaluation development-listwise --output-dir data/outputs/importance
.\.venv\Scripts\python.exe -m backend.app.services.importance_evaluation development-pairwise --output-dir data/outputs/importance
.\.venv\Scripts\python.exe -m backend.app.services.importance_evaluation development-hybrid --output-dir data/outputs/importance
```

## Untouched holdout

```powershell
.\.venv\Scripts\python.exe -m backend.app.services.importance_evaluation holdout-listwise --output-dir data/outputs/importance
.\.venv\Scripts\python.exe -m backend.app.services.importance_evaluation holdout-pairwise --output-dir data/outputs/importance
.\.venv\Scripts\python.exe -m backend.app.services.importance_evaluation holdout-hybrid --output-dir data/outputs/importance
```

## Order sensitivity and reports

```powershell
.\.venv\Scripts\python.exe -m backend.app.services.importance_evaluation order-sensitivity --output-dir data/outputs/importance/order-sensitivity
.\.venv\Scripts\python.exe -m backend.app.services.importance_evaluation report --input data/outputs/importance/holdout-listwise.json --output-dir data/outputs/importance/holdout-listwise
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest backend/app/tests/test_importance_estimator.py backend/app/tests/test_scorer.py backend/app/tests/test_evidence_validator.py backend/app/tests/test_signal_deduplicator.py backend/app/tests/test_cross_encoder_experiment.py backend/app/tests/test_reranker.py backend/app/tests/test_pipeline_runner.py backend/app/tests/test_transcripts_api.py
.\.venv\Scripts\python.exe -m pytest backend/app/tests
.\.venv\Scripts\python.exe -m compileall -q backend/app backend/scripts
```
