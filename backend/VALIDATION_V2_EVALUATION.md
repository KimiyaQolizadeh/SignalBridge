# Validation 2.0 model evaluation

This evaluation uses only synthetic, de-identified examples. It does not read or update the SignalBridge database. Live inference uses the configured evidence-validator model through the existing Responses API wrapper with `temperature=0` and `store=false`.

## Dataset and deterministic checks

```powershell
.\.venv\Scripts\python.exe -m backend.app.services.validation_evaluation validate
.\.venv\Scripts\python.exe -m backend.app.services.validation_evaluation oracle
```

The fixed dataset contains 120 examples: 80 development and 40 untouched holdout. The split and expert labels are constructed before inference.

## Opt-in live inference

Development:

```powershell
.\.venv\Scripts\python.exe -m backend.app.services.validation_evaluation live-development --output-dir data/outputs/validation-v2
```

Untouched holdout:

```powershell
.\.venv\Scripts\python.exe -m backend.app.services.validation_evaluation live-holdout --output-dir data/outputs/validation-v2
```

Three repeated calls over a fixed 12-example development subset:

```powershell
.\.venv\Scripts\python.exe -m backend.app.services.validation_evaluation stability --output-dir data/outputs/validation-v2
```

## Reports

```powershell
.\.venv\Scripts\python.exe -m backend.app.services.validation_evaluation report --input data/outputs/validation-v2/live-development.json --output-dir data/outputs/validation-v2/development
.\.venv\Scripts\python.exe -m backend.app.services.validation_evaluation report --input data/outputs/validation-v2/live-holdout.json --output-dir data/outputs/validation-v2/holdout
```

JSON artifacts contain example IDs and structured findings, but do not persist evidence quotes or context text.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest backend/app/tests/test_validation_evaluation.py backend/app/tests/test_validation_v2.py backend/app/tests/test_evidence_validator.py backend/app/tests/test_llm_schemas.py
.\.venv\Scripts\python.exe -m pytest backend/app/tests
.\.venv\Scripts\python.exe -m compileall -q backend/app backend/scripts
```

## Development-only review prompt experiment

The production prompt remains `03_evidence_validator.txt`. The isolated experiment
uses `03_evidence_validator_review_experiment.txt` and is code-guarded against
holdout selection.

```powershell
.\.venv\Scripts\python.exe -m backend.app.services.validation_evaluation experiment-v1 --output-dir data/outputs/validation-v2-review-experiment
.\.venv\Scripts\python.exe -m backend.app.services.validation_evaluation report --input data/outputs/validation-v2-review-experiment/experiment-v1.json --output-dir data/outputs/validation-v2-review-experiment/v1
.\.venv\Scripts\python.exe -m backend.app.services.validation_evaluation experiment-stability --output-dir data/outputs/validation-v2-review-experiment
```

Experimental modes always use the development split. They cannot select or run
against the frozen holdout. The experimental prompt is not loaded by production.
