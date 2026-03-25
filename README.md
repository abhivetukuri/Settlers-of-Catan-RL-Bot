# Settlers of Catan RL Bot

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run baseline matches

```bash
python -m scripts.run_match --bot-a random --bot-b greedy --num-games 10 --seed 42
```

Artifacts are written to `artifacts/`.

## Project layout

- `src/sim`: simulator adapter and determinism helpers.
- `src/bots`: baseline bot implementations.
- `src/representations`: state and action encoders.
- `src/eval`: evaluation metrics and match runner.
- `src/logging_utils.py`: JSONL and summary logging utilities.
- `tests`: milestone tests for determinism, action encoding, and match runner.

