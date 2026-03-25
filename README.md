# Learn To Use Time SPY Options Bot

Separate research repo for the Learn To Use Time Academy SPY options bot.

This repo exists so the options workflow stays isolated from the crypto bot and is easier to teach, maintain, and evolve.

Current first path:
- underlying: `SPY`
- structure: `single-leg long puts`
- contract window: `7-21 DTE`

## What This Repo Does

- captures filtered SPY put-chain snapshots from Alpaca
- appends snapshots into a local history file
- builds option-specific features and labels
- runs a starter walk-forward ML research workflow

## What This Repo Does Not Do Yet

- paper trade options
- place live options orders
- model assignment / exercise risk
- run a full production-grade options PnL and fill simulator

## Repo Structure

- `src/data/`
  - Alpaca options and stock market data helpers
- `src/options/`
  - contract normalization, universe selection, chain capture, and history collection
- `src/backtests/`
  - starter options research pipeline
- `src/ml/`
  - shared walk-forward model training utilities
- `src/validation/`
  - purged walk-forward splitter
- `ops/options/`
  - planning notes for the SPY options bot branch

## Setup

1. Create a virtual environment.
2. Install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Create the local Alpaca env file:

```bash
cp deployment/alpaca_options.env.example deployment/alpaca_options.env
```

4. Add your Alpaca keys to `deployment/alpaca_options.env`.
5. Optional: source the repo aliases:

```bash
source project_aliases.zsh
```

## Main Commands

Capture the latest filtered SPY put chain:

```bash
./.venv/bin/python -m src.options.capture_alpaca_option_chain
```

Append one snapshot into the local history file:

```bash
./.venv/bin/python -m src.options.collect_alpaca_option_history
```

Check whether the local history is deep enough to research:

```bash
./.venv/bin/python -m src.options.history_status
```

Run a scheduled-safe market-hours collection step:

```bash
./.venv/bin/python -m src.options.market_hours_collect
```

Run starter walk-forward ML research on a chain snapshot CSV:

```bash
./.venv/bin/python -m src.backtests.run_options_research --csv-path centralized_data/options/SPY_put_chain_history.csv
```

## Suggested Workflow

1. Collect snapshots repeatedly during market hours.
2. Watch `history_status` until the dataset has enough unique snapshots and contracts.
3. Run the starter research workflow on the accumulated history.
4. Use that output to decide whether to build a real options PnL backtest next.

Suggested automation:
- run `market_hours_collect` every 15 minutes
- the script will skip outside regular US market hours automatically

## Notes

- This repo is intentionally separate from the crypto bot repo.
- The crypto bot lives here:
  - `https://github.com/kalenthedon/learntousetime-crypto-bot`
