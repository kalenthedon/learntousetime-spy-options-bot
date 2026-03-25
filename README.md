# Learn To Use Time SPY Options Bot

Separate repo for the SPY options bot research branch.

Current first path:
- underlying: `SPY`
- structure: `single-leg long puts`
- contract window: `7-21 DTE`

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

## Main Commands

Capture the latest filtered SPY put chain:

```bash
./.venv/bin/python -m src.options.capture_alpaca_option_chain
```

Append one snapshot into the local history file:

```bash
./.venv/bin/python -m src.options.collect_alpaca_option_history
```

Run starter walk-forward ML research on a chain snapshot CSV:

```bash
./.venv/bin/python -m src.backtests.run_options_research --csv-path centralized_data/options/SPY_put_chain_history.csv
```
