# Learn To Use Time SPY Options Bot SOP

## Scope

This repo is for the SPY options bot only.

Current first structure:
- `SPY`
- long puts
- `7-21 DTE`

## Workflow

1. Capture filtered live chain snapshots from Alpaca.
2. Append snapshots into the local history CSV.
3. Run walk-forward research on the accumulated history.
4. Add contract selection and options-specific PnL logic before any paper execution work.

## Commands

- latest chain:
  - `./.venv/bin/python -m src.options.capture_alpaca_option_chain`
- append history:
  - `./.venv/bin/python -m src.options.collect_alpaca_option_history`
- research:
  - `./.venv/bin/python -m src.backtests.run_options_research --csv-path centralized_data/options/SPY_put_chain_history.csv`

## Important Notes

- This repo is intentionally separate from the crypto bot repo.
- Do not mix crypto deployment files into this codebase.
- No live or paper options execution exists yet.
