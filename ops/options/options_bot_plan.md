# Options Bot Plan

## Goal

Build a separate ML-assisted options bot family without changing the approved crypto bot.

## Why Separate

Options add:
- strike selection
- expiration selection
- implied volatility and Greeks
- spread/liquidity constraints
- assignment and exercise risk

That makes them a different product line, not a new symbol on the crypto stack.

## First Research Track

Start with one underlying and one simple contract-selection problem.

Recommended first track:
- underlying: `SPY` or `QQQ`
- structure: long premium only
- side: directional contracts selected from the chain
- entry horizon: short swing windows
- exit style: fixed holding window + premium target / adverse limit

## Repo Components

Research bootstrap lives in:
- `src/options/contracts.py`
- `src/options/features.py`
- `src/options/labels.py`
- `src/backtests/run_options_research.py`

## Data Contract

The starter research branch expects one flat option-chain snapshot CSV with these columns:
- `timestamp`
- `underlying_symbol`
- `underlying_price`
- `option_symbol`
- `option_type`
- `strike`
- `expiration`
- `days_to_expiry`
- `bid`
- `ask`
- `mid`
- `mark_iv`
- `delta`
- `gamma`
- `theta`
- `vega`
- `open_interest`
- `volume`

Each row is one contract snapshot at one timestamp.

## Feature Direction

Current starter features include:
- moneyness
- time to expiry
- spread as percent of premium
- extrinsic value share
- IV / Greeks
- short-horizon premium returns
- short-horizon underlying returns
- liquidity / open-interest / volume context
- chain-relative spread and OI ranks

## Initial Label

The starter label asks:
- does this contract reach a target premium return within the next `N` bars
- without exceeding a maximum adverse premium move first

This is a practical first target for long-premium systems.

## Promotion Logic We Will Eventually Need

Do not reuse crypto gates directly.

Options-specific gates should include:
- minimum open interest
- minimum volume
- maximum spread as percent of premium
- fill realism
- expiration discipline
- contract concentration limits

## Next Build Steps

1. Choose first underlying and broker/data source.
2. Collect a clean chain-snapshot dataset.
3. Run `src/backtests/run_options_research.py`.
4. Inspect OOS quality and top-contract hit rate.
5. Add contract selection rules and a true options PnL backtest.
6. Add paper execution only after chain selection is stable.
