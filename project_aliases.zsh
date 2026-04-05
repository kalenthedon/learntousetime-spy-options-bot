export LTUT_OPTIONS_REPO="/Users/kalenthedon/Projects/LearnToUseTime-SPY-OptionsBot"
export LTUT_OPTIONS_PY="$LTUT_OPTIONS_REPO/.venv/bin/python"
export LTUT_OPTIONS_DASH_CERT="$LTUT_OPTIONS_REPO/deployment/certs/localhost-cert.pem"
export LTUT_OPTIONS_DASH_KEY="$LTUT_OPTIONS_REPO/deployment/certs/localhost-key.pem"

alias ltut-opt-cd='cd "$LTUT_OPTIONS_REPO"'
alias ltut-opt-status='cd "$LTUT_OPTIONS_REPO" && git status --short && git log --oneline -1'
alias ltut-opt-capture='cd "$LTUT_OPTIONS_REPO" && "$LTUT_OPTIONS_PY" -m src.options.capture_alpaca_option_chain'
alias ltut-opt-collect='cd "$LTUT_OPTIONS_REPO" && "$LTUT_OPTIONS_PY" -m src.options.collect_alpaca_option_history'
alias ltut-opt-market-collect='cd "$LTUT_OPTIONS_REPO" && "$LTUT_OPTIONS_PY" -m src.options.market_hours_collect'
alias ltut-opt-history='cd "$LTUT_OPTIONS_REPO" && "$LTUT_OPTIONS_PY" -m src.options.history_status'
alias ltut-opt-research='cd "$LTUT_OPTIONS_REPO" && "$LTUT_OPTIONS_PY" -m src.backtests.run_options_research --csv-path centralized_data/options/SPY_put_chain_history.csv'
alias ltut-opt-paper-status='cd "$LTUT_OPTIONS_REPO" && "$LTUT_OPTIONS_PY" -m src.options.paper_ops_status'

ltut-opt-dashboard() {
  cd "$LTUT_OPTIONS_REPO" || return 1
  "$LTUT_OPTIONS_PY" -m src.options.paper_dashboard \
    --host 127.0.0.1 \
    --port 8797 \
    --certfile "$LTUT_OPTIONS_DASH_CERT" \
    --keyfile "$LTUT_OPTIONS_DASH_KEY"
}
