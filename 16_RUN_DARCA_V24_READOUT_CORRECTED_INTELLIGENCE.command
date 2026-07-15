#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-$HOME/Downloads}"
PY="${PY:-python3}"
PLAN="${PLAN:-main}"                    # smoke | quick | main
MODE="${MODE:-integrated-only}"         # integrated-only | native-only | both
WORKERS="${WORKERS:-auto}"
OVERWRITE="${OVERWRITE:-0}"
OUTDIR="${OUTDIR:-$HOME/Desktop/DARCA_V24_READOUT_CORRECTED_INTELLIGENCE}"

BATTERY="${BATTERY:-15_DARCA_V24_INTELLIGENCE_EMERGENCE_LEVEL_BATTERY_READOUT_FIXED.py}"
DARCA="${DARCA:-14_darca_v24_directional_readout_fixed.py}"
CORE="${CORE:-02_darca_v24_integrated_agent_core_gravity_fixed.py}"

fail() {
  echo "ERROR: $*" >&2
  [ -t 1 ] && read -r -p "Press Enter to close..." _ || true
  exit 1
}

[ -d "$CODE" ] || fail "code folder not found: $CODE"
cd "$CODE"
command -v "$PY" >/dev/null 2>&1 || fail "python not found: $PY"
[ -f "$BATTERY" ] || fail "battery not found: $CODE/$BATTERY"
[ -f "$DARCA" ] || fail "readout-corrected DARCA not found: $CODE/$DARCA"
[ -f "$CORE" ] || fail "integrated core not found: $CODE/$CORE"

case "$MODE" in
  integrated-only) MODE_FLAG=(--integrated-only) ;;
  native-only)     MODE_FLAG=(--native-only) ;;
  both)            MODE_FLAG=() ;;
  *) fail "MODE must be integrated-only, native-only, or both" ;;
esac

if [ "$OVERWRITE" = "1" ]; then
  RUN_FLAG=(--overwrite)
else
  RUN_FLAG=(--resume)
fi

mkdir -p "$OUTDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUTDIR/run_log_${STAMP}.txt"

{
  echo "============================================================"
  echo "DARCA v24 readout-corrected intelligence battery"
  echo "started : $(date)"
  echo "python  : $($PY -V 2>&1)"
  echo "code    : $CODE"
  echo "battery : $BATTERY"
  echo "darca   : $DARCA"
  echo "core    : $CORE"
  echo "plan    : $PLAN"
  echo "mode    : $MODE"
  echo "workers : $WORKERS"
  echo "outdir  : $OUTDIR"
  echo "============================================================"
} | tee "$LOG"

set +e
"$PY" -u "$BATTERY" \
  --darca-file "$DARCA" \
  --agent-core "$CORE" \
  --outdir "$OUTDIR" \
  --plan "$PLAN" \
  --workers "$WORKERS" \
  "${MODE_FLAG[@]}" \
  "${RUN_FLAG[@]}" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -eq 0 ]; then
  echo "DONE: $OUTDIR" | tee -a "$LOG"
  echo "First read: 00_FIRST_READ_DARCA_v24_readout_corrected_intelligence_report.txt" | tee -a "$LOG"
  echo "Primary tests: 08_PRIMARY_CAPABILITY_TESTS.csv" | tee -a "$LOG"
else
  echo "FAILED (exit $STATUS). See $LOG and 07_validation_errors.csv." | tee -a "$LOG"
fi

if [ -t 1 ]; then
  read -r -p "Press Enter to close..." _ || true
fi
exit "$STATUS"
