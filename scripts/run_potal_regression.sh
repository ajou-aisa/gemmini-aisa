#!/usr/bin/env bash
# W3-2c-10: PoTal Unit + SoC Regression
#
# Usage:
#   ./scripts/run_potal_regression.sh unit
#   ./scripts/run_potal_regression.sh soc 16
#   ./scripts/run_potal_regression.sh soc 16 32 64
#   ./scripts/run_potal_regression.sh all 16 32 64

set -Eeuo pipefail

usage() {
  cat <<'HELP'
Usage: run_potal_regression.sh {unit|soc|all} [16] [32] [64]

  unit          Run PoTal Scala unit tests
  soc 16        Run DIM16 SoC tests
  soc 16 32 64  Run DIM16/32/64 SoC tests
  all 16 32 64  Run all unit and SoC tests

Optional:
  CHIPYARD
  POTAL_LOG_DIR
  POTAL_TIMEOUT_CYCLES
HELP
}

mode="${1:-unit}"

case "$mode" in
  -h|--help|help)
    usage
    exit 0
    ;;
  unit|soc|all)
    shift || true
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

CHIPYARD="${CHIPYARD:-$HOME/firesim/target-design/chipyard}"
GEMMINI="$CHIPYARD/generators/gemmini"
ROCC="$GEMMINI/software/gemmini-rocc-tests"

LOG_DIR="${POTAL_LOG_DIR:-$GEMMINI/test_run_dir/potal-regression/$(date +%Y%m%d-%H%M%S)}"
TIMEOUT="${POTAL_TIMEOUT_CYCLES:-100000000}"

if [[ ! -d "$GEMMINI/src/test/scala/gemmini" ]]; then
  echo "ERROR: Gemmini source not found: $GEMMINI" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

{
  printf 'date=%s\n' "$(date -Is)"
  printf 'mode=%s\n' "$mode"
  printf 'chipyard=%s\n' "$CHIPYARD"
  printf 'gemmini=%s\n' "$GEMMINI"
  printf 'chipyard_sha=%s\n' \
    "$(git -C "$CHIPYARD" rev-parse HEAD)"
  printf 'gemmini_sha=%s\n' \
    "$(git -C "$GEMMINI" rev-parse HEAD)"
  git -C "$GEMMINI" status --short
} > "$LOG_DIR/manifest.txt"

echo "Regression mode: $mode"
echo "Logs: $LOG_DIR"

run_in() {
  local log="$1"
  local where="$2"
  shift 2

  echo "+ (cd $where && $*)"

  if (cd "$where" && "$@") > "$log" 2>&1; then
    tail -n 6 "$log"
  else
    echo "FAIL: $log" >&2
    tail -n 80 "$log" >&2
    return 1
  fi
}

# ============================================================
# Scala Unit Regression
# ============================================================

run_unit() {
  local specs=(
    PoTalConfigSpec
    PoTalSCUSpec
    PoTalScaleMemorySpec
    PoTalScaleBlockSpec
    PoTalScaleCommandIngressSpec
    PoTalAccPipeSpec
    PoTalAccumulatorRMWSpec
    PoTalPreloadReservationGateSpec
    PoTalAccWriteRouterSpec
    PoTalWritebackSpec
    PoTalWritebackRMWSpec
    PoTalDrainHandshakeSpec
  )

  local files=''
  local only='gemmini/testOnly'
  local name path

  for name in "${specs[@]}"; do
    path="src/test/scala/gemmini/${name}.scala"

    if [[ ! -f "$GEMMINI/$path" ]]; then
      echo "Missing test: $path" >&2
      return 1
    fi

    [[ -z "$files" ]] || files+=', '

    files+="(gemmini / baseDirectory).value / \"${path}\""
    only+=" gemmini.${name}"
  done

  echo "[Scala] Running ${#specs[@]} PoTal suites"

  run_in "$LOG_DIR/unit.log" "$CHIPYARD" \
    sbt \
    "set gemmini / Test / unmanagedSources := Seq(${files})" \
    "$only"

  if grep -Eq \
      'No tests to run|Total number of tests run: 0' \
      "$LOG_DIR/unit.log" ||
     ! grep -Eq \
      'Total number of tests run: [1-9][0-9]*|Tests: succeeded [1-9][0-9]*' \
      "$LOG_DIR/unit.log"; then
    echo "FAIL: Scala test summary unexpected" >&2
    return 1
  fi

  echo "PASS Scala unit suites"
}

# ============================================================
# SoC Regression
# ============================================================

run_soc_dim() {
  local dim="$1"
  local config
  local tests=()
  local item test marker binary log

  case "$dim" in

    16)
      config=PoTalSat32GemminiRocketConfig

      tests=(
        'potal_scale_control:POTAL_SCALE_CONTROL PASS'
        'potal_fence_dma:POTAL_FENCE_DMA PASS'
        'potal_gemm_d16:POTAL_GEMM_D16 PASS'
        'potal_gemm_k32:POTAL_K32 PASS'
        'potal_gemm_n32:POTAL_N32 PASS'
        'potal_gemm_m32n32:POTAL_N32 PASS'
        'potal_release_inflight:POTAL_RELEASE_INFLIGHT PASS'
        'potal_scale_reuse_p1:POTAL_SCALE_REUSE_P1 PASS'
        'potal_scale_reuse_p2:POTAL_SCALE_REUSE_P2 PASS'
        'potal_sat32_d16:POTAL_SAT32_D16 PASS'
      )
      ;;

    32)
      config=PoTalSat32D32GemminiRocketConfig

      tests=(
        'potal_gemm_d32:POTAL_GEMM_D32 PASS'
        'potal_gemm_d32_k64:POTAL_D32_K64 PASS'
        'potal_sat32_d32:POTAL_SAT32_D32 PASS'
      )
      ;;

    64)
      config=PoTalSat32D64GemminiRocketConfig

      tests=(
        'potal_gemm_d64:POTAL_GEMM_D64 PASS'
        'potal_gemm_d64_k128:POTAL_D64_K128 PASS'
        'potal_sat32_d64:POTAL_SAT32_D64 PASS'
      )
      ;;

    *)
      echo "Unsupported DIM: $dim" >&2
      return 2
      ;;
  esac

  echo "[SoC] DIM=$dim CONFIG=$config"

  # Build the matching Verilator simulator.
  run_in "$LOG_DIR/soc-d${dim}-build.log" "$CHIPYARD" \
    make -C sims/verilator \
    "CONFIG=$config" \
    CONFIG_PACKAGE=chipyard

  mkdir -p "$ROCC/build/bareMetalC"

  for item in "${tests[@]}"; do

    test="${item%%:*}"
    marker="${item#*:}"

    binary="$ROCC/build/bareMetalC/${test}-baremetal"

    if [[ ! -f "$ROCC/bareMetalC/${test}.c" ]]; then
      echo "Missing C test: $test" >&2
      return 1
    fi

    echo "[Build] DIM=$dim $test"

    run_in \
      "$LOG_DIR/soc-d${dim}-${test}-compile.log" \
      "$ROCC" \
      make -C "$ROCC/build/bareMetalC" \
      -f "$ROCC/bareMetalC/Makefile" \
      "abs_top_srcdir=$ROCC" \
      "src_dir=$ROCC/bareMetalC" \
      XLEN=64 \
      BAREMETAL_ONLY=1 \
      "${test}-baremetal"

    echo "[Run] DIM=$dim $test"

    log="$LOG_DIR/soc-d${dim}-${test}.log"

    run_in "$log" "$CHIPYARD" \
      make -C sims/verilator \
      "CONFIG=$config" \
      CONFIG_PACKAGE=chipyard \
      "timeout_cycles=$TIMEOUT" \
      "BINARY=$binary" \
      run-binary

    if ! grep -Fq "$marker" "$log"; then
      echo "FAIL: Missing PASS marker: $marker" >&2
      echo "Log: $log" >&2
      return 1
    fi

    if grep -Eq \
      '(^|[[:space:]])FAIL([[:space:]:]|$)' \
      "$log"; then
      echo "FAIL: Test reported failure" >&2
      echo "Log: $log" >&2
      return 1
    fi

    echo "PASS DIM=$dim $test"

  done
}

# ============================================================
# Main
# ============================================================

case "$mode" in

  unit)
    run_unit
    ;;

  soc|all)
    if [[ "$mode" == all ]]; then
      run_unit
    fi

    if (( $# == 0 )); then
      set -- 16
    fi

    for d in "$@"; do
      run_soc_dim "$d"
    done
    ;;

esac

echo "ALL SELECTED REGRESSIONS PASS"
echo "Results: $LOG_DIR"
