#!/bin/bash
#SBATCH --job-name=FIRES
#SBATCH --output=FIRES_simulation_%A_%a.out
#SBATCH --error=FIRES_simulation_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-19

source "/fred/oz313/processing/jbalzan/venv.sh"
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PLOT="l_frac"
CONFDIR="/fred/oz313/processing/jbalzan/packages/FIRES/examples/20240318A/"
GPARAMS="${CONFDIR}/fires.toml"

# --- Read sweep parameters from TOML via FIRES CLI ---
SWEEP_INFO=$(fires --config-dir "$CONFDIR" --print-sweep-info) || {
  echo "ERROR: failed to parse sweep parameters from $GPARAMS"
  exit 1
}

IFS=$'\t' read -r VAR START STOP STEP <<<"$SWEEP_INFO"

if [[ -z "$VAR" || "$VAR" == "UNKNOWN" ]]; then
  echo "ERROR: unrecognised sweep variable (raw: $SWEEP_INFO)"
  exit 1
fi
echo "Sweep: $VAR  $START -> $STOP  step $STEP  |  array task $SLURM_ARRAY_TASK_ID"

# --- Build fires override args from env vars injected by sbatch --export ---
FIRES_PARAMS=(t0 width A spec_idx tau DM RM PA lfrac vfrac dPA band_centre band_width
  mg_width_low mg_width_high)

OVERRIDE_ARGS=()
OVERRIDE_SUFFIX=""

for param in "${FIRES_PARAMS[@]}"; do
  val="${!param:-}"
  sd="${param}_sd"
  val_sd="${!sd:-}"
  if [[ -n "$val" ]]; then
    OVERRIDE_ARGS+=(--override-param "${param}=${val}")
    OVERRIDE_SUFFIX+="${param}${val}"
  fi
  if [[ -n "$val_sd" ]]; then
    OVERRIDE_ARGS+=(--override-param "sd_${param}=${val_sd}")
    OVERRIDE_SUFFIX+="sd_${param}${val_sd}"
  fi
done

N_VAL="${N:-}"
if [[ -n "$N_VAL" ]]; then
  OVERRIDE_ARGS+=(--override-param "N=${N_VAL}")
  OVERRIDE_SUFFIX+="N${N_VAL}"
fi

OUTDIR="240318A/${PLOT}/${VAR}/start_${START}_stop_${STOP}_step_${STEP}/A1-P3/${OVERRIDE_SUFFIX}"
mkdir -p "$OUTDIR"
echo "Output dir: $OUTDIR"
echo "Overrides:  ${OVERRIDE_ARGS[*]:-none}"

fires \
  -f "sweep_${SLURM_ARRAY_TASK_ID}" \
  -v \
  --plot "$PLOT" \
  --config-dir "$CONFDIR" \
  --output-dir "$OUTDIR" \
  --override-plot show_plots=false save_plots=false \
  ${OVERRIDE_ARGS[@]+"${OVERRIDE_ARGS[@]}"}
# MAKE SURE TO SET --snr 110 IN CONFIG
