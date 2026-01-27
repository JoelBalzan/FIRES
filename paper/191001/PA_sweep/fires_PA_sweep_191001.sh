#!/bin/bash
#SBATCH --job-name=FIRES
#SBATCH --output=FIRES_simulation_%A_%a.out
#SBATCH --error=FIRES_simulation_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              # Number of CPUs per task
#SBATCH --mem=16G                        # Memory allocation
#SBATCH --time=03:00:00                 # Time limit
#SBATCH --array=0-19                    # Array job index (20 chunks)

source "/fred/oz313/processing/jbalzan/venv.sh"
set -euo pipefail

PLOT="l_frac" # or pa_var

CONFDIR="/fred/oz313/processing/jbalzan/packages/FIRES/paper/191001/"
GPARAMS="${CONFDIR}/gparams.toml" # adjust if using ~/.config/fires/gparams.*

# Auto-detect varied column (single non-zero in last row)
read VAR START STOP STEP <<<"$(
  python - "$GPARAMS" <<'PY'
import sys, numpy as np
path = sys.argv[1]
arr = np.loadtxt(path)
last = arr[-1]
idx = np.where(last!=0)[0]
cols = ["t0","width","A","spec_idx","tau","DM","RM","PA","lfrac","vfrac","dPA","band_centre","band_width","N","mg_width_low","mg_width_high"]
if len(idx)!=1:
    print("UNKNOWN 0 0 0"); sys.exit(0)
i = idx[0]
start, stop, step = arr[-3,i], arr[-2,i], arr[-1,i]
print(cols[i], start, stop, step)
PY
)"

if [[ "$VAR" == "UNKNOWN" ]]; then
  echo "No unique sweep defined (last row)."
  exit 1
fi

echo "Sweep var: $VAR  range: $START:$STOP:$STEP"
echo "Array task: $SLURM_ARRAY_TASK_ID"

PARAMS=("t0" "width" "A" "spec_idx" "tau" "DM" "RM" "PA" "lfrac" "vfrac" "dPA" "band_centre" "band_width" "N" "mg_width_low" "mg_width_high")
OVERRIDE_ARGS=()
OVERRIDE_SUFFIX=""
for param in "${PARAMS[@]}"; do
  val="${!param:-}"
  val_sd_var="${param}_sd"
  val_sd="${!val_sd_var:-}"
  if [[ -n "$val" ]]; then
    OVERRIDE_ARGS+=(--override-param "${param}=${val}")
    OVERRIDE_SUFFIX+="${param}${val}"
  fi
  if [[ -n "$val_sd" ]]; then
    OVERRIDE_ARGS+=(--override-param "sd_${param}=${val_sd}")
    OVERRIDE_SUFFIX+="sd${param}${val_sd}"
  fi
done

CHUNK="$SLURM_ARRAY_TASK_ID"

BASE_OUTDIR="191001/${PLOT}/${VAR}/start_${START}_stop_${STOP}_step_${STEP}/${OVERRIDE_SUFFIX}"
mkdir -p "$BASE_OUTDIR"

OUTDIR="${BASE_OUTDIR}"
mkdir -p "$OUTDIR"
echo "Running chunk=${CHUNK} overrides: ${OVERRIDE_ARGS[*]} -> $OUTDIR"

fires \
  -f "sweep_${CHUNK}" \
  -v \
  --mode psn \
  --seed 3 \
  --nseed 500 \
  --write \
  --plot ${PLOT} \
  --config-dir "$CONFDIR" \
  --output-dir "$OUTDIR" \
  --override-plot show_plots=false save_plots=false \
  --ncpu ${SLURM_CPUS_PER_TASK} \
  --sweep-mode sd \
  "${OVERRIDE_ARGS[@]}" \
  --snr 174

