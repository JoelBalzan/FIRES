#!/bin/bash
#SBATCH --job-name=FIRES
#SBATCH --output=FIRES_simulation_%A_%a.out
#SBATCH --error=FIRES_simulation_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              # Number of CPUs per task
#SBATCH --mem=16G                        # Memory allocation
#SBATCH --time=00:30:00                 # Time limit
#SBATCH --array=0-19                    # Array job index (20 chunks)

source "/fred/oz313/processing/jbalzan/venv.sh"
set -euo pipefail

PLOT="l_frac" # or pa_var

CONFDIR="/fred/oz313/processing/jbalzan/packages/FIRES/paper/240318A/"
GPARAMS="${CONFDIR}/fires.toml" # now read master fires.toml for sweep info

# Auto-detect varied column (single non-zero in last row)
read VAR START STOP STEP <<<"$(
  python - "$GPARAMS" <<'PY'
import sys
from pathlib import Path
try:
    import tomllib as toml
except Exception:
    import tomli as toml

path = Path(sys.argv[1])
data = toml.loads(path.read_bytes())
param = data.get('analysis', {}).get('sweep', {}).get('parameter', {})
name = str(param.get('name', 'UNKNOWN'))
start = param.get('start', 0)
stop = param.get('stop', 0)
step = param.get('step', 0)

# Map toml parameter names to legacy short names used by scripts
nm = name.lower()
nm = nm.replace('sd_', '').replace('_sigma', '').replace('_deg', '').replace('deg', '')
mapping = {
    't0': 't0', 'width': 'width', 'a': 'A', 'amplitude': 'A', 'amplitude_jy': 'A',
    'spec_idx': 'spec_idx', 'spectral_index': 'spec_idx',
    'tau': 'tau', 'dm': 'DM', 'rm': 'RM', 'pa': 'PA', 'pa_deg': 'PA',
    'lfrac': 'lfrac', 'vfrac': 'vfrac', 'dpa': 'dPA', 'dpa_deg_per_ms': 'dPA',
    'band_centre': 'band_centre', 'band_centre_mhz': 'band_centre',
    'band_width': 'band_width', 'band_width_mhz': 'band_width',
    'n': 'N', 'mg_width_low': 'mg_width_low', 'mg_width_high': 'mg_width_high'
}

var = mapping.get(nm, 'UNKNOWN')
print(var, start, stop, step)
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

BASE_OUTDIR="240318A/${PLOT}/${VAR}/start_${START}_stop_${STOP}_step_${STEP}/${OVERRIDE_SUFFIX}"
mkdir -p "$BASE_OUTDIR"

OUTDIR="${BASE_OUTDIR}"
mkdir -p "$OUTDIR"
echo "Running chunk=${CHUNK} overrides: ${OVERRIDE_ARGS[*]} -> $OUTDIR"

fires \
  -f "sweep_${CHUNK}" \
  -v \
  --mode psn \
  --seed 0 \
  --nseed 500 \
  --write \
  --plot ${PLOT} \
  --config-dir "$CONFDIR" \
  --output-dir "$OUTDIR" \
  --override-plot show_plots=false save_plots=false \
  "${OVERRIDE_ARGS[@]}" \
  #MAKE SURE TO SET --snr 110 IN CONFIG

