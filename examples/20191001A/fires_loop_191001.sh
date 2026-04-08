#!/bin/bash
#SBATCH --job-name=FIRES
#SBATCH --output=FIRES_simulation_%A_%a.out
#SBATCH --error=FIRES_simulation_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --array=0-19

source "/fred/oz313/processing/jbalzan/venv.sh"
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PLOT="l_frac"
CONFDIR="/fred/oz313/processing/jbalzan/packages/FIRES/examples/20191001A/"
GPARAMS="${CONFDIR}/fires.toml"

# --- Read sweep parameters from TOML ---
SWEEP_INFO=$(python - "$GPARAMS" <<'PY'
import sys
from pathlib import Path
try:
    import tomllib as toml
except ImportError:
    import tomli as toml

data = toml.loads(Path(sys.argv[1]).read_text())
param = data.get('analysis', {}).get('sweep', {}).get('parameter', {})
name  = str(param.get('name',  'UNKNOWN'))
start = param.get('start', 0)
stop  = param.get('stop',  0)
step  = param.get('step',  0)

nm = name.lower().strip()
if nm.startswith('meas_mean_'):
  nm = nm[len('meas_mean_'):]
if nm.startswith('meas_std_'):
  nm = nm[len('meas_std_'):]
if nm.startswith('meas_var_'):
  nm = nm[len('meas_var_'):]
if nm.startswith('mean_'):
  nm = nm[len('mean_'):]
if nm.endswith('_mean'):
  nm = nm[:-len('_mean')]
if nm.startswith('sd_'):
  nm = nm[len('sd_'):]
if nm.endswith('_sigma'):
  nm = nm[:-len('_sigma')]
nm = nm.replace('_deg_per_ms', '').replace('_deg', '').replace('deg', '')
nm = nm.replace('_mhz', '').replace('_hz', '').replace('_ms', '')
mapping = {
  't0':'t0', 'width':'width', 'a':'A', 'amplitude':'A', 'amplitude_jy':'A',
  'spec_idx':'spec_idx', 'spectral_index':'spec_idx',
  'tau':'tau', 'dm':'DM', 'rm':'RM', 'pa':'PA', 'pa_deg':'PA',
  'pa_sigma_deg':'pa_sigma_deg', 'sd_pa':'pa_sigma_deg', 'pa_sigma':'pa_sigma_deg',
  'lfrac':'lfrac', 'vfrac':'vfrac', 'dpa':'dPA', 'dpa_deg_per_ms':'dPA',
  'band_centre':'band_centre', 'band_centre_mhz':'band_centre',
  'band_width':'band_width',   'band_width_mhz':'band_width',
  'n':'N', 'mg_width_low':'mg_width_low', 'mg_width_high':'mg_width_high',
}
print(f"{mapping.get(nm,'UNKNOWN')}\t{start}\t{stop}\t{step}")
PY
) || { echo "ERROR: failed to parse sweep parameters from $GPARAMS"; exit 1; }

IFS=$'\t' read -r VAR START STOP STEP <<< "$SWEEP_INFO"

if [[ -z "$VAR" || "$VAR" == "UNKNOWN" ]]; then
  echo "ERROR: unrecognised sweep variable (raw: $SWEEP_INFO)"; exit 1
fi
echo "Sweep: $VAR  $START -> $STOP  step $STEP  |  array task $SLURM_ARRAY_TASK_ID"

# --- Build fires override args from env vars injected by sbatch --export ---
# N is the microshot count (always included in path; only forwarded to fires if set).
# All other params follow the same pattern: forward value and optional _sd variant.
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

# N handled separately: always appended to path suffix
N_VAL="${N:-}"
if [[ -n "$N_VAL" ]]; then
  OVERRIDE_ARGS+=(--override-param "N=${N_VAL}")
  OVERRIDE_SUFFIX+="N${N_VAL}"
fi

OUTDIR="191001/${PLOT}/${VAR}/start_${START}_stop_${STOP}_step_${STEP}/${OVERRIDE_SUFFIX}"
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
  #MAKE SURE TO SET --snr 174 IN CONFIG

