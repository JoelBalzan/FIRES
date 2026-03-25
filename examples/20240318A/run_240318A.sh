#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMBINE_PY="$SCRIPT_DIR/combine_sweeps.py"

JOBIDS=()
trap 'echo "Interrupted; submitted job ids: ${JOBIDS[*]:-none}"; exit 2' INT TERM

# --- Jobs to submit (comment out lines to skip) ---
declare -a JOBS=(
  "sbatch --job-name=5   --export=ALL,N=5   fires_loop_240318A.sh"
  "sbatch --job-name=10  --export=ALL,N=10  fires_loop_240318A.sh"
  "sbatch --job-name=20  --export=ALL,N=20  fires_loop_240318A.sh"
  "sbatch --job-name=100 --export=ALL,N=100 fires_loop_240318A.sh"
  #"sbatch --job-name=1000 --export=ALL,N=1000 fires_loop_240318A.sh"
)

for cmd in "${JOBS[@]}"; do
  echo "Running: $cmd"
  out=$($cmd)           # no eval — word-splits cleanly on the declared string
  echo "$out"
  jid="${out##* }"      # last token = job id
  [[ -n "$jid" ]] && JOBIDS+=("$jid")
done

if [[ ${#JOBIDS[@]} -eq 0 ]]; then
  echo "No jobs submitted. Exiting."
  exit 0
fi

ids=$(IFS=,; echo "${JOBIDS[*]}")
echo "Submitted job ids: $ids"
echo "Waiting for jobs to finish..."

while true; do
  remaining=$(squeue -h -j "$ids" 2>/dev/null | wc -l)
  [[ "$remaining" -eq 0 ]] && break
  sleep 30
done

echo "All jobs finished. Running combine_sweeps.py"
python3 "$COMBINE_PY"
echo "Combine step completed."

# sbatch lines
#sbatch --job-name=10 --export=ALL,N=10,mg_width_low=1,mg_width_high=5 fires_n_loop.sh
#sbatch --job-name=100 --export=ALL,N=100,mg_width_low=1,mg_width_high=5 fires_n_loop.sh
#sbatch --job-name=600 --export=ALL,N=600,mg_width_low=1,mg_width_high=5 fires_n_loop.sh
#sbatch --job-name=1000 --export=ALL,N=1000,mg_width_low=1,mg_width_high=5 fires_n_loop.sh

#sbatch --job-name=10 --export=ALL,N=10,mg_width_low=20,mg_width_high=40 fires_n_loop.sh
#sbatch --job-name=100 --export=ALL,N=100,mg_width_low=20,mg_width_high=40 fires_n_loop.sh
#sbatch --job-name=600 --export=ALL,N=600,mg_width_low=20,mg_width_high=40 fires_n_loop.sh
#sbatch --job-name=1000 --export=ALL,N=1000,mg_width_low=20,mg_width_high=40 fires_n_loop.sh