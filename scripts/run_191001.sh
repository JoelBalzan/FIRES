#!/usr/bin/env bash
set -euo pipefail

# Submit jobs and wait for them to finish, then combine sweep fragments.
JOBIDS=()

# sbatch lines (keep commented ones for reference)
#sbatch --job-name=PA0 --export=ALL,PA_sd=0 fires_n_loop.sh
#sbatch --job-name=PA30 --export=ALL,PA_sd=30 fires_n_loop.sh
#sbatch --job-name=PA60 --export=ALL,PA_sd=60 fires_n_loop.sh
#sbatch --job-name=PA90 --export=ALL,PA_sd=90 fires_n_loop_191001.sh

for cmd in \
  "sbatch --job-name=5 --export=ALL,N=5 fires_loop_191001.sh" \
  "sbatch --job-name=10 --export=ALL,N=10 fires_loop_191001.sh" \
  #"sbatch --job-name=15 --export=ALL,N=15 fires_loop_191001.sh" \
  "sbatch --job-name=20 --export=ALL,N=20 fires_loop_191001.sh" \
  #"sbatch --job-name=50 --export=ALL,N=50 fires_loop_191001.sh" \
  "sbatch --job-name=100 --export=ALL,N=100 fires_loop_191001.sh" \
  #"sbatch --job-name=200 --export=ALL,N=200 fires_loop_191001.sh" \
  #"sbatch --job-name=600 --export=ALL,N=600 fires_loop_191001.sh" \
  "sbatch --job-name=1000 --export=ALL,N=1000 fires_loop_191001.sh"; do
	if [[ "$cmd" =~ ^# ]]; then
		continue
	fi
	echo "Running: $cmd"
	out=$(eval $cmd)
	echo "$out"
	jid=$(echo "$out" | awk '{print $NF}')
	if [[ -n "$jid" ]]; then
		JOBIDS+=("$jid")
	fi
done

if [ ${#JOBIDS[@]} -eq 0 ]; then
	echo "No jobs submitted. Exiting."
	exit 0
fi

ids=$(IFS=,; echo "${JOBIDS[*]}")
echo "Submitted job ids: $ids"

echo "Waiting for jobs to finish..."
while true; do
	remaining=$(squeue -h -j $ids | wc -l)
	if [ "$remaining" -eq 0 ]; then
		break
	fi
	sleep 30
done

echo "All jobs finished. Running combine_sweeps.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/combine_sweeps.py" --glob "./**/*_mode_psn_*.pkl" -o "./combined_sweeps"
echo "Combine step completed."

#sbatch --job-name=10 --export=ALL,N=10,mg_width_low=1,mg_width_high=5 fires_n_loop.sh
#sbatch --job-name=100 --export=ALL,N=100,mg_width_low=1,mg_width_high=5 fires_n_loop.sh
#sbatch --job-name=600 --export=ALL,N=600,mg_width_low=1,mg_width_high=5 fires_n_loop.sh
#sbatch --job-name=1000 --export=ALL,N=1000,mg_width_low=1,mg_width_high=5 fires_n_loop.sh

#sbatch --job-name=10 --export=ALL,N=10,mg_width_low=20,mg_width_high=40 fires_n_loop.sh
#sbatch --job-name=100 --export=ALL,N=100,mg_width_low=20,mg_width_high=40 fires_n_loop.sh
#sbatch --job-name=600 --export=ALL,N=600,mg_width_low=20,mg_width_high=40 fires_n_loop_sh
#sbatch --job-name=1000 --export=ALL,N=1000,mg_width_low=20,mg_width_high=40 fires_n_loop.sh
