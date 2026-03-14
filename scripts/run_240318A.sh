#!/usr/bin/env bash
set -euo pipefail

# Submit jobs and wait for them to finish, then combine sweep fragments.
JOBIDS=()

# locate combine script (next to this run script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMBINE_PY="$SCRIPT_DIR/combine_sweeps.py"

# clean shutdown on Ctrl-C: print submitted jobs and exit
trap 'echo "Interrupted; submitted job ids: ${JOBIDS[*]}"; exit 2' INT TERM

# sbatch lines 
#sbatch --job-name=10 --export=ALL,N=10,mg_width_low=1,mg_width_high=5 fires_n_loop.sh
#sbatch --job-name=100 --export=ALL,N=100,mg_width_low=1,mg_width_high=5 fires_n_loop.sh
#sbatch --job-name=600 --export=ALL,N=600,mg_width_low=1,mg_width_high=5 fires_n_loop.sh
#sbatch --job-name=1000 --export=ALL,N=1000,mg_width_low=1,mg_width_high=5 fires_n_loop.sh

#sbatch --job-name=10 --export=ALL,N=10,mg_width_low=20,mg_width_high=40 fires_n_loop.sh
#sbatch --job-name=100 --export=ALL,N=100,mg_width_low=20,mg_width_high=40 fires_n_loop.sh
#sbatch --job-name=600 --export=ALL,N=600,mg_width_low=20,mg_width_high=40 fires_n_loop.sh
#sbatch --job-name=1000 --export=ALL,N=1000,mg_width_low=20,mg_width_high=40 fires_n_loop.sh


for cmd in \
	"sbatch --job-name=5 --export=ALL,N=5 fires_loop_240318A.sh" \
	"sbatch --job-name=10 --export=ALL,N=10 fires_loop_240318A.sh" \
  	"sbatch --job-name=20 --export=ALL,N=20 fires_loop_240318A.sh" \
  	"sbatch --job-name=100 --export=ALL,N=100 fires_loop_240318A.sh" \
  	"sbatch --job-name=1000 --export=ALL,N=1000 fires_loop_240318A.sh"; do
	if [[ "$cmd" =~ ^# ]]; then
		continue
	fi

	echo "Running: $cmd"
	out=$(eval $cmd)
	echo "$out"
	# get last token (job id)
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
	# squeue returns headerless lines with -h; if none remain, break
	remaining=$(squeue -h -j $ids | wc -l)
	if [ "$remaining" -eq 0 ]; then
		break
	fi
	sleep 30
done

echo "All jobs finished. Running combine_sweeps.py"
python3 "$COMBINE_PY"
echo "Combine step completed."