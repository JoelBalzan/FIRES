import logging
import os
import sys

def slurm_array_size():
    cnt = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if cnt is not None:
        return int(cnt)
    min_s = os.environ.get("SLURM_ARRAY_TASK_MIN")
    max_s = os.environ.get("SLURM_ARRAY_TASK_MAX")
    if max_s is not None:
        min_i = int(min_s) if min_s is not None else 0
        return int(max_s) - min_i + 1
    return int(os.environ.get("FIRESSWEEP_COUNT", "1"))

def slurm_array_id():
    min_s = os.environ.get("SLURM_ARRAY_TASK_MIN")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        if min_s is not None:
            return int(task_id) - int(min_s)
        return int(task_id)
    return int(os.environ.get("FIRESSWEEP_ID", "0"))

def slurm_chunk_xvals(xvals):
    _array_count = slurm_array_size()
    _array_id = slurm_array_id()
    if _array_count > 1:
        total = len(xvals)
        base = total // _array_count
        rem = total % _array_count
        if _array_id < rem:
            start_idx = _array_id * (base + 1)
            end_idx = start_idx + (base + 1)
        else:
            start_idx = rem * (base + 1) + (_array_id - rem) * base
            end_idx = start_idx + base
        if start_idx >= total or start_idx == end_idx:
            logging.info(f"Array task {_array_id}/{_array_count - 1}: no assigned xvals.")
            sys.exit(0)
        xvals = xvals[start_idx:min(end_idx, total)]
        logging.info(
            f"Array task {_array_id}/{_array_count - 1}: "
            f"processing {len(xvals)} sweep values (idx {start_idx}:{min(end_idx, total)} of {total})."
        )
    return xvals, _array_id, _array_count

def pool_workers_and_chunksize(n_cpus, n_tasks):
    tasks = max(1, int(n_tasks))
    req_workers = int(n_cpus) if n_cpus is not None else 1
    workers = max(1, min(req_workers, tasks))
    chunksize = max(1, min(4, tasks // max(1, workers * 16)))
    return workers, chunksize
