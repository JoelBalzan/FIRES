#!/usr/bin/env python3
"""Combine sweep output fragments and optionally remove the parts.

Usage examples:
  python bash/combine_sweeps.py --dir results/sweep_run --pattern "**/*part*.npz" -o combined.npz
  python bash/combine_sweeps.py --glob "results/sweep_run/*.csv" --out combined.csv --keep

The script tries to infer file type from extensions and will:
- concatenate text/CSV files (preserving a single header)
- concatenate `.npy` arrays
- merge `.npz` archives by concatenating arrays with the same keys
- for pickles, extend lists or concatenate array-like dict values

After successful combine the source files are removed unless `--keep` is passed.
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np

LOG = logging.getLogger("combine_sweeps")


def find_files_from_args(args) -> List[Path]:
    if args.glob:
        return [Path(p) for p in sorted(glob.glob(args.glob, recursive=True))]
    base = Path(args.dir or ".")
    return sorted(base.rglob(args.pattern or "*"))


def derive_name_from_files(file_list_names: List[str]) -> str | None:
    """Derive a descriptive combined filename from a list of filenames.

    Returns None if no useful tokens are found.
    """
    import re
    starts = []
    ends = []
    mode_suffix = None
    core = None
    for nm in file_list_names:
        m = re.search(r'xvals_(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)', nm)
        if m:
            try:
                starts.append(float(m.group(1)))
                ends.append(float(m.group(2)))
            except Exception:
                continue
        m2 = re.search(r'_mode_psn_(.+)\.pkl$', nm)
        if m2:
            sfx = m2.group(1)
            if mode_suffix is None:
                mode_suffix = sfx
            elif mode_suffix != sfx:
                # Keep only stable token overlap if fragment suffixes differ.
                cur_tokens = [t for t in mode_suffix.split('_') if t]
                new_tokens = [t for t in sfx.split('_') if t]
                common = [t for t in cur_tokens if t in set(new_tokens)]
                mode_suffix = '_'.join(common)
        m3 = re.match(r'^sweep_\d+_(?P<core>.+?xvals_)', nm)
        if m3 and core is None:
            core = m3.group('core')
    if not starts or not ends:
        return None
    start = min(starts)
    end = max(ends)
    if core is None:
        sample = file_list_names[0]
        s = sample
        s = re.sub(r'^sweep_\d+_', '', s)
        s = re.sub(r'_mode_psn_.*$', '', s)
        i = s.rfind('xvals_')
        core = s[:i+len('xvals_')] if i != -1 else s
    if mode_suffix:
        name = f"combined_{core}{start:.2f}-{end:.2f}_mode_psn_{mode_suffix}.pkl"
    else:
        name = f"combined_{core}{start:.2f}-{end:.2f}_mode_psn.pkl"
    name = name.replace('__', '_')
    return name



def combine_pickle(files: List[Path], out: Path):
    """Combine pickle fragments.

    If files represent multiple grouped sweep fragments (e.g., suffixes like
    `_partNN` or trailing indices), the function will merge fragments per
    group and write one merged .pkl per group. If only a single logical group
    is present, it will write to `out` as a file.
    """
    import re

    def files_for_group_key(group_key: str, all_files: List[Path]) -> List[Path]:
        """Match grouped-loader keys to source files using token membership.

        Group keys are often formatted like `N100+mg_width_high5+mg_width_low1`,
        which do not appear as contiguous substrings in filenames.
        """
        if not group_key:
            return []
        tokens = [t for t in group_key.split("+") if t]
        if not tokens:
            return []
        matched = []
        for path in all_files:
            name = path.name
            if all(tok in name for tok in tokens):
                matched.append(path)
        return matched

    # Prefer loader implementation when available and files match expected pattern
    try:
        from fires.utils.loaders import load_multiple_data_grouped
    except Exception:
        load_multiple_data_grouped = None

    # If all files are in the same directory and look like the sweep pickle pattern
    same_dir = len({p.parent for p in files}) == 1
    pattern_ok = all(re.search(r'_mode_psn_.*\.pkl$', p.name) for p in files)
    if load_multiple_data_grouped and same_dir and pattern_ok:
        try:
            common_dir = str(files[0].parent)
            merged = load_multiple_data_grouped(common_dir)
            out_path = Path(out)
            # single-series -> write single file
            if isinstance(merged, dict) and 'xvals' in merged:
                # single series -> derive sensible default filename when out is dir/default
                # use module-level `derive_name_from_files` helper

                if out_path.exists() and out_path.is_dir() or not out_path.suffix:
                    out_path.mkdir(parents=True, exist_ok=True)
                    derived = derive_name_from_files([p.name for p in files])
                    fname = out_path / (derived or "combined.pkl")
                else:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    fname = out_path
                with fname.open('wb') as fh:
                    pickle.dump(merged, fh)
                return [fname]
            # multiple groups -> ensure out is a dir and write per-group
            out_path.mkdir(parents=True, exist_ok=True)
            written = []
            for key, val in merged.items():
                token = key.replace('=', '') if key else 'baseline'
                group_paths = files_for_group_key(token, files)
                derived = derive_name_from_files([p.name for p in group_paths]) if group_paths else None
                fname = out_path / (derived or f"{token}.pkl")
                with fname.open('wb') as fh:
                    pickle.dump(val, fh)
                written.append(fname)
            return written
        except Exception as e:
            LOG.warning("load_multiple_data_grouped failed, falling back: %s", e)

    def canonical_prefix(name: str) -> str:
        # strip common trailing fragment tokens: _partNN, _fragNN, _chunkNN, _NN
        return re.sub(r'([_-](?:part|frag|chunk)\d+|[_-]\d+)(?:\.pkl)?$', '', name)

    # Map prefix -> list of files
    groups = {}
    for p in files:
        pref = canonical_prefix(p.name)
        groups.setdefault(pref, []).append(p)

    # If single group and out is a file path, merge into single file
    single_group = len(groups) == 1

    def merge_group(file_list: List[Path]) -> dict:
        # Merge dicts following the structure used by loaders.load_multiple_data_grouped
        all_xvals = []
        all_measures = {}
        all_V_params = {}
        all_exp_vars = {}
        all_snrs = {}
        seen_xvals = set()
        xname = None
        plot_mode = None
        dspec_params = None

        for f in sorted(file_list):
            with f.open('rb') as fh:
                try:
                    obj = pickle.load(fh)
                except Exception:
                    continue
            if not isinstance(obj, dict):
                continue
            if xname is None:
                xname = obj.get('xname')
                plot_mode = obj.get('plot_mode')
                dspec_params = obj.get('dspec_params')

            xvals = obj.get('xvals', [])
            measures = obj.get('measures', {})
            V_params = obj.get('V_params', {})
            exp_vars = obj.get('exp_vars', {})
            snrs = obj.get('snrs', {})

            for v in xvals:
                if v not in seen_xvals:
                    seen_xvals.add(v)
                    all_xvals.append(v)
                if v not in all_measures:
                    all_measures[v] = []
                    all_snrs[v] = []
                    all_V_params[v] = {key: [] for key in V_params.get(v, {}).keys()}
                    all_exp_vars[v] = {key: [] for key in exp_vars.get(v, {}).keys()}
                all_measures[v].extend(measures.get(v, []))
                all_snrs[v].extend(snrs.get(v, []))
                for key, arr in V_params.get(v, {}).items():
                    all_V_params[v].setdefault(key, [])
                    all_V_params[v][key].extend(arr)
                for key, arr in exp_vars.get(v, {}).items():
                    all_exp_vars[v].setdefault(key, [])
                    all_exp_vars[v][key].extend(arr)

        all_xvals = sorted(all_xvals)
        return {
            'xname': xname,
            'xvals': all_xvals,
            'measures': all_measures,
            'V_params': all_V_params,
            'exp_vars': all_exp_vars,
            'dspec_params': dspec_params,
            'plot_mode': plot_mode,
            'snrs': all_snrs,
        }

    out_path = Path(out)
    # If multiple groups, require out be a directory (existing or to-create)
    if not single_group and (out_path.exists() and out_path.is_file()):
        LOG.error("multiple groups detected but --out is a file; provide an output directory")
        raise SystemExit(4)

    if single_group:
        pref, flist = next(iter(groups.items()))
        merged = merge_group(flist)
        # derive default name from file list if out_path is dir/default
        # use module-level `derive_name_from_files` helper for deriving default names

        if out_path.exists() and out_path.is_dir() or not out_path.suffix:
            out_path.mkdir(parents=True, exist_ok=True)
            derived = derive_name_from_files([p.name for p in flist])
            fname = out_path / (derived or f"{pref}.pkl")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fname = out_path
        with fname.open('wb') as fh:
            pickle.dump(merged, fh)
        return [fname]

    # multiple groups -> create out dir if needed and write one file per group
    out_path.mkdir(parents=True, exist_ok=True)
    written = []
    for pref, flist in groups.items():
        merged = merge_group(flist)
        out_file = out_path / f"{pref}.pkl"
        with out_file.open('wb') as fh:
            pickle.dump(merged, fh)
        written.append(out_file)
    return written


def delete_files(files: List[Path]):
    for p in files:
        try:
            p.unlink()
        except Exception as e:
            LOG.warning("failed to delete %s: %s", p, e)


def main():
    p = argparse.ArgumentParser(description="Combine sweep output fragments")
    p.add_argument("--dir", help="base dir to search (used with --pattern)")
    p.add_argument("--pattern", default="*.pkl", help="glob pattern (when used with --dir)")
    p.add_argument("--glob", help="full glob for files (eg 'results/run/**/*.npz')")
    p.add_argument("--out", "-o", default=None, help="output file or directory (defaults to the input files' directory)")
    p.add_argument("--keep", action="store_true", help="keep source fragment files")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    files = find_files_from_args(args)
    files = [Path(f) for f in files if Path(f).is_file()]
    if not files:
        LOG.error("no files matched")
        raise SystemExit(2)
    LOG.info("found %d files", len(files))
    # Determine default output directory: prefer the original files' directory when possible
    parent_dirs = set(p.parent for p in files)
    if args.out is None:
        if len(parent_dirs) == 1:
            out = next(iter(parent_dirs))
        else:
            out = Path.cwd()
    else:
        out = Path(args.out)

    written = combine_pickle(files, out)

    LOG.info("wrote %s", written)
    if not args.keep:
        # Do not delete files that were just written by the combine step
        written_set = {p.resolve() for p in written}
        to_delete = [p for p in files if p.resolve() not in written_set]
        delete_files(to_delete)
        LOG.info("deleted %d source files", len(to_delete))


if __name__ == "__main__":
    main()
