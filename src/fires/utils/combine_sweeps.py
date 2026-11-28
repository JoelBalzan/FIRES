#!/usr/bin/env python3
"""Combine sweep output fragments and optionally remove the parts.

Usage examples:
  python bash/combine_sweeps.py --dir results/sweep_run --pattern "**/*part*.pkl" -o combined.pkl
  python bash/combine_sweeps.py --glob "results/sweep_run/*.pkl" --out combined.pkl --keep

The script tries to infer file type from extensions and will:
- for pickles, extend lists or concatenate array-like dict values

After successful combine the source files are removed unless `--keep` is passed.
"""
from __future__ import annotations

import argparse
import glob
import logging
import pickle
import re
from pathlib import Path
from typing import List

from fires.utils.loaders import load_multiple_data_grouped, merge_sweep_results


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


def _files_for_group_key(group_key: str, all_files: List[Path]) -> List[Path]:
    """Match grouped-loader keys to source files using token membership.

    Group keys are often formatted like `N100+mg_width_high5+mg_width_low1`,
    which do not appear as contiguous substrings in filenames.
    """
    tokens = [t for t in group_key.split("+") if t] if group_key else []
    return [path for path in all_files if all(tok in path.name for tok in tokens)] if tokens else []


def _canonical_prefix(name: str) -> str:
    """Strip common trailing fragment tokens: _partNN, _fragNN, _chunkNN, _NN."""
    return re.sub(r'([_-](?:part|frag|chunk)\d+|[_-]\d+)(?:\.pkl)?$', '', name)


def _single_output_path(out_path: Path, source_names: List[str], fallback_name: str) -> Path:
    """Output file for a single merged series: a derived name inside out_path if it is a directory, else out_path."""
    if out_path.exists() and out_path.is_dir() or not out_path.suffix:
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path / (derive_name_from_files(source_names) or fallback_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _dump(obj, fname: Path) -> Path:
    with fname.open('wb') as fh:
        pickle.dump(obj, fh)
    return fname


def _load_fragments(file_list: List[Path]):
    """Unpickle fragment files in name order, skipping any that fail to load."""
    for f in sorted(file_list):
        with f.open('rb') as fh:
            try:
                yield pickle.load(fh)
            except Exception:
                continue


def combine_pickle(files: List[Path], out: Path):
    """Combine pickle fragments.

    Files in one directory named like sweep outputs (`..._mode_psn_...pkl`) are grouped by
    override parameters with loaders.load_multiple_data_grouped. Otherwise fragments are
    grouped by name with `_partNN`-style suffixes removed. One logical group is written to
    `out` (a file, or a derived name inside a directory); several groups go one file each
    into the directory `out`. Returns the written paths.
    """
    out_path = Path(out)

    same_dir = len({p.parent for p in files}) == 1
    pattern_ok = all(re.search(r'_mode_psn_.*\.pkl$', p.name) for p in files)
    if same_dir and pattern_ok:
        try:
            merged = load_multiple_data_grouped(str(files[0].parent))
            if isinstance(merged, dict) and 'xvals' in merged:
                return [_dump(merged, _single_output_path(out_path, [p.name for p in files], "combined.pkl"))]
            out_path.mkdir(parents=True, exist_ok=True)
            written = []
            for key, val in merged.items():
                token = key.replace('=', '') if key else 'baseline'
                group_paths = _files_for_group_key(token, files)
                derived = derive_name_from_files([p.name for p in group_paths]) if group_paths else None
                written.append(_dump(val, out_path / (derived or f"{token}.pkl")))
            return written
        except Exception as e:
            LOG.warning("load_multiple_data_grouped failed, falling back: %s", e)

    groups = {}
    for p in files:
        groups.setdefault(_canonical_prefix(p.name), []).append(p)

    if len(groups) > 1 and out_path.exists() and out_path.is_file():
        LOG.error("multiple groups detected but --out is a file; provide an output directory")
        raise SystemExit(4)

    if len(groups) == 1:
        pref, flist = next(iter(groups.items()))
        merged = merge_sweep_results(_load_fragments(flist))
        return [_dump(merged, _single_output_path(out_path, [p.name for p in flist], f"{pref}.pkl"))]

    out_path.mkdir(parents=True, exist_ok=True)
    return [_dump(merge_sweep_results(_load_fragments(flist)), out_path / f"{pref}.pkl")
            for pref, flist in groups.items()]


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
