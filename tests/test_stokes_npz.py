import os

import numpy as np
import pytest

from fires.utils.io import write_stokes_cube
from fires.utils.loaders import load_data
from fires.utils.stokes_npz import read_stokes_npz, write_stokes_npz

SIM_FILE = os.path.join(os.path.dirname(__file__), "..", "src", "fires", "config", "fires.toml")


@pytest.fixture
def cube():
    rng = np.random.default_rng(1)
    freq = 752.0 + np.arange(8)            # ascending
    time = 0.1 * np.arange(32)
    return rng.normal(size=(4, 8, 32)), freq, time


def test_round_trip_without_pickle(tmp_path, cube):
    dspec, freq, time = cube
    path = write_stokes_npz(tmp_path / "x_stokes.npz", dspec, freq, time, label="FRB 1",
                            dm=12.5, params={"a": np.float64(1.5), "b": [1, 2]})
    with np.load(path, allow_pickle=False) as f:
        assert str(f["format"]) == "stokes-dspec" and int(f["nchan"]) == 8
    out, f2, t2, meta = read_stokes_npz(path)
    np.testing.assert_array_equal(out, dspec)
    np.testing.assert_array_equal(f2, freq)
    np.testing.assert_array_equal(t2, time)
    assert meta["label"] == "FRB 1" and meta["dm"] == 12.5 and meta["params"] == {"a": 1.5, "b": [1, 2]}


def test_reads_dmop_style_archive_with_missing_v(tmp_path, cube):
    dspec, freq, time = cube
    np.savez(tmp_path / "d.npz", stokes=dspec[:3], pol_order="IQU", freq_mhz=freq, time_ms=time)
    out, _, _, _ = read_stokes_npz(tmp_path / "d.npz")
    np.testing.assert_array_equal(out[:3], dspec[:3])
    assert not out[3].any()


def test_shape_mismatch_rejected(tmp_path, cube):
    dspec, freq, time = cube
    with pytest.raises(ValueError):
        write_stokes_npz(tmp_path / "bad.npz", dspec, freq[:-1], time)


def test_load_data_npz_matches_npy(tmp_path, cube):
    dspec, freq, time = cube
    cube_dir = write_stokes_cube(dspec, freq, time, str(tmp_path), "SIM", source="test")
    from_npy = load_data(cube_dir, None, sim_file=SIM_FILE)
    from_npz = load_data(os.path.join(cube_dir, "SIM_stokes.npz"), None, sim_file=SIM_FILE)
    for a, b in zip(from_npy[:3], from_npz[:3]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(from_npy[1], freq)   # loader returns ascending frequency
    np.testing.assert_array_equal(from_npy[0], dspec)
