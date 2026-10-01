import numpy as np
import pytest

from fires.core.efield import generate_efield_from_stokes, stokes_from_efield


@pytest.mark.parametrize("q,u,v,std", [(0, 0, 0, 2 ** -0.5), (0.6, 0, 0.8, 1.0), (0.3, 0.2, 0.1, None)])
def test_mean_stokes_and_stats(q, u, v, std):
    np.random.seed(0)
    n = 200000
    one = np.ones(n)
    I, Q, U, V = stokes_from_efield(*generate_efield_from_stokes(one, q * one, u * one, v * one))
    for got, want in zip((I, Q, U, V), (1, q, u, v)):
        assert abs(got.mean() - want) < 0.02
    if std is not None:
        assert abs(I.std() - std) < 0.02


def test_nonpositive_intensity_gives_zero_field():
    Ex, Ey = generate_efield_from_stokes(np.array([-1.0, 0.0]), np.zeros(2), np.zeros(2), np.zeros(2))
    assert not Ex.any() and not Ey.any()


@pytest.mark.parametrize("m", [1.0, 4.0, 100.0])
def test_nsamp_averaging(m):
    from fires.core.efield import efield_from_mean_stokes
    np.random.seed(0)
    n = 200000
    one = np.ones(n)
    I, Q, U, V = efield_from_mean_stokes(one, 0.6 * one, 0.0 * one, 0.8 * one, nsamp=m)
    assert abs(I.mean() - 1) < 0.02 and abs(Q.mean() - 0.6) < 0.02 and abs(V.mean() - 0.8) < 0.02
    assert abs(I.std() - m ** -0.5) < 0.02 * m ** -0.5 + 0.005
    I0 = efield_from_mean_stokes(one, 0 * one, 0 * one, 0 * one, nsamp=m)[0]
    assert abs(I0.std() - (2 * m) ** -0.5) < 0.02 * (2 * m) ** -0.5 + 0.005
