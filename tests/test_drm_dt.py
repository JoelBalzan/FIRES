import numpy as np

from fires.core.dspec import scatter_dspec


def _delta_pulse():
    f = np.linspace(760.0, 1080.0, 5)
    t = np.arange(0.0, 20.0, 0.0167)
    X = np.zeros((4, len(f), len(t)))
    X[0, :, 30] = 1.0
    X[1, :, 30] = 1.0          # fully linearly polarised, PA = 0 at all frequencies
    return f, t, X


def test_drm_dt_zero_is_unchanged():
    f, t, X = _delta_pulse()
    dt = t[1] - t[0]
    tau = 1.45 * (f / 919.5) ** -4.19
    a = scatter_dspec(X, dt, tau)
    b = scatter_dspec(X, dt, tau, drm_dt=0.0, freq_mhz=f)
    assert np.array_equal(a, b)


def test_drm_dt_rotates_tail_pa_linearly_in_lambda_sq():
    f, t, X = _delta_pulse()
    dt = t[1] - t[0]
    tau = 1.45 * (f / 919.5) ** -4.19
    drm = -0.8
    a = scatter_dspec(X, dt, tau)
    c = scatter_dspec(X, dt, tau, drm_dt=drm, freq_mhz=f)
    lam2 = (299.792458 / f) ** 2
    for k in range(len(f)):
        m = (t > t[30] + 0.3) & (t < t[30] + 3.0 * tau[k])
        pa = 0.5 * np.unwrap(np.arctan2(c[2, k, m], c[1, k, m]))
        slope = np.polyfit(t[m], np.rad2deg(pa), 1)[0]
        assert np.isclose(slope, np.rad2deg(drm * lam2[k]), rtol=1e-3)
        assert np.allclose(c[0, k], a[0, k]) and np.allclose(c[3, k], a[3, k])
