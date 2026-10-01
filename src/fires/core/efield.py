# -----------------------------------------------------------------------------
# efield.py
# FIRES: The Fast, Intense Radio Emission Simulator
#
# Stochastic complex electric-field realisation of the intrinsic emission.
#
# The 'psn' emission model builds the Stokes dynamic spectrum directly from the
# PSN intensity profiles. The 'efield' model instead treats the PSN profile as
# the *expected* intrinsic intensity envelope <I>(t, nu) and realises the actual
# instantaneous radiation as stochastic complex electric fields.
#
# For each (time, frequency) sample a 2x2 coherency (polarisation) matrix
#
#     J = 1/2 [[ I + Q ,  U - i V ],
#              [ U + i V ,  I - Q ]]
#
# is constructed from the desired mean Stokes vectors (using the FIRES Stokes
# convention, i.e. the injected vfrac appears directly as V/I).  A factorisation
# J = L L^dag is used to draw a pair of independent standard complex Gaussian
# samples
#
#     E = L z ,   z ~ CN(0, I),
#
# whose covariance/coherency equals J by construction.  This correctly
# represents both fully polarised (rank-1, p_L^2 + p_V^2 = 1) and partially
# polarised (rank-2, p_L^2 + p_V^2 < 1) emission, which cannot be captured by a
# single deterministic Jones vector.
#
# The instantaneous Stokes parameters are then computed from the fields
#
#     I = |Ex|^2 + |Ey|^2
#     Q = |Ex|^2 - |Ey|^2
#     U = 2 Re(Ex Ey*)
#     V = -2 Im(Ex Ey*)   =  2 Im(Ey Ex*)
#
# so that <I>, <Q>, <U>, <V> recover the requested mean Stokes state.  The sign
# convention for V is the one consistent with J = <E E^dag> (see module docstring
# of stokes_from_efield): a pure mode (1, i)/sqrt(2) carries V/I = +1.
#
# For the simplest case of an unpolarised mean state the field carries four
# independent real Gaussian degrees of freedom and the normalised instantaneous
# intensity I / <I> has a chi-square distribution with 4 degrees of freedom
# (equivalently a Gamma distribution with shape 2 and scale 1/2):
#
#     I = <I> * chi2_4 / 4 ,   <I / <I>> = 1 ,   std/mean = 1/sqrt(2).
#
# The draw is made once, on the summed mean Stokes dynamic spectrum (after all
# microshots, dispersion, scattering, RM and scintillation have acted on the
# mean state), independently for every time sample and frequency channel
# (Nyquist-sampled interpretation).  Overlapping microshots therefore do not
# average the noise down.  A pixel of bandwidth dnu and duration dt holds
# nsamp = dnu*dt independent field samples, so its self-noise is the average over
# nsamp draws (std/mean = 1/sqrt(2*nsamp) for unpolarised emission); see
# ``efield_from_mean_stokes``.  Pixels finer than Nyquist (nsamp < 1) are treated
# as nsamp = 1.
# -----------------------------------------------------------------------------


import numpy as np
from scipy import stats

# Samples with mean intensity below this fraction of the peak are left at zero field.
_I_REL_EPS = 1e-12


def stokes_from_efield(Ex, Ey):
    """Return instantaneous Stokes (I, Q, U, V) from complex E-field components.

    The V sign convention is tied to the coherency matrix form used by
    ``coherency_from_stokes``:

        J = 1/2 [[ I + Q ,  U - i V ],
                 [ U + i V ,  I - Q ]]  = <E E^dag>.

    With this convention ``V = -2 Im(Ex Ey*) = 2 Im(Ey Ex*)``, so that a pure
    mode ``(Ex, Ey) = (1, i)/sqrt(2)`` yields ``V/I = +1``.  This is the unique
    choice consistent with the FIRES convention that the injected ``vfrac``
    appears directly as the mean ``V/I`` of the realisation.
    """
    I = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    Q = np.abs(Ex) ** 2 - np.abs(Ey) ** 2
    U = 2.0 * np.real(Ex * np.conj(Ey))
    V = -2.0 * np.imag(Ex * np.conj(Ey))
    return I, Q, U, V


def coherency_from_stokes(I, Q, U, V):
    """Build the 2x2 coherency matrix J = 1/2 [[I+Q, U-iV], [U+iV, I-Q]].

    Returns an array of shape (..., 2, 2) real-valued, matching J = <E E^dag>:
      J[..., 0, 0] = (I + Q) / 2
      J[..., 0, 1] = (U - iV) / 2
      J[..., 1, 0] = (U + iV) / 2
      J[..., 1, 1] = (I - Q) / 2
    """
    I = np.asarray(I, dtype=float)
    Q = np.asarray(Q, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    J = np.zeros(np.broadcast_shapes(I.shape, Q.shape, U.shape, V.shape) + (2, 2), dtype=complex)
    J[..., 0, 0] = (I + Q) / 2.0
    J[..., 0, 1] = (U - 1j * V) / 2.0
    J[..., 1, 0] = (U + 1j * V) / 2.0
    J[..., 1, 1] = (I - Q) / 2.0
    return J


def _sqrt_coherency(I, Q, U, V, rng):
    """Hermitian square root L (n, 2, 2) of J for the active samples (L L^dag = J).

    J = I[(1+p)/2 e e^dag + (1-p)/2 e_perp e_perp^dag], p = |P|/I (clipped to 1),
    e = (cos t, sin t e^{i phi}), e_perp = (-conj(ey), conj(ex)).
    """
    P = np.sqrt(Q ** 2 + U ** 2 + V ** 2)
    p = np.minimum(P / I, 1.0)            # unphysical P > I is clipped to fully polarised
    pol = P > 0.0
    Pn = np.where(pol, P, 1.0)
    qhat, uhat, vhat = Q / Pn, U / Pn, V / Pn   # unit Stokes direction (normalised by |P|)
    ex = np.where(pol, np.sqrt(np.maximum((1.0 + qhat) / 2.0, 0.0)), 1.0)
    ey = np.where(pol, np.sqrt(np.maximum((1.0 - qhat) / 2.0, 0.0)) * np.exp(1j * np.arctan2(vhat, uhat)), 0.0)
    e = np.stack([ex, ey], axis=-1)
    ep = np.stack([-np.conj(ey), np.conj(ex)], axis=-1)
    s = np.sqrt(I)
    c1 = (s * np.sqrt((1.0 + p) / 2.0))[:, None, None]
    c2 = (s * np.sqrt((1.0 - p) / 2.0))[:, None, None]
    return c1 * e[:, :, None] * np.conj(e)[:, None, :] + c2 * ep[:, :, None] * np.conj(ep)[:, None, :]


def _active(I):
    # Samples with mean intensity below a tiny fraction of the peak are left at zero.
    return I > _I_REL_EPS * max(float(np.max(I, initial=0.0)), 0.0)


def _cn(draw, n):
    """Standard complex normal, E|z|^2 = 1."""
    return (draw.standard_normal(n) + 1j * draw.standard_normal(n)) / np.sqrt(2.0)


def generate_efield_from_stokes(I, Q, U, V, rng=None):
    """Draw one stochastic complex field pair (Ex, Ey) per sample, E = L z, z ~ CN(0, 1).

    I, Q, U, V are the desired *mean* Stokes parameters (broadcastable).  Samples with
    non-positive mean intensity get zero field.  Uses the global ``np.random`` state
    unless ``rng`` is given.  Returns complex arrays of the broadcast shape.
    """
    I, Q, U, V = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (I, Q, U, V)))
    draw = rng if rng is not None else np.random
    Ex = np.zeros(I.shape, dtype=complex)
    Ey = np.zeros(I.shape, dtype=complex)
    act = _active(I)
    n = int(act.sum())
    if n:
        L = _sqrt_coherency(I[act], Q[act], U[act], V[act], draw)
        z = np.stack([_cn(draw, n), _cn(draw, n)], axis=-1)
        E = np.einsum("nij,nj->ni", L, z)
        Ex[act], Ey[act] = E[:, 0], E[:, 1]
    return Ex, Ey


def efield_from_mean_stokes(I, Q, U, V, nsamp=1.0, rng=None):
    """Realised Stokes (I, Q, U, V) of a pixel averaging ``nsamp`` independent field samples.

    A pixel of bandwidth dnu and duration dt contains ``nsamp = dnu * dt`` independent
    (Nyquist) complex-field samples.  The averaged coherency is
    ``J_hat = L W L^dag`` with ``W ~ ComplexWishart(2, nsamp) / nsamp``, drawn exactly in
    O(1) per pixel by the Bartlett decomposition (non-integer nsamp = effective dof).
    ``nsamp = 1`` reproduces a single field draw (chi2_4/4 for unpolarised emission);
    fluctuations shrink as 1/sqrt(nsamp).  ``nsamp < 1`` (oversampled, correlated
    pixels) is clamped to 1: that correlation is not modelled.
    """
    I, Q, U, V = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (I, Q, U, V)))
    draw = np.random
    M = max(float(nsamp), 1.0)
    out = np.zeros((4,) + I.shape)
    act = _active(I)
    n = int(act.sum())
    if n == 0:
        return tuple(out)
    L = _sqrt_coherency(I[act], Q[act], U[act], V[act], draw)
    a2 = draw.gamma(M, size=n)               # |A00|^2 ~ Gamma(M)
    b2 = draw.gamma(M - 1.0, size=n) if M > 1.0 else np.zeros(n)   # |A11|^2 ~ Gamma(M-1)
    c = _cn(draw, n)
    W = np.empty((n, 2, 2), dtype=complex)
    W[:, 0, 0] = a2
    W[:, 0, 1] = np.sqrt(a2) * np.conj(c)
    W[:, 1, 0] = np.sqrt(a2) * c
    W[:, 1, 1] = np.abs(c) ** 2 + b2
    J = np.einsum("nij,njk,nlk->nil", L, W / M, np.conj(L))
    out[0][act] = (J[:, 0, 0] + J[:, 1, 1]).real
    out[1][act] = (J[:, 0, 0] - J[:, 1, 1]).real
    out[2][act] = 2.0 * J[:, 0, 1].real
    out[3][act] = -2.0 * J[:, 0, 1].imag
    return tuple(out)


def efield_chi2_statistics(samples, mean=None):
    """Quantify the intensity statistics of a sampled instantaneous intensity.

    Compares the empirical distribution of the normalised instantaneous
    intensity ``x = I / <I>`` against the expected four-dof complex-field result
    ``x ~ chi2_4 / 4`` (equivalently Gamma(shape=2, scale=1/2)).  For the
    simplest case (unpolarised mean state) the normalised intensity has
    ``<x> = 1`` and ``std(x) = 1/sqrt(2)``.

    Parameters
    ----------
    samples : array_like
        Instantaneous intensity samples (e.g. flattened I over a constant
        envelope).
    mean : float or None
        Expected mean intensity.  If None it is estimated from ``samples``.

    Returns
    -------
    stats : dict
        ``{'mean': <I>, 'std_over_mean': std(I)/<I>, 'chi2_k': 4,
         'ks_stat': ..., 'ks_p': ...}`` with the Kolmogorov-Smirnov test of
        ``I / <I>`` against ``chi2_4 / 4``.
    """
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return {"mean": np.nan, "std_over_mean": np.nan, "chi2_k": 4,
                "ks_stat": np.nan, "ks_p": np.nan}
    mu = float(np.mean(samples)) if mean is None else float(mean)
    if mu <= 0:
        return {"mean": mu, "std_over_mean": np.nan, "chi2_k": 4,
                "ks_stat": np.nan, "ks_p": np.nan}
    x = samples / mu
    std_over_mean = float(np.std(samples, ddof=1) / mu)
    ks_stat, ks_p = stats.kstest(x, lambda xx: stats.gamma.cdf(xx, a=2.0, scale=0.5))
    return {"mean": float(mu), "std_over_mean": std_over_mean, "chi2_k": 4,
            "ks_stat": float(ks_stat), "ks_p": float(ks_p)}