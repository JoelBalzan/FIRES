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
# The stochastic draws are independent for every time sample and frequency
# channel (Nyquist-sampled complex-field interpretation); no temporal or
# frequency correlation is introduced here.
# -----------------------------------------------------------------------------


import numpy as np
from scipy import stats

# Square-root threshold below which a state is treated as unpolarised.  The
# polarised eigenbasis becomes numerically degenerate as the polarisation
# fraction p -> 0.
_POL_FRAC_EPS = 1e-12


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


def generate_efield_from_stokes(I, Q, U, V, rng=None):
    """Draw stochastic complex fields (Ex, Ey) from a desired mean Stokes state.

    The arrays I, Q, U, V give the desired *mean* Stokes parameters at each
    sample (e.g. shape (n_freq, n_time)).  For every sample an independent
    standard complex Gaussian pair is drawn, and the field is set via the
    coherency-matrix factorisation J = L L^dag so that

        <I> = I0 ,  <Q> = Q0 ,  <U> = U0 ,  <V> = V0

    over an ensemble of realisations.  The realisation is implemented with the
    closed-form eigen-decomposition of the 2x2 coherency matrix:

        J = I [ (1+p)/2 e e^dag  +  (1-p)/2 e_perp e_perp^dag ]

    with total polarisation fraction p = sqrt(Q^2+U^2+V^2)/I and a normalised
    pure-state Jones vector e matching the polarised component.  The unpolarised
    (quantum) component contributes a rank-2 identity term; together they give
    the correct statistics for both fully and partially polarised emission.

    Parameters
    ----------
    I, Q, U, V : array_like
        Desired mean Stokes parameters (broadcastable to a common shape).
    rng : np.random.Generator or None
        If provided, used for the Gaussian draws; otherwise the global
        ``np.random`` state is used (consistent with the rest of FIRES).

    Returns
    -------
    (Ex, Ey) : tuple of ndarray
        Complex field components of the given shape.
    """
    I = np.asarray(I, dtype=float)
    Q = np.asarray(Q, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    shape = np.broadcast_shapes(I.shape, Q.shape, U.shape, V.shape)
    I = np.broadcast_to(I, shape)
    Q = np.broadcast_to(Q, shape)
    U = np.broadcast_to(U, shape)
    V = np.broadcast_to(V, shape)

    if rng is None:
        # Standard complex Gaussian: E[|z|^2] = 1, E[z^2] = 0.
        zx = (np.random.standard_normal(shape) + 1j * np.random.standard_normal(shape)) / np.sqrt(2.0)
        zy = (np.random.standard_normal(shape) + 1j * np.random.standard_normal(shape)) / np.sqrt(2.0)
    else:
        zx = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)
        zy = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)

    Ipos = np.maximum(I, 0.0)

    # Total polarisation fraction (clipped to the physical limit I^2 >= P^2).
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.sqrt(Q ** 2 + U ** 2 + V ** 2) / np.maximum(Ipos, 1e-300)
    p = np.clip(p, 0.0, 1.0)

    # Fully / partially polarised branch: J = I[(1+p)/2 e e^dag + (1-p)/2 e_perp e_perp^dag].
    # For a pure state with Stokes direction (q, u, v) = (Q, U, V)/(p I):
    #   e = (cos(theta), sin(theta) exp(i phi)) with
    #   cos(2 theta) = q,  sin(2 theta) = sqrt(u^2 + v^2),  phi = atan2(v, u).
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denom = np.maximum(Ipos * p, 1e-300)
        qhat = Q / denom
        uhat = U / denom
        vhat = V / denom

        sin2t = np.sqrt(np.maximum(uhat ** 2 + vhat ** 2, 0.0))
        cos_t = np.sqrt(np.maximum((1.0 + qhat) / 2.0, 0.0))
        sin_t = np.sqrt(np.maximum((1.0 - qhat) / 2.0, 0.0))
        phi = np.arctan2(vhat, uhat)

        ex = cos_t
        ey = sin_t * np.exp(1j * phi)

        c1 = np.sqrt((1.0 + p) / 2.0)
        c2 = np.sqrt((1.0 - p) / 2.0)

        # z projected on the pure eigenmode and its orthogonal complement.
        a1 = np.conj(ex) * zx + np.conj(ey) * zy             # e^dag z
        a2 = -ey * zx + ex * zy                              # e_perp^dag z, e_perp = (-ey*, ex*)

        Ex = np.sqrt(Ipos) * (c1 * ex * a1 + c2 * (-np.conj(ey)) * a2)
        Ey = np.sqrt(Ipos) * (c1 * ey * a1 + c2 * (np.conj(ex)) * a2)

    # Unpolarised (or vanishing-intensity) branch: J = (I/2) Id.
    iso = (p < _POL_FRAC_EPS) | (Ipos <= 0.0)
    if np.any(iso):
        Ex = np.where(iso, np.sqrt(Ipos / 2.0) * zx, Ex)
        Ey = np.where(iso, np.sqrt(Ipos / 2.0) * zy, Ey)

    return Ex, Ey


def efield_from_mean_stokes(I, Q, U, V, rng=None):
    """Generate (Ex, Ey) and return (I, Q, U, V), the instantaneous Stokes.

    Convenience wrapper around :func:`generate_efield_from_stokes` and
    :func:`stokes_from_efield` for use by the efield emission mode.
    """
    Ex, Ey = generate_efield_from_stokes(I, Q, U, V, rng=rng)
    return stokes_from_efield(Ex, Ey)


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