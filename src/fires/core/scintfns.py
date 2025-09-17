import numpy as np
from .genfns import _init_seed
def apply_scintillation(dspec, freq_mhz, time_ms,
									   z_scr_pc=100.0,
									   v_eff_kms=(50.0, 0.0),
									   rdiff_km=1e4,
									   aniso_ratio=1.0,
									   aniso_pa_deg=0.0,
									   seed=None):
	"""
	Multiply FIRES dynspec (4, nchan, ntime) by a scintillation gain G(f,t)
	computed from a thin Kolmogorov phase screen using scintools.

	Notes:
	- This assumes scalar scintillation (same gain for I,Q,U,V).
	"""
	from scintools.scint_sim import Simulation

	c = 2.99792458e8
	pc_m = 3.085677581e16
	z_m = z_scr_pc * pc_m

	freq_hz = np.asarray(freq_mhz) * 1e6
	lam = c / freq_hz
	time_s = np.asarray(time_ms) * 1e-3

	# Screen sampling (set field-of-view to cover motion across the screen)
	# Choose pixel scale so that r_F and r_diff are sampled; keep it simple:
	# nx, ny sizes and dx, dy in meters on the screen plane
	nx = 2048
	ny = 2048
	# Fresnel scale at band center
	lam0 = c / np.mean(freq_hz)
	rF0 = np.sqrt(lam0 * z_m / (2.0 * np.pi))
	dx = dy = rF0 / 8.0  # oversample Fresnel scale
	seed = _init_seed(seed, False)

	# Build Kolmogorov phase screen with specified diffractive scale (rdiff sets phase variance normalization)
	screen = Simulation(nx=nx, ny=ny, dx=dx, dy=dy,
								   rf=rdiff_km * 1e3,
								   ar=aniso_ratio,
								   psi=np.deg2rad(aniso_pa_deg),
								   seed=seed)
	screen.generate()  # φ(x,y)

	# Effective transverse velocity at the screen plane
	vx, vy = v_eff_kms
	v_mps = np.array([vx, vy]) * 1e3

	# Build gain cube G(f,t): sample the field along the screen trajectory for each time, with Fresnel propagation per frequency
	nchan = freq_hz.size
	ntime = time_s.size
	G = np.empty((nchan, ntime), dtype=float)

	# Choose a reference screen point where the line-of-sight hits at t=0
	x0 = (nx // 2) * dx
	y0 = (ny // 2) * dy

	# Precompute screen coords vs time
	xt = x0 + v_mps[0] * (time_s - time_s[0])
	yt = y0 + v_mps[1] * (time_s - time_s[0])

	# Wrap into screen bounds
	def idx_wrap(x, dx, n):
		i = np.floor(x / dx).astype(int) % n
		return np.clip(i, 0, n - 1)

	ix_t = idx_wrap(xt, dx, nx)
	iy_t = idx_wrap(yt, dy, ny)

	# For each frequency, compute Fresnel field from local screen patch at (ix_t, iy_t)
	# Here we use a very simplified proxy: evaluate the complex field as exp(i φ) at the sampled points
	# and apply a frequency-dependent Fresnel smoothing. Replace with scintools’ fresnel_field for accuracy.
	for ci, f in enumerate(freq_hz):
		lam_c = c / f
		# Fresnel scale at this frequency
		rF = np.sqrt(lam_c * z_m / (2.0 * np.pi))

		# Simple local sampling (you can replace this with a small windowed Fresnel propagation around each (ix,iy))
		phi_t = screen.phi[iy_t, ix_t]
		E_t = np.exp(1j * phi_t)
		# Optionally, smooth in time on scale set by rF / |v_eff|
		if np.linalg.norm(v_mps) > 0:
			tF = rF / np.linalg.norm(v_mps)
			# causal Gaussian smoothing length in samples
			if tF > 0 and ntime > 3:
				from scipy.ndimage import gaussian_filter1d
				sig = tF / np.mean(np.diff(time_s))
				Re = gaussian_filter1d(E_t.real, sig, mode='wrap')
				Im = gaussian_filter1d(E_t.imag, sig, mode='wrap')
				E_t = Re + 1j * Im

		I_t = np.abs(E_t) ** 2
		# Normalize to unit mean (so modulation index is controlled by screen strength)
		G[ci] = I_t / np.nanmean(I_t)

	# Apply scalar gain to all Stokes
	dspec_scint = dspec.copy()
	dspec_scint *= G[None, :, :]  # shape (1,nchan,ntime) broadcast to (4,nchan,ntime)
	return dspec_scint, G
