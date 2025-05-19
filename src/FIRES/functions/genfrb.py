# Simulating scattering
# AB, Sep 2024
# -------------------------- Import modules ---------------------------
from importlib.resources import files
import matplotlib.pyplot as plt
import numpy as np
from FIRES.functions.genfns import *
from FIRES.utils.utils import *
import os
import pickle as pkl  # Import pickle
from concurrent.futures import ProcessPoolExecutor
import functools
from itertools import product
from tqdm import tqdm
import inspect

from FIRES.functions.plotmodes import plot_modes
from FIRES.functions.basicfns import scatter_stokes_chan, add_noise_to_dynspec

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
obs_params_path = os.path.join(parent_dir, "utils/obsparams.txt")
gauss_params_path = os.path.join(parent_dir, "utils/gparams.txt")


def subtract_baseline_offpulse(dspec, off_start, off_end, axis=-1, method='median'):
	"""
	Subtracts the baseline using a specific off-pulse window, specified as fractions (0-1) of the axis length.
	Args:
		dspec: np.ndarray, shape (nstokes, nchan, ntime)
		off_start: float, start of off-pulse window as fraction (0-1)
		off_end: float, end of off-pulse window as fraction (0-1)
		axis: int, axis along which to estimate the baseline (default: time axis)
		method: 'median' or 'mean'
	Returns:
		dspec_baseline_subtracted: np.ndarray, same shape as dspec
	"""
	axis_len = dspec.shape[axis]
	start_idx = int(np.floor(off_start * axis_len))
	end_idx = int(np.floor(off_end * axis_len))
	slicer = [slice(None)] * dspec.ndim
	slicer[axis] = slice(start_idx, end_idx)
	offpulse_region = dspec[tuple(slicer)]

	if method == 'median':
		baseline = np.nanmedian(offpulse_region, axis=axis, keepdims=True)
	elif method == 'mean':
		baseline = np.nanmean(offpulse_region, axis=axis, keepdims=True)
	else:
		raise ValueError("method must be 'median' or 'mean'")
	return dspec - baseline


def find_offpulse_window(profile, window_frac=0.2):
	"""
	Find the off-pulse window in a 1D profile by sliding a window and finding the minimum mean.
	Args:
		profile: 1D numpy array (summed over frequency and Stokes if needed)
		window_frac: Fractional width of the window (e.g., 0.2 for 20%)
	Returns:
		off_start_frac, off_end_frac: Start and end as fractions (0-1)
	"""
	nbin = len(profile)
	win_size = int(window_frac * nbin)
	min_mean = np.inf
	min_start = 0
	for i in range(nbin - win_size + 1):
		win_mean = np.mean(profile[i:i+win_size])
		if win_mean < min_mean:
			min_mean = win_mean
			min_start = i
	off_start_frac = min_start / nbin
	off_end_frac = (min_start + win_size) / nbin
	return off_start_frac, off_end_frac


def select_offpulse_window(profile):
    """
    Plot the profile and let the user select the off-pulse window by clicking twice.
    Only mouse clicks are accepted.
    Returns:
        off_start_frac, off_end_frac: Start and end as fractions (0-1)
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(profile, lw=0.5, color='k')
    ax.set_xlim(0, len(profile))
    ax.set_title("Click twice to select the off-pulse window")

    clicks = []

    def onclick(event):
        # Only accept mouse button presses inside the axes
        if event.inaxes == ax and event.button in [1, 2, 3]:
            ax.axvline(event.xdata, color='r', linestyle='--')
            fig.canvas.draw()
            clicks.append(event.xdata)
            if len(clicks) == 2:
                fig.canvas.mpl_disconnect(cid)
                plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(clicks) < 2:
        raise RuntimeError("Fewer than two mouse clicks registered.")

    idx = sorted([int(round(x)) for x in clicks])
    nbin = len(profile)
    off_start_frac = max(0, idx[0] / nbin)
    off_end_frac = min(1, idx[1] / nbin)
    return off_start_frac, off_end_frac


def scatter_loaded_dynspec(dspec, freq_mhz, time_ms, tau_ms, sc_idx, ref_freq_mhz):
	"""
	Scatter all Stokes channels of a loaded dynamic spectrum.
	Args:
		dspec: 3D array [4, nchan, ntime] (Stokes I, Q, U, V)
		freq_mhz: Frequency array (nchan,)
		time_ms: Time array (ntime,)
		tau_ms: Scattering timescale (float)
		sc_idx: Scattering index (float)
		ref_freq_mhz: Reference frequency (float)
	Returns:
		dspec_scattered: Scattered dynamic spectrum (same shape as input)
	"""
	dspec_scattered = dspec.copy()
	for stokes_idx in range(dspec.shape[0]):  # Loop over I, Q, U, V
		for c in range(len(freq_mhz)):
			dspec_scattered[stokes_idx, c] = scatter_stokes_chan(
				dspec[stokes_idx, c], freq_mhz[c], time_ms, tau_ms, sc_idx, ref_freq_mhz
			)
	return dspec_scattered


def load_data(data, freq_mhz, time_ms):
	if isinstance(data, str):
		if os.path.isdir(data):
			# Expect exactly one file for each Stokes parameter: I, Q, U, V
			stokes_labels = ['I', 'Q', 'U', 'V']
			stokes_files = []
			for s in stokes_labels:
				# Find the file for this Stokes parameter
				matches = [f for f in os.listdir(data) if f.endswith(f'ds{s}_crop.npy')]
				#matches = [f for f in os.listdir(data) if f.endswith(f'{s}.npy')]
				
				if len(matches) != 1:
					raise ValueError(f"Expected one file for Stokes {s}, found {len(matches)}")
				stokes_files.append(os.path.join(data, matches[0]))
			# Load and stack
			arrs = [np.load(f) for f in stokes_files]  # each (nchan, ntime)
			dspec = np.stack(arrs, axis=0)  # shape: (4, nchan, ntime)
			
			summary_file = "parameters.txt"
			summary = get_parameters(os.path.join(data, summary_file))
			cfreq_mhz = float(summary['centre_freq_frb'])
			bw_mhz = float(summary['bw'])
			freq_mhz = np.linspace(cfreq_mhz - bw_mhz / 2, cfreq_mhz + bw_mhz / 2, dspec.shape[1])
			time_res_ms = 0.001 # Convert to milliseconds
			time_ms = np.arange(0, dspec.shape[2] * time_res_ms, time_res_ms)
			print(f"Loaded data from {data} with frequency range: {freq_mhz[0]} - {freq_mhz[-1]} MHz")
			
		elif data.endswith('.pkl'):
			with open(data, 'rb') as f:
				dspec = pkl.load(f)['dspec']
		elif data.endswith('.npy'):
			arr = np.load(data)
			if arr.ndim == 2:
				dspec = arr[np.newaxis, ...]  # shape: (1, nchan, ntime)
			else:
				dspec = arr
		else:
			raise ValueError("Unsupported file format: only .pkl and .npy are supported")
	else:
		raise ValueError("Unsupported data type for 'data'")

	#start, stop = find_offpulse_window(np.nanmean(dspec[0], axis=0), window_frac=0.2)
	start, stop = select_offpulse_window(np.nanmean(dspec[0], axis=0))
	dspec = subtract_baseline_offpulse(dspec, start, stop, axis=-1, method='mean')

	return dspec, freq_mhz, time_ms


def load_multiple_data(data):
	# store aggregated data
	all_scatter_ms = []
	all_vals = {}  
	all_errs = {}  
	all_var_PA_microshots = {}  
	width_ms = None  # This will be extracted from the first file

	file_names = [file_name for file_name in sorted(os.listdir(data)) if file_name.endswith(".pkl")]

	# Sort file_names by the number after the first '_'
	file_names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

	for file_name in sorted(file_names):
		with open(file_name, "rb") as f:
			scatter_ms, vals, errs, width, var_PA_microshots = pkl.load(f)

		for s_val in scatter_ms:
			if s_val not in all_vals:
				all_vals[s_val] = []
				all_errs[s_val] = []
				all_var_PA_microshots[s_val] = []

			all_vals[s_val].extend(vals[s_val])
			all_errs[s_val].extend(errs[s_val])
			all_var_PA_microshots[s_val].extend(var_PA_microshots[s_val])

		all_scatter_ms.extend(scatter_ms)

		# Set width_ms (assume it's the same for all files)
		if width_ms is None:
			width_ms = width
		
		vals = all_vals
		errs = all_errs
		var_PA_microshots = all_var_PA_microshots
		scatter_ms = all_scatter_ms
		width = width_ms
		
	return scatter_ms, vals, errs, width, var_PA_microshots

def generate_dynspec(mode, s_val, plot_multiple_tau, **params):
    """Generate dynamic spectrum based on mode."""
    s_val = s_val if plot_multiple_tau else params["tau_ms"]

    # Choose the correct function
    if mode == 'gauss':
        dynspec_func = gauss_dynspec
    else:  # mode == 'mgauss'
        dynspec_func = m_gauss_dynspec

    # Get the argument names for the selected function
    sig = inspect.signature(dynspec_func)
    allowed_args = set(sig.parameters.keys())

    # Always pass tau_ms as s_val
    params_filtered = {k: v for k, v in params.items() if k in allowed_args and k != "tau_ms"}
    return dynspec_func(**params_filtered, tau_ms=s_val)


def process_task(task, mode, plot_mode, **params):
	"""
	Process a single task (combination of timescale and realization).
	Dynamically uses the provided process_func for mode-specific processing.
	"""
	s_val, realization = task
	current_seed = params["seed"] + realization if params["seed"] is not None else None
	params["seed"] = current_seed
	
	requires_multiple_tau = plot_mode.requires_multiple_tau

	# Generate dynamic spectrum
	dspec, PA_microshot = generate_dynspec(
		mode=mode,
		s_val=s_val,
		plot_multiple_tau=requires_multiple_tau,
		**params
	)

	process_func = plot_mode.process_func
	# Use the provided process_func for mode-specific processing
	result, result_err = process_func(dspec, params["freq_mhz"], params["time_ms"], params["rm"], params["plot_window"])

	return s_val, result, result_err, PA_microshot


def generate_frb(data, scatter_ms, frb_id, out_dir, mode, n_gauss, seed, nseed, width_range, save,
				 obs_file, gauss_file, noise, scatter, n_cpus, plot_mode, plot_window):
	"""
	Generate a simulated FRB with a dispersed and scattered dynamic spectrum.
	"""
	obs_params = get_parameters(obs_file)

	# Extract frequency and time parameters
	f_start = float(obs_params['f0'])
	f_end = float(obs_params['f1'])
	t_start = float(obs_params['t0'])
	t_end = float(obs_params['t1'])
	f_res = float(obs_params['f_res'])
	t_res = float(obs_params['t_res'])

	scatter_idx = float(obs_params['scattering_index'])
	ref_freq = float(obs_params['reference_freq'])

	# Generate frequency and time arrays
	freq_mhz = np.arange(f_start, f_end + f_res, f_res, dtype=float)
	time_ms = np.arange(t_start, t_end + t_res, t_res, dtype=float)

	# Load Gaussian parameters
	gauss_params = np.loadtxt(gauss_file)

	# Extract Gaussian parameters
	t0 = gauss_params[:, 0]
	width = gauss_params[:, 1]
	peak = gauss_params[:, 2]
	spec_idx = gauss_params[:, 3]
	dm = gauss_params[:, 4]
	rm = gauss_params[:, 5]
	pol_angle = gauss_params[:, 6]
	lin_pol = gauss_params[:, 7]
	circ_pol = gauss_params[:, 8]
	delta_pa = gauss_params[:, 9]
	band_center = gauss_params[:, 10]
	band_width = gauss_params[:, 11]

	# Create dynamic spectrum parameters
	dspec_params = DynspecParams(
		freq_mhz=freq_mhz,
		time_ms=time_ms,
		time_res_ms=t_res,
		spec_idx=spec_idx,
		peak_amp=peak,
		width_ms=width,
		loc_ms=t0,
		dm=dm,
		pol_angle=pol_angle,
		lin_pol_frac=lin_pol,
		circ_pol_frac=circ_pol,
		delta_pol_angle=delta_pa,
		rm=rm,
		seed=seed,
		nseed=nseed,
		noise=noise,
		scatter=scatter,
		tau_ms=scatter_ms,
		sc_idx=scatter_idx,
		ref_freq_mhz=ref_freq,
		num_micro_gauss=n_gauss,
		width_range=width_range,
		band_centre_mhz=band_center,
		band_width_mhz=band_width,
		plot_window=plot_window,
	)

	if (lin_pol + circ_pol).any() > 1.0:
		print("WARNING: Linear and circular polarization fractions sum to more than 1.0.\n")

	if plot_mode.requires_multiple_tau == False:
		
		if data != None:
			dspec, freq_mhz, time_ms = load_data(data, freq_mhz, time_ms)
			if scatter and scatter_ms > 0:
				dspec = scatter_loaded_dynspec(dspec, freq_mhz, time_ms, scatter_ms, scatter_idx, ref_freq)
			if noise > 0:
				dspec = add_noise_to_dynspec(dspec, peak_amp = peak, SNR = noise)
	
	
		else:
			dspec, _ = generate_dynspec(
			mode=mode,
			s_val=None,
			plot_multiple_tau=False,
			**dspec_params._asdict()
		)
		_, _, _, noise_spec = process_dynspec(dspec, freq_mhz, time_ms, rm)
		frb_data = simulated_frb(
			frb_id, freq_mhz, time_ms, scatter_ms, scatter_idx, gauss_params, obs_params, dspec
		)
		if save:
			out_file = f"{out_dir}{frb_id}_sc_{scatter_ms:.2f}.pkl"
			with open(out_file, 'wb') as frb_file:
				pkl.dump(frb_data, frb_file)
		return frb_data, noise_spec, rm

	else:
		if data is not None:
			scatter_ms, vals, errs, width, var_PA_microshots = load_multiple_data(data)                
				
		else:
			# Create a list of tasks (timescale, realization)
			tasks = list(product(scatter_ms, range(nseed)))

			with ProcessPoolExecutor(max_workers=n_cpus) as executor:
				partial_func = functools.partial(
					process_task,
					mode=mode,
					plot_mode=plot_mode,
					**dspec_params._asdict()
				)

				results = list(tqdm(executor.map(partial_func, tasks),
									total=len(tasks),
									desc="Processing scattering timescales and realizations"))

			# Aggregate results by timescale
			vals = {s_val: [] for s_val in scatter_ms}
			errs = {s_val: [] for s_val in scatter_ms}
			var_PA_microshots = {s_val: [] for s_val in scatter_ms}

			for s_val, val, err, PA_microshot in results:
				vals[s_val].append(val)
				errs[s_val].append(err)
				var_PA_microshots[s_val].append(PA_microshot)
			
		if save:
			# Create a descriptive filename
			if len(scatter_ms) > 1:
				tau = f"{scatter_ms[0]:.2f}-{scatter_ms[-1]:.2f}"
			else:
				tau = f"{scatter_ms[0]:.2f}"
				
			out_file = (
				f"{out_dir}{frb_id}_mode_{mode}_sc_{tau}_"
				f"sgwidth_{width_range[0]:.2f}-{width_range[1]:.2f}_"
				f"gauss_{n_gauss}_seed_{seed}_nseed_{nseed}_PA{pol_angle[-1]:.2f}.pkl"
			)
			with open(out_file, 'wb') as frb_file:
				pkl.dump((scatter_ms, vals, errs, width[1], var_PA_microshots), frb_file)
			print(f"Saved FRB data to {out_file}")

		return vals, errs, width[1], var_PA_microshots
