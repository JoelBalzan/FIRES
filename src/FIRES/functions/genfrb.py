# Simulating scattering
# AB, Sep 2024
# -------------------------- Import modules ---------------------------
from importlib.resources import files
import matplotlib as mpl
import numpy as np
from FIRES.functions.genfns import *
from FIRES.utils.utils import *
import os
import pickle as pkl  # Import pickle
from concurrent.futures import ProcessPoolExecutor
import functools
from itertools import product
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
obs_params_path = os.path.join(parent_dir, "utils/obsparams.txt")
gauss_params_path = os.path.join(parent_dir, "utils/gparams.txt")


def generate_dynspec(mode, s_val, plot_var, **params):
    """Generate dynamic spectrum based on mode."""
    s_val = s_val if plot_var else params["tau_ms"]
    
    # Remove 'tau_ms' from params to avoid conflict
    params = {k: v for k, v in params.items() if k != "tau_ms" and k != "nseed"}
    
    if mode == 'gauss':
        params = {k: v for k, v in params.items() if k != "num_micro_gauss" and k != "width_range"}
        return gauss_dynspec(**params, tau_ms=s_val)
    else:  # mode == 'mgauss'
        return sub_gauss_dynspec(**params, tau_ms=s_val)


def process_task(task, mode, plot, process_func, **params):
    """
    Process a single task (combination of timescale and realization).
    Dynamically uses the provided process_func for mode-specific processing.
    """
    s_val, realization = task
    current_seed = params["seed"] + realization if params["seed"] is not 1 else None
    params["seed"] = current_seed

    # Generate dynamic spectrum
    dspec, PA_microshot = generate_dynspec(
        mode=mode,
        s_val=s_val,
        plot_var=(plot == ['pa_var']),
        **params
    )

    # Use the provided process_func for mode-specific processing
    result, result_err = process_func(dspec, params["freq_mhz"], params["time_ms"], params["rm"])

    return s_val, result, result_err, PA_microshot


def generate_frb(scatter_ms, frb_id, out_dir, mode, n_gauss, seed, nseed, width_range, save,
                 obs_file, gauss_file, noise, scatter, plot, n_cpus, plot_mode):
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
    )

    if (lin_pol + circ_pol).any() > 1.0:
        print("WARNING: Linear and circular polarization fractions sum to more than 1.0.\n")

    if plot != ['pa_var'] and plot != ['lfrac']:
        dspec, _ = generate_dynspec(
            mode=mode,
            s_val=None,
            plot_var=False,
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
        # Create a list of tasks (timescale, realization)
        tasks = list(product(scatter_ms, range(nseed)))
        print(tasks)

        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            partial_func = functools.partial(
                process_task,
                mode=mode,
                plot=plot,
                process_func=plot_mode.process_func,
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
            

        # Compute averages and errors for each timescale
        if plot == ['pa_var']:
            vals /= var_PA_microshots
            errs /= var_PA_microshots
            #vals /= {s_val: np.array(vals[s_val]) for s_val in scatter_ms}
            #errs = {s_val: np.array(errs[s_val]) for s_val in scatter_ms}
        else:
            var_PA_microshots = None

        med_vals = []
        percentile_errs = []

        for s_val in scatter_ms:
            # Calculate the median of pa_var values
            median_val = np.median(vals[s_val])
            med_vals.append(median_val)

            # Calculate the 1-sigma percentiles (16th and 84th percentiles)
            lower_percentile = np.percentile(vals[s_val], 16)
            upper_percentile = np.percentile(vals[s_val], 84)

            # Error bars are the difference between the median and the percentiles
            percentile_errs.append((lower_percentile, upper_percentile))

        if save:
            # Create a descriptive filename
            out_file = (
                f"{out_dir}{frb_id}_mode_{mode}_sc_{scatter_ms[0]:.2f}_"
                f"sgwidth_{width_range[0]:.2f}-{width_range[1]:.2f}_"
                f"gauss_{n_gauss}_seed_{seed}_nseed_{nseed}_PA{pol_angle[-1]:.2f}.pkl"
            )
            with open(out_file, 'wb') as frb_file:
                pkl.dump((scatter_ms, med_vals, errs, width[1], var_PA_microshots), frb_file)
            print(f"Saved FRB data to {out_file}")

        return np.array(med_vals), np.array(errs), width[1], var_PA_microshots
