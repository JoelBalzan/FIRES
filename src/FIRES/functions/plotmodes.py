import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt

from FIRES.functions.basicfns import process_dynspec
from FIRES.functions.plotfns import plot_stokes, plot_ilv_pa_ds, plot_dpa, estimate_rm


class PlotMode:
	def __init__(self, name, process_func, plot_func):
		self.name = name
		self.process_func = process_func
		self.plot_func = plot_func
		
def basic_plots(fname, frb_data, mode, rm, out_dir, save, figsize, scatter_ms, show_plots):

    ds_data = frb_data

    ts_data, corr_dspec, noise_spec, noise_stokes = process_dynspec(
        ds_data.dynamic_spectrum, ds_data.freq_mhz, ds_data.time_ms, rm
    )

    iquvt = ts_data.iquvt
    time_ms = ds_data.time_ms
    freq_mhz = ds_data.freq_mhz

    if mode == "all":
        plot_ilv_pa_ds(corr_dspec, freq_mhz, time_ms, save, fname, out_dir, ts_data, figsize, scatter_ms, show_plots)
        plot_stokes(fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize, show_plots)
        plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots)
        estimate_rm(corr_dspec, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
    elif mode == "iquv":
        plot_stokes(fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize, show_plots)
    elif mode == "lvpa":
        plot_ilv_pa_ds(corr_dspec, freq_mhz, time_ms, save, fname, out_dir, ts_data, figsize, scatter_ms, show_plots)
    elif mode == "dpa":
        plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots)
    elif mode == "rm":
        estimate_rm(corr_dspec, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
    else:
        print(f"Invalid mode: {mode} \n")

# Define the iquv mode
iquv = PlotMode(
	name="iquv",
	process_func=None,  # No specific processing needed for iquv
	plot_func=basic_plots  # Use the general-purpose plot_stokes function
)

# Define the lvpa mode
lvpa = PlotMode(
	name="lvpa",
	process_func=None,  # No specific processing needed for lvpa
	plot_func=basic_plots  # Use the general-purpose plot_ilv_pa_ds function
)

# Define the dpa mode
dpa = PlotMode(
	name="dpa",
	process_func=None,  # No specific processing needed for dpa
	plot_func=basic_plots  # Use the general-purpose plot_dpa function
)
# Define the rm mode
rm = PlotMode(
	name="rm",
	process_func=None,  # No specific processing needed for rm
	plot_func=basic_plots  # Use the general-purpose estimate_rm function
)



# Processing function for pa_var
def process_pa_var(dspec, freq_mhz, time_ms, rm):
	ts_data, _, _, _ = process_dynspec(dspec, freq_mhz, time_ms, rm)
	peak_index = np.argmax(ts_data.iquvt[0])
	phits = ts_data.phits[peak_index:]
	dphits = ts_data.dphits[peak_index:]
	pa_var = np.nanvar(phits)
	pa_var_err = np.sqrt(np.nansum((phits * dphits)**2)) / (pa_var * len(phits))
	return pa_var, pa_var_err


def plot_pa_var(scatter_ms, vals, errs, save, fname, out_dir, figsize, show_plots, width_ms):
	"""
	Plot the var of the polarization angle (PA) and its error bars vs the scattering timescale.
	"""
	fig, ax = plt.subplots(figsize=figsize)

	# weight the scattering timescale by initial Gaussian width
	tau_weighted = scatter_ms / width_ms

	# Extract lower and upper errors relative to the median
	lower_errors = [median - lower for (lower, upper), median in zip(errs, vals)]
	upper_errors = [upper - median for (lower, upper), median in zip(errs, vals)]
	
	# Pass the errors as a tuple to yerr
	ax.errorbar(tau_weighted, vals, 
				yerr=(lower_errors, upper_errors), 
				fmt='o', capsize=1, color='black', label=r'\psi$_{var}$', markersize=2)
 
	ax.set_xlabel(r"$\tau_{ms} / \sigma_{ms}$")
	ax.set_ylabel(r"Var(\psi) / Var(\psi$_{microshots}$)")
	ax.grid(True, linestyle='--', alpha=0.6)

	if show_plots:
		plt.show()

	if save:
		fig.savefig(os.path.join(out_dir, fname + "_pa_var_vs_scatter.pdf"), bbox_inches='tight', dpi=600)
		print(f"Saved figure to {os.path.join(out_dir, fname + '_pa_var_vs_scatter.pdf')}  \n")

# Define the pa_var mode
pa_var = PlotMode(name="pa_var", 
					   process_func=process_pa_var, 
					   plot_func=plot_pa_var
)




def process_lfrac(dspec, freq_mhz, time_ms, rm):
	ts_data, _, _, _ = process_dynspec(dspec, freq_mhz, time_ms, rm)
	iquvt = ts_data.iquvt
	I = ts_data.iquvt[0]
	Q = ts_data.iquvt[1]
	U = ts_data.iquvt[2]
	V = ts_data.iquvt[3]
 
	threshold = 0.1 * np.nanmax(I)
	mask = I <= threshold
 
	itsub = np.where(mask, np.nan, iquvt[0])
	qtsub = np.where(mask, np.nan, iquvt[1])
	utsub = np.where(mask, np.nan, iquvt[2])
	vtsub = np.where(mask, np.nan, iquvt[3])
	
	L = np.sqrt(Q**2 + U**2)
 
	integrated_I = np.nansum(itsub)
	integrated_L = np.nansum(L)
	lfrac = integrated_L / integrated_I
 
	mask = I > threshold
	noise_I = np.nanstd(I[mask])
	noise_L = np.nanstd(L[mask])
	lfrac_err = np.sqrt((noise_L / integrated_I)**2 + (integrated_L * noise_I / integrated_I**2)**2)
	
 
	return lfrac, lfrac_err


def plot_lfrac_var(scatter_ms, vals, errs, save, fname, out_dir, figsize, show_plots, width_ms):
	fig, ax = plt.subplots(figsize=figsize)


	# Extract lower and upper errors relative to the median
	lower_errors = [median - lower for (lower, upper), median in zip(errs, vals)]
	upper_errors = [upper - median for (lower, upper), median in zip(errs, vals)]
	
	# Pass the errors as a tuple to yerr
	ax.errorbar(scatter_ms, vals, 
				yerr=(lower_errors, upper_errors), 
				fmt='o', capsize=1, color='black', label=r'\psi$_{var}$', markersize=2)
 
	ax.set_xlabel(r"L/I")
	ax.set_ylabel(r"Var(\psi) / Var(\psi$_{microshots}$)")
	ax.grid(True, linestyle='--', alpha=0.6)

	if show_plots:
		plt.show()

	if save:
		fig.savefig(os.path.join(out_dir, fname + "_lfrac_vs_scatter.pdf"), bbox_inches='tight', dpi=600)
		print(f"Saved figure to {os.path.join(out_dir, fname + '_lfrac_vs_scatter.pdf')}  \n")


lfrac = PlotMode(name = "lfrac", 
						  process_func = process_lfrac, 
						  plot_func = plot_lfrac_var
						  )

# Register all available plot modes
plot_modes = {
	"pa_var": pa_var,
	"iquv": iquv,
	"lvpa": lvpa,
	"dpa": dpa,
	"rm": rm,
	"lfrac": lfrac,
}