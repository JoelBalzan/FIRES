# -----------------------------------------------------------------------------
# plotfns.py
# FIRES: The Fast, Intense Radio Emission Simulator
#
# This module provides plotting functions for visualizing FRB simulation results,
# including Stokes parameter profiles, dynamic spectra, polarisation angle,
# and derived quantities as a function of simulation parameters.
#
# Author: JB
# Date: 2025-05-20
# -----------------------------------------------------------------------------

#	--------------------------	Import modules	---------------------------

import logging
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from fires.core.basicfns import (on_off_pulse_masks_from_profile,
                                 pa_variance_deg2, print_global_stats,
                                 wrap_pa_deg)
from fires.plotting.plot_helper import draw_plot_text
from fires.utils.utils import normalise_freq_window, normalise_phase_window

logger = logging.getLogger(__name__)
#	----------------------------------------------------------------------------------------------------------

def plot_stokes(fname, outdir, dspec, iquvt, fmhzarr, tmsarr, save, figsize, show_plots, extension):
	"""
	Plot Stokes IQUV profiles and dynamic spectra.
	Inputs:
		- fname, outdir: Directory to save the plot
		- dspec: Dynamic spectrum array
		- iquvt: IQUV time series array
		- fmhzarr: Frequency array in MHz
		- tmsarr: Time array in ms
		- xlim: X-axis limits for the plot
		- fsize: Figure size
	"""
	chan_width_mhz = np.abs(fmhzarr[0] - fmhzarr[1])  # Calculate channel width in MHz
	
	if figsize is None:
		figsize = (7, 9)
	Lts = np.sqrt(np.asarray(iquvt[1])**2 + np.asarray(iquvt[2])**2)

	# On-pulse mask from I
	on_mask, _, _ = on_off_pulse_masks_from_profile(iquvt[0], intrinsic_width_bins=1, frac=0.95, buffer_frac=None)
	I_int = np.nansum(np.where(on_mask, iquvt[0], 0.0))
	L_int = np.nansum(np.where(on_mask, Lts, 0.0))
	Lfrac = (L_int / I_int) if I_int > 0 else np.nan

	fig = plt.figure(figsize=(figsize[0], figsize[1]))
	ax = fig.add_axes([0.08, 0.70, 0.90, 0.28])
	ax.tick_params(axis="both", direction="in", bottom=True, right=True, top=True, left=True)

	ax.axhline(c='c', ls='--', lw=0.25)
	ax.plot(tmsarr, iquvt[0] / np.nanmax(iquvt[0]), 'k-', lw=0.5, label='I')
	ax.plot(tmsarr, iquvt[1] / np.nanmax(iquvt[0]), 'r-', lw=0.5, label='Q')
	ax.plot(tmsarr, iquvt[2] / np.nanmax(iquvt[0]), 'm-', lw=0.5, label='U')
	ax.plot(tmsarr, iquvt[3] / np.nanmax(iquvt[0]), 'b-', lw=0.5, label='V')
	# Also plot L
	ax.plot(tmsarr, Lts / np.nanmax(iquvt[0]), color='C2', lw=0.7, label='L')
	ax.set_ylim(ymax=1.1)
	ax.set_xlim([np.amin(tmsarr), np.amax(tmsarr)])
	ax.legend(loc='upper right', ncol=2)
	ax.set_ylabel(r'Normalised flux density')
	ax.set_xticklabels([])
	ax.set_title(f'Integrated L/I (95% boxcar) = {Lfrac:.3f}', fontsize=9)
	ax.yaxis.set_label_coords(-0.05, 0.5)
		
	ax0 = fig.add_axes([0.08, 0.54, 0.90, 0.16])
	ax0.tick_params(axis="both", direction="in", bottom=True, right=True, top=True, left=True)
	ax0.imshow(dspec[0], aspect='auto', cmap='seismic', interpolation="none", vmin=-np.nanmax(np.abs(dspec[0])), vmax=np.nanmax(np.abs(dspec[0])), \
		extent=([tmsarr[0], tmsarr[-1], fmhzarr[0] / 1.0e3, fmhzarr[1] / 1.0e3]))
	ax0.text(0.95, 0.1, 'I', fontsize=10, fontweight='bold', transform=ax0.transAxes)
	ax0.set_xticklabels([])
	ax0.set_ylabel(r'$\nu$ (GHz)')
	ax0.yaxis.set_label_coords(-0.05, 0.5)
	
	ax1 = fig.add_axes([0.08, 0.38, 0.90, 0.16])
	ax1.tick_params(axis="both", direction="in", bottom=True, right=True, top=True, left=True)
	ax1.imshow(dspec[1], aspect='auto', cmap='seismic', interpolation="none", vmin=-np.nanmax(np.abs(dspec[1])), vmax=np.nanmax(np.abs(dspec[1])), \
		extent=([tmsarr[0], tmsarr[-1], fmhzarr[0] / 1.0e3, fmhzarr[1] / 1.0e3]))
	ax1.text(0.95, 0.1, 'Q', fontsize=10, fontweight='bold', transform=ax1.transAxes)
	ax1.set_xticklabels([])
	ax1.set_ylabel(r'$\nu$ (GHz)')
	ax1.yaxis.set_label_coords(-0.05, 0.5)
	
	ax2 = fig.add_axes([0.08, 0.22, 0.90, 0.16])
	ax2.tick_params(axis="both", direction="in", bottom=True, right=True, top=True, left=True)
	ax2.imshow(dspec[2], aspect='auto', cmap='seismic', interpolation="none", vmin=-np.nanmax(np.abs(dspec[2])), vmax=np.nanmax(np.abs(dspec[2])), \
		extent=([tmsarr[0], tmsarr[-1], fmhzarr[0] / 1.0e3, fmhzarr[1] / 1.0e3]))
	ax2.text(0.95, 0.1, 'U', fontsize=10, fontweight='bold', transform=ax2.transAxes)
	ax2.set_xticklabels([])
	ax2.set_ylabel(r'$\nu$ (GHz)')
	ax2.yaxis.set_label_coords(-0.05, 0.5)
	
	ax3 = fig.add_axes([0.08, 0.06, 0.90, 0.16])
	ax3.tick_params(axis="both", direction="in", bottom=True, right=True, top=True, left=True)
	ax3.imshow(dspec[3], aspect='auto', cmap='seismic', interpolation="none", vmin=-np.nanmax(np.abs(dspec[3])), vmax=np.nanmax(np.abs(dspec[3])), \
		extent=([tmsarr[0], tmsarr[-1], fmhzarr[0] / 1.0e3, fmhzarr[1] / 1.0e3]))
	ax3.text(0.95, 0.1, 'V', fontsize=10, fontweight='bold', transform=ax3.transAxes)
	ax3.set_xlabel(r'Time (ms)')
	ax3.set_ylabel(r'$\nu$ (GHz)')
	ax3.yaxis.set_label_coords(-0.05, 0.5)
	
	if show_plots:
		plt.show()

	if save==True:
		fig.savefig(os.path.join(outdir, fname + "_iquv." + extension), bbox_inches='tight', dpi=600)
		logging.info("Saved figure to %s \n" % (os.path.join(outdir, fname + "_iquv." + extension)))


#	----------------------------------------------------------------------------------------------------------

def plot_dpa(fname, outdir, noise_stokes, frbdat, tmsarr, ntp, save, figsize, show_plots, extension):
	"""
	Plot PA profile and dPA/dt.
	Inputs:
		- fname, outdir: Directory to save the plot
		- noise_stokes: Noise levels for each Stokes parameter
		- frbdat: FRB data object
		- tmsarr: Time array in ms
		- fsize: Figure size
		- ntp: Number of points for slope calculation
	"""
	logging.info("Calculating dpa slope from %d points \n" % (2 * ntp + 1))

	phits = frbdat.phits
	ephits = frbdat.ephits
	iquvt = frbdat.iquvt

		
	dpadt = np.zeros(phits.shape, dtype=float)
	edpadt = np.zeros(phits.shape, dtype=float)	
	dpadt[:ntp] = np.nan
	edpadt[:ntp] = np.nan
	dpadt[-ntp:] = np.nan
	edpadt[-ntp:] = np.nan
	
	phits[iquvt[0] < 10.0 * noise_stokes[0]] = np.nan
	ephits[iquvt[0] < 10.0 * noise_stokes[0]] = np.nan
	
	for ti in range(ntp, len(phits) - ntp):
		phi3 = phits[ti - ntp:ti + ntp + 1]
		dphi3 = ephits[ti - ntp:ti + ntp + 1]
		tarr3 = tmsarr[ti - ntp:ti + ntp + 1]
		
		if np.count_nonzero(np.isfinite(phi3)) == (2 * ntp + 1):
			popt, pcov = np.polyfit(tarr3, phi3, deg=1, w=1.0 / dphi3, cov=True)
			perr = np.sqrt(np.diag(pcov))
			dpadt[ti] = popt[0]
			edpadt[ti] = perr[0]
		else:
			dpadt[ti] = np.nan
			edpadt[ti] = np.nan
	
	dpamax = np.nanargmax(dpadt)

	logging.info("Max (dPA/dt) = %.2f +/- %.2f deg/ms \n" % (dpadt[dpamax], edpadt[dpamax]))

	if figsize is None:
		figsize = (11, 8)
	fig = plt.figure(figsize=(figsize[0], figsize[1]))
	ax = fig.add_axes([0.15, 0.48, 0.83, 0.50])
	ax.tick_params(axis="both", direction="in", bottom=True, right=True, top=True, left=True)
	ax2 = ax.twinx()	
	ax2.axhline(c='c', ls='--', lw=0.25)
	ax2.plot(tmsarr, frbdat.iquvt[0] / np.nanmax(frbdat.iquvt[0]), 'c-', lw=0.5)
	ax2.set_xlim([np.amin(tmsarr), np.amax(tmsarr)])
	ax2.set_ylim([-0.1, 1.1])
	ax2.set_yticks([])
	
	ax.errorbar(tmsarr, phits, ephits, fmt='b*', markersize=5, lw=0.5, capsize=2)
	
	ax.set_xlim([np.amin(tmsarr), np.amax(tmsarr)])
	ax.set_xticklabels([])
	ax.set_ylabel(r'PA (deg)')
	ax.yaxis.set_label_coords(-0.12, 0.5)	
	
	ax1 = fig.add_axes([0.15, 0.10, 0.83, 0.38])
	ax1.tick_params(axis="both", direction="in", bottom=True, right=True, top=True, left=True)
	
	ax1.errorbar(tmsarr, dpadt, edpadt, fmt='ro', markersize=3, lw=0.5, capsize=2)
	
	ax1.set_xlim([np.amin(tmsarr), np.amax(tmsarr)])
	ax1.set_xlabel(r'Time (ms)')
	ax1.set_ylabel(r'Rate (deg / ms)')
	ax1.yaxis.set_label_coords(-0.12, 0.5)
	
	if show_plots:
		plt.show()

	if save==True:
		fig.savefig(os.path.join(outdir, fname + "_dpa." + extension), bbox_inches='tight', dpi=600)
		logging.info("Saved figure to %s \n" % (os.path.join(outdir, fname + "_dpa." + extension)))


#	----------------------------------------------------------------------------------------------------------

def plot_ilv_pa_ds(dspec, dspec_params, plot_config, freq_mhz, time_ms, save, fname, outdir, tsdata, figsize, tau, show_plots, extension, 
					legend, buffer_frac, show_onpulse, show_offpulse, segments=None, display_text=None):
	"""
		Plot I, L, V, dynamic spectrum and polarisation angle.
		Inputs:
			- dspec: Dynamic spectrum data
			- freq_mhz: Frequency array in MHz
			- time_ms: Time array in ms
			- save: Boolean indicating whether to save the plot
			- fname: Filename for saving the plot
			- outdir: Output directory for saving the plot
			- tsdata: Time series data object
			- noise_stokes: Noise levels for each Stokes parameter
	"""

	# Select the requested phase/freq windows (fall back to total/all)
	phase_win = getattr(dspec_params, "phase_window", None)
	freq_win  = getattr(dspec_params, "freq_window", None)
	phase_key = normalise_phase_window(phase_win, "segments") if phase_win else "total"
	freq_key  = normalise_freq_window(freq_win, "segments") if freq_win else "all"

	# Pull the values that will be used elsewhere (e.g. overlays/labels)
	vpsi = segments["phase"].get(phase_key, {}).get("Vpsi", np.nan)
	lfrac = segments["phase"].get(phase_key, {}).get("Lfrac", np.nan)
	vfrac = segments["phase"].get(phase_key, {}).get("Vfrac", np.nan)
	logging.info("Var(psi) [%s]: %.3f deg^2  | L/I=%.3f  V/I=%.3f", phase_key, vpsi, lfrac, vfrac)
	# Print global stats to log
	print_global_stats(segments["global"], logger=True)


	pa_rad = tsdata.phits
	pa_deg = np.rad2deg(pa_rad)
	finite_pa = pa_rad[np.isfinite(pa_rad)]
	
	if finite_pa.size == 0:
		logging.warning("All PA values are NaN or non-finite. Cannot compute variance.")
		pa_var_deg2 = np.nan
	else:
		pa_var_deg2 = pa_variance_deg2(finite_pa)
	
	pa_deg = np.rad2deg(pa_rad)

	phits = wrap_pa_deg(pa_deg)
	pa_err_rad = tsdata.ephits 
	finite_pa_err = pa_err_rad[np.isfinite(pa_err_rad)]
	if finite_pa_err.size == 0:
		epa_deg2 = np.nan
	else:
		epa_deg2 = pa_variance_deg2(finite_pa_err) 
	pa_err_deg = np.rad2deg(pa_err_rad)

	I, Q, U, V = tsdata.iquvt / 1e3  # Convert from Jy to kJy
	L = tsdata.Lts / 1e3  # Convert from Jy to kJy
	
	if figsize is None:
		figsize = (7, 9)
	
	fig, axs = plt.subplots(nrows=3, ncols=1, height_ratios=[0.5, 0.5, 1], figsize=figsize)
	fig.subplots_adjust(hspace=0.)

	# Plot polarisation angle
	axs[0].scatter(time_ms, phits, c='black', s=2.5, zorder=8)
	axs[0].errorbar(time_ms, pa_deg, yerr=pa_err_deg, fmt='none', ecolor='black', elinewidth=0.5, capsize=1, zorder=7)
 
	#axs[0].plot(time_ms, phits, c='black', lw=0.5, zorder=8)
	#axs[0].fill_between(time_ms, phits - dphits, phits + dphits, color='gray', alpha=0.3, label='Error')
	axs[0].set_xlim(time_ms[0], time_ms[-1])
	axs[0].set_ylim(-90, 90)
	axs[0].set_ylabel(r"$\psi$ [deg.]")
	axs[0].set_xticklabels([])  # Hide x-tick labels for the first subplot
	axs[0].yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
	axs[0].tick_params(axis='x', direction='in', length=3)  # Make x-ticks stick up
	draw_plot_text(axs[0], display_text, 'general', plot_config)
	
	# Plot the mean across all frequency channels (axis 0)
	axs[1].hlines(0, time_ms[0], time_ms[-1], color='Gray', lw=0.5)
	axs[1].plot(time_ms, I, markersize=1 ,label='I', color='Black')
	axs[1].plot(time_ms, L, markersize=1, label='L', color='Red')
	#axs[1].plot(time_ms, Q, markersize=2, label='Q', color='Green')
	#axs[1].plot(time_ms, U, markersize=2, label='U', color='Orange')
	axs[1].plot(time_ms, V, markersize=1, label='V', color='Blue')
	axs[1].yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))

	axs[1].set_xlim(time_ms[0], time_ms[-1])

	# Set fixed unit label with math italic S
	axs[1].set_ylabel(r"$S$ [arb.]")


	# Highlight on- and off-pulse regions if requested
	gdict = dspec_params.gdict
	inset_bounds = None
	peak_idx = None
	peak_time = None
	left_time = None
	init_width = gdict["width"][0] / dspec_params.time_res_ms
	_, off_mask, (left, right) = on_off_pulse_masks_from_profile(I, init_width, frac=0.95, buffer_frac=buffer_frac)
	if np.any(np.isfinite(I)):
		peak_idx = int(np.nanargmax(I))
		peak_time = time_ms[peak_idx]
		left_time = time_ms[left-1]
		if left <= peak_idx:
			inset_bounds = (left, peak_idx)
		else:
			logging.warning("Cannot build PA leading-edge inset: on-pulse left edge is after I peak.")
	else:
		logging.warning("Cannot build PA leading-edge inset: I profile is all non-finite.")

	if show_onpulse or show_offpulse:
		if show_onpulse:
			axs[1].axvspan(time_ms[left], time_ms[right], color='lightblue', alpha=0.35, zorder=0)
		if show_offpulse:
			axs[1].fill_between(
			time_ms, 0, 1, where=off_mask,
			color='lightcoral', alpha=0.15,
			transform=axs[1].get_xaxis_transform(), zorder=0, label='Off-pulse'
		)
	
	axs[1].tick_params(axis='x', direction='in', length=3)  
	axs[1].set_xticklabels([])  

	# Inset: PA profile over the leading edge (on-pulse first edge to I peak)
	if inset_bounds is not None:
		left_idx, peak_idx = inset_bounds
		if peak_idx > left_idx:
			inset_ax = axs[0].inset_axes([0.60, 0.11, 0.39, 0.33])
			t_slice = slice(left_idx, peak_idx + 1)
			t_inset = time_ms[t_slice]
			pa_inset = phits[t_slice]
			epa_inset = pa_err_deg[t_slice]
			inset_ax.scatter(t_inset, pa_inset, c='black', s=7, zorder=3)
			inset_ax.errorbar(
				t_inset,
				pa_inset,
				yerr=epa_inset,
				fmt='none',
				ecolor='black',
				elinewidth=0.5,
				capsize=1,
				zorder=2,
			)
			inset_ax.set_xlim(left_time, peak_time)
			# Keep inset y-range tight around finite PA points in this interval.
			finite_pa_inset = pa_inset[np.isfinite(pa_inset)]
			if finite_pa_inset.size > 1:
				ypad = max(3.0, 0.1 * (np.nanmax(finite_pa_inset) - np.nanmin(finite_pa_inset)))
				inset_ax.set_ylim(np.nanmin(finite_pa_inset) - ypad, np.nanmax(finite_pa_inset) + ypad)
			else:
				inset_ax.set_ylim(-90, 90)
			inset_ax.tick_params(axis='both', labelsize=12, direction='in', length=2)
			inset_ax.tick_params(axis='y', pad=1)
			inset_ax.tick_params(axis='x', pad=1)
			#for label in inset_ax.get_xticklabels():
			#	label.set_horizontalalignment('right')
			for spine in inset_ax.spines.values():
				spine.set_linewidth(0.7)

			# Mark the inset region on the parent PA panel.
			axs[0].axvspan(left_time, peak_time, color='gray', alpha=0.12, zorder=1)
			axs[0].axvline(left_time, color='gray', ls='--', lw=0.6, zorder=1)
			axs[0].axvline(peak_time, color='gray', ls='--', lw=0.6, zorder=1)

	#mn = np.mean(dspec[0])
	#std = np.std(dspec[0])
	#vmin = mn - 3*std
	#vmax = mn + 7*std

	vmin = np.nanpercentile(dspec[0], 1)
	vmax = np.nanpercentile(dspec[0], 99)
	axs[2].imshow(dspec[0], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
		vmin=vmin, vmax=vmax, 
  		extent=[time_ms[0], time_ms[-1], freq_mhz[0], freq_mhz[-1]])
	axs[2].set_xlabel("Time [ms]")
	axs[2].set_ylabel("Freq. [MHz]")
	axs[2].yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

	if legend:
		axs[1].legend(loc='upper right')

	if show_plots:
		plt.show()

	if save==True:
		fig.savefig(os.path.join(outdir, fname + f"_t_{tau[0]}" + "_ILVPA." + extension), bbox_inches='tight', dpi=600)
		logging.info("Saved figure to %s \n" % (os.path.join(outdir, fname + f"_t_{tau[0]}" + "_ILVPA." + extension)))


	#	----------------------------------------------------------------------------------------------------------


def plot_pa_profile(fname, outdir, tsdata, time_ms, save, figsize, show_plots, extension, xlim=None, ylim=None):
	"""
	Plot only the PA profile vs. time (deg), similar to the PA panel in lvpa.

	Inputs:
		- fname, outdir: output naming and directory
		- tsdata: time series data object (expects .phits [rad], .ephits [rad])
		- time_ms: time array in ms
		- save: save figure if True
		- figsize: figure size tuple or None (defaults to (7, 3))
		- show_plots: show figure interactively if True
		- extension: output file extension (e.g., 'pdf', 'png')
	"""
	pa_rad = tsdata.phits
	pa_deg = np.rad2deg(pa_rad)

	phits = wrap_pa_deg(pa_deg)
	pa_err_rad = tsdata.ephits
	pa_err_deg = np.rad2deg(pa_err_rad)

	# Log variance info (deg^2) if finite
	finite_pa = pa_rad[np.isfinite(pa_rad)]
	finite_pa_err = pa_err_rad[np.isfinite(pa_err_rad)]
	if finite_pa.size == 0:
		pa_var_deg2 = np.nan
	else:
		pa_var_deg2 = pa_variance_deg2(finite_pa)
	epa_deg2 = pa_variance_deg2(finite_pa_err) if finite_pa_err.size > 0 else np.nan
	logging.info("Var(psi) = %.3f +/- %.3f deg^2", pa_var_deg2, epa_deg2)

	if figsize is None:
		figsize = (7, 3)

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
	fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.95)

	# Scatter points and error bars (mirror lvpa style)
	ax.scatter(time_ms, phits, c='black', s=6, zorder=3)
	ax.errorbar(time_ms, pa_deg, yerr=pa_err_deg, fmt='none',
				ecolor='black', elinewidth=0.6, capsize=1, zorder=2)

	if xlim is not None:
		ax.set_xlim(xlim)
	else:
		ax.set_xlim(time_ms[0], time_ms[-1])
	if ylim is not None:
		ax.set_ylim(ylim)
	else:
		ax.set_ylim(-90, 90)
	ax.set_xlabel("Time [ms]")
	ax.set_ylabel(r"$\psi$ [deg.]")
	ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
	ax.tick_params(axis='x', direction='in', length=3)
	ax.tick_params(axis='y', direction='in', length=3)

	if show_plots:
		plt.show()

	if save:
		fpath = os.path.join(outdir, f"{fname}_pa.{extension}")
		fig.savefig(fpath, bbox_inches='tight', dpi=600)
		logging.info("Saved figure to %s \n", fpath)


def plot_pa_li_scatter(
	fname,
	outdir,
	tsdata,
	time_ms,
	noise_stokes=None,
	save=True,
	figsize=None,
	show_plots=False,
	extension="png",
	snr_cut=2.0,
	use_onpulse=True,
):
	"""
	Scatter plot of:
		x = PA - <PA>
		y = (L/I) - <L/I>

	Inputs:
		- tsdata: object with .iquvt, .phits (rad), .ephits (rad), .Lts
		- time_ms: time array
		- noise_stokes: optional noise levels [I, Q, U, V]
		- snr_cut: mask points below S/N threshold (in I)
		- use_onpulse: restrict to on-pulse region
	"""

	from scipy.stats import pearsonr, rankdata, spearmanr

	# --- Extract data ---
	I, Q, U, V = tsdata.iquvt
	L = tsdata.Lts

	# --- Compute PA (deg, wrapped) ---
	pa_deg = np.rad2deg(tsdata.phits)
	pa_wrapped = wrap_pa_deg(pa_deg)

	# --- Compute L/I safely ---
	with np.errstate(divide='ignore', invalid='ignore'):
		li = L / I
	li[~np.isfinite(li)] = np.nan

	# --- Masking ---
	mask = np.isfinite(pa_wrapped) & np.isfinite(li)

	# S/N cut
	if noise_stokes is not None:
		mask &= I > snr_cut * noise_stokes[0]

	# On-pulse mask
	if use_onpulse:
		on_mask, _, _ = on_off_pulse_masks_from_profile(
			I, intrinsic_width_bins=1, frac=0.95, buffer_frac=None
		)
		mask &= on_mask

	if np.count_nonzero(mask) < 10:
		logger.warning("Not enough valid points for PA–L/I scatter.")
		return

	# --- Apply mask ---
	pa = pa_wrapped[mask]
	li = li[mask]

	# --- Compute mean PA properly (circular mean!) ---
	pa_rad = np.deg2rad(pa)
	mean_pa_rad = 0.5 * np.angle(np.nanmean(np.exp(1j * 2 * pa_rad)))

	# --- Compute ΔPA correctly (circular residual) ---
	delta_pa = 0.5 * np.rad2deg(
		np.angle(np.exp(1j * 2 * (pa_rad - mean_pa_rad)))
	)

	# --- Compute Δ(L/I) ---
	mean_li = np.nanmean(li)
	delta_li = li - mean_li

	# Pulse-profile peak (max I) projected into this ΔPA–Δ(L/I) space
	peak_idx = None
	peak_delta_pa = np.nan
	peak_delta_li = np.nan
	if np.any(np.isfinite(I)):
		peak_idx = int(np.nanargmax(I))
		peak_pa = pa_wrapped[peak_idx]
		peak_li = (L / I)[peak_idx] if np.isfinite(I[peak_idx]) and I[peak_idx] != 0 else np.nan
		if np.isfinite(peak_pa) and np.isfinite(peak_li):
			peak_pa_rad = np.deg2rad(peak_pa)
			peak_delta_pa = 0.5 * np.rad2deg(
				np.angle(np.exp(1j * 2 * (peak_pa_rad - mean_pa_rad)))
			)
			peak_delta_li = peak_li - mean_li

	# --- Correlations and test stack ---
	def _safe_spearman_stat(xvals, yvals):
		"""Return Spearman rho robustly across SciPy versions."""
		try:
			res = spearmanr(xvals, yvals)
			if hasattr(res, "statistic"):
				return float(res.statistic), float(res.pvalue)
			return float(res[0]), float(res[1])
		except Exception:
			return np.nan, np.nan

	def _partial_spearman_time(xvals, yvals, tvals):
		"""Partial Spearman between x and y after linear control for time ranks."""
		xr = rankdata(xvals)
		yr = rankdata(yvals)
		tr = rankdata(tvals)

		A = np.column_stack((np.ones_like(tr), tr))
		rx = xr - A @ np.linalg.lstsq(A, xr, rcond=None)[0]
		ry = yr - A @ np.linalg.lstsq(A, yr, rcond=None)[0]
		return pearsonr(rx, ry)

	try:
		pear_r, pear_p = pearsonr(delta_pa, delta_li)
	except Exception:
		pear_r, pear_p = np.nan, np.nan

	spear_r, spear_p = _safe_spearman_stat(delta_pa, delta_li)

	# Circular-shift permutation test for Spearman rho (preserves 1D time structure).
	# Use all unique non-zero circular lags exactly once: 1..(N-1).
	npts = delta_pa.size
	rng = np.random.default_rng(42)
	if npts > 2 and np.isfinite(spear_r):
		lags = np.arange(1, npts, dtype=int)
		n_perm = int(lags.size)
		perm_rhos = np.full(n_perm, np.nan, dtype=float)
		for ii, lag in enumerate(lags):
			y_shift = np.roll(delta_li, lag)
			perm_rhos[ii], _ = _safe_spearman_stat(delta_pa, y_shift)
		valid_perm = np.isfinite(perm_rhos)
		n_valid_perm = int(np.count_nonzero(valid_perm))
		if n_valid_perm > 0:
			perm_p = (1.0 + np.count_nonzero(np.abs(perm_rhos[valid_perm]) >= np.abs(spear_r))) / (n_valid_perm + 1.0)
		else:
			perm_p = np.nan
	else:
		n_perm = 0
		n_valid_perm = 0
		perm_p = np.nan

	# Bootstrap CI for Spearman rho
	n_boot = 10000
	boot_rhos = np.full(n_boot, np.nan, dtype=float)
	if npts > 2:
		for ii in range(n_boot):
			idx = rng.integers(0, npts, size=npts)
			boot_rhos[ii], _ = _safe_spearman_stat(delta_pa[idx], delta_li[idx])
		valid_boot = boot_rhos[np.isfinite(boot_rhos)]
		if valid_boot.size > 0:
			spear_ci_lo, spear_ci_hi = np.percentile(valid_boot, [2.5, 97.5])
		else:
			spear_ci_lo, spear_ci_hi = np.nan, np.nan
	else:
		spear_ci_lo, spear_ci_hi = np.nan, np.nan

	# Partial Spearman controlling for time
	t_plot = time_ms[mask]
	try:
		partial_r, partial_p = _partial_spearman_time(delta_pa, delta_li, t_plot)
	except Exception:
		partial_r, partial_p = np.nan, np.nan

	logger.info(
		"PA-L/I correlation: Pearson r=%.3f (p=%.2e), Spearman r=%.3f (p=%.2e)",
		pear_r, pear_p, spear_r, spear_p
	)
	logger.info(
		"PA-L/I test stack: Spearman perm p=%.2e (n_perm=%d, valid=%d), Spearman 95%% CI=[%.3f, %.3f], partial Spearman|time r=%.3f (p=%.2e)",
		perm_p, n_perm, n_valid_perm, spear_ci_lo, spear_ci_hi, partial_r, partial_p
	)

	# --- Plot ---
	if figsize is None:
		figsize = (8, 8)

	# Use time as colour (only masked points)

	# 2D PA-L/I scatter (existing output)
	fig, ax = plt.subplots(figsize=figsize)
	sc = ax.scatter(
		delta_pa,
		delta_li,
		c=t_plot,
		s=8,
		alpha=0.7
	)

	if np.isfinite(peak_delta_pa) and np.isfinite(peak_delta_li):
		ax.scatter(
			peak_delta_pa,
			peak_delta_li,
			marker='*',
			s=150,
			color='pink',
			edgecolor='red',
			linewidth=1.0,
			zorder=5,
			label='I peak'
		)
		#ax.legend(loc='best', fontsize=14, frameon=False)

	cbar = plt.colorbar(sc, ax=ax, orientation='vertical')
	cbar.set_label("Time [ms]")

	ax.axhline(0, lw=0.5)
	ax.axvline(0, lw=0.5)
	ax.set_xlabel(r'$\Delta \psi$ [deg]')
	ax.set_ylabel(r'$\Delta \Pi_L$')
	#ax.set_xlim(-25,25)
	#ax.set_ylim(-0.6,0.6)
	#ax.set_title(f"Pearson r={pear_r:.2f}, Spearman r={spear_r:.2f}",fontsize=9)
	#stats_text = (
	#	f"Spearman p={spear_p:.2e}\\n"
	#	f"Perm p={perm_p:.2e} (n={n_valid_perm})\\n"
	#	f"Spearman 95% CI=[{spear_ci_lo:.2f}, {spear_ci_hi:.2f}]\\n"
	#	f"Partial Spearman|t={partial_r:.2f} (p={partial_p:.2e})"
	#)
	#ax.text(
	#	0.03, 0.97, stats_text,
	#	transform=ax.transAxes,
	#	va='top', ha='left',
	#	fontsize=8,
	#	bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='0.6')
	#)
	ax.tick_params(direction="in", top=True, right=True)

	if save:
		fpath = os.path.join(outdir, f"{fname}_pa_li_scatter.{extension}")
		fig.savefig(fpath, bbox_inches='tight', dpi=600)
		logger.info("Saved figure to %s \n", fpath)

	# Additional 3D PA-L/I-time scatter with time as an explicit axis
	fig3d = plt.figure(figsize=figsize)
	ax3d = fig3d.add_subplot(111, projection='3d')

	sc3d = ax3d.scatter(
		t_plot,
		delta_pa,
		delta_li,
		c=t_plot,
		s=8,
		alpha=0.75
	)

	if np.isfinite(peak_delta_pa) and np.isfinite(peak_delta_li):
		peak_time = time_ms[peak_idx] if peak_idx is not None else np.nan
		if np.isfinite(peak_time):
			ax3d.scatter(
				peak_time,
				peak_delta_pa,
				peak_delta_li,
				marker='*',
				s=150,
				color='pink',
				edgecolor='red',
				linewidth=1.0,
				zorder=200,
				label='I peak'
			)
			#ax3d.legend(loc='best', fontsize=14, frameon=False)

	#cbar3d = plt.colorbar(sc3d, ax=ax3d, pad=0.08, orientation='horizontal')
	#cbar3d.set_label("Time [ms]")

	ax3d.set_xlabel("Time [ms]", labelpad=15)
	ax3d.set_ylabel(r'$\Delta \psi$ [deg]', labelpad=10)
	ax3d.set_zlabel(r'$\Delta \Pi_L$', labelpad=15)
	ax3d.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
	ax3d.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
	ax3d.zaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
	#ax3d.set_title(f"PA-L/I-Time (Pearson r={pear_r:.2f}, Spearman r={spear_r:.2f})",fontsize=9)
	# Emphasise time axis and use a view where both PA and L/I axes are visible.
	try:
		ax3d.set_box_aspect((1.45, 1.0, 1.0))
	except Exception:
		pass
	ax3d.view_init(elev=24, azim=-58)
	ax3d.tick_params(direction="in")

	if save:
		fpath3d = os.path.join(outdir, f"{fname}_pa_li_scatter_3d.{extension}")
		# For mplot3d, tight bbox can clip z-axis labels; use explicit margins instead.
		fig3d.subplots_adjust(left=0.06, right=0.88, bottom=0.08, top=0.97)
		fig3d.savefig(fpath3d, dpi=600)
		logger.info("Saved figure to %s \n", fpath3d)

	if show_plots:
		plt.show()