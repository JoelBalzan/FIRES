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
		figsize = (7, 7)
	
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
	if show_onpulse or show_offpulse:
		init_width = gdict["width"][0]/dspec_params.time_res_ms
		_, off_mask, (left, right) = on_off_pulse_masks_from_profile(I, init_width, frac=0.95, buffer_frac=buffer_frac)
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

