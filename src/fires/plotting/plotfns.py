# -----------------------------------------------------------------------------
# plotfns.py
# FIRES: The Fast, Intense Radio Emission Simulator
#
# This module provides plotting functions for visualizing FRB simulation results,
# including Stokes parameter profiles, dynamic spectra, polarization angle,
# and derived quantities as a function of simulation parameters.
#
# Author: JB
# Date: 2025-05-20
# -----------------------------------------------------------------------------

#	--------------------------	Import modules	---------------------------

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ..utils.utils import *
from ..core.basicfns import *

import logging
logging.basicConfig(level=logging.INFO)
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
	
	fig = plt.figure(figsize=(figsize[0], figsize[1]))
	ax = fig.add_axes([0.08, 0.70, 0.90, 0.28])
	ax.tick_params(axis="both", direction="in", bottom=True, right=True, top=True, left=True)
	
	ax.axhline(c='c', ls='--', lw=0.25)
	ax.plot(tmsarr, iquvt[0] / np.nanmax(iquvt[0]), 'k-', lw=0.5, label='I')
	ax.plot(tmsarr, iquvt[1] / np.nanmax(iquvt[0]), 'r-', lw=0.5, label='Q')
	ax.plot(tmsarr, iquvt[2] / np.nanmax(iquvt[0]), 'm-', lw=0.5, label='U')
	ax.plot(tmsarr, iquvt[3] / np.nanmax(iquvt[0]), 'b-', lw=0.5, label='V')
	ax.set_ylim(ymax=1.1)
	ax.set_xlim([np.amin(tmsarr), np.amax(tmsarr)])
	ax.legend(loc='upper right', ncol=2)
	ax.set_ylabel(r'Normalized flux density')
	ax.set_xticklabels([])
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
	dphits = frbdat.dphits
	iquvt = frbdat.iquvt

		
	dpadt = np.zeros(phits.shape, dtype=float)
	edpadt = np.zeros(phits.shape, dtype=float)	
	dpadt[:ntp] = np.nan
	edpadt[:ntp] = np.nan
	dpadt[-ntp:] = np.nan
	edpadt[-ntp:] = np.nan
	
	phits[iquvt[0] < 10.0 * noise_stokes[0]] = np.nan
	dphits[iquvt[0] < 10.0 * noise_stokes[0]] = np.nan
	
	for ti in range(ntp, len(phits) - ntp):
		phi3 = phits[ti - ntp:ti + ntp + 1]
		dphi3 = dphits[ti - ntp:ti + ntp + 1]
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

	fig = plt.figure(figsize=(figsize[0], figsize[1]))
	ax = fig.add_axes([0.15, 0.48, 0.83, 0.50])
	ax.tick_params(axis="both", direction="in", bottom=True, right=True, top=True, left=True)
	ax2 = ax.twinx()	
	ax2.axhline(c='c', ls='--', lw=0.25)
	ax2.plot(tmsarr, frbdat.iquvt[0] / np.nanmax(frbdat.iquvt[0]), 'c-', lw=0.5)
	ax2.set_xlim([np.amin(tmsarr), np.amax(tmsarr)])
	ax2.set_ylim([-0.1, 1.1])
	ax2.set_yticks([])
	
	ax.errorbar(tmsarr, phits, dphits, fmt='b*', markersize=5, lw=0.5, capsize=2)
	
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

def plot_ilv_pa_ds(dspec, freq_mhz, time_ms, save, fname, outdir, tsdata, figsize, tau_ms, show_plots, snr, extension, 
					legend, info, buffer_frac, show_onpulse, show_offpulse):
	"""
		Plot I, L, V, dynamic spectrum and polarization angle.
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

	# Wrap PA to [-90, 90] range
	phits = ((np.rad2deg(tsdata.phits) + 90) % 180) - 90
	ephits = np.rad2deg(tsdata.ephits)
	logging.info("Var(psi) = %.3f +/- %.3f" % (np.nanvar(phits), np.nanvar(ephits)))

	# Linear polarisation
	I, Q, U, V = tsdata.iquvt / 1e3  # Convert from Jy to kJy
	L = np.sqrt(Q**2 + U**2)
	
	if figsize is None:
		figsize = (7, 9)
	
	fig, axs = plt.subplots(nrows=3, ncols=1, height_ratios=[0.5, 0.5, 1], figsize=figsize)
	fig.subplots_adjust(hspace=0.)

	# Plot polarisation angle
	axs[0].scatter(time_ms, phits, c='black', s=1, zorder=8)
	axs[0].errorbar(time_ms, phits, yerr=ephits, fmt='none', ecolor='black', elinewidth=0.5, capsize=1, zorder=7)
 
	#axs[0].plot(time_ms, phits, c='black', lw=0.5, zorder=8)
	#axs[0].fill_between(time_ms, phits - dphits, phits + dphits, color='gray', alpha=0.3, label='Error')
	axs[0].set_xlim(time_ms[0], time_ms[-1])
	axs[0].set_ylim(-90, 90)
	axs[0].set_ylabel(r"$\psi$ [deg.]")
	axs[0].set_xticklabels([])  # Hide x-tick labels for the first subplot
	axs[0].yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
	axs[0].tick_params(axis='x', direction='in', length=3)  # Make x-ticks stick up
	
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
	axs[1].set_ylabel(r"$S$ [kJy]")


	# Highlight on- and off-pulse regions if requested
	if show_onpulse or show_offpulse:
		_, off_mask, (left, right) = on_off_pulse_masks_from_profile(I, frac=0.95, buffer_frac=buffer_frac, one_sided_offpulse=True)
		if show_onpulse:
			# Shade on-pulse region
			axs[1].axvspan(time_ms[left], time_ms[right], color='lightblue', alpha=0.35, zorder=0)
		if show_offpulse:
			# Shade off-pulse regions using a normalized-y transform (fills full height)
			axs[1].fill_between(
			time_ms, 0, 1, where=off_mask,
			color='lightcoral', alpha=0.15,
			transform=axs[1].get_xaxis_transform(), zorder=0, label='Off-pulse'
		)
	
	axs[1].tick_params(axis='x', direction='in', length=3)  # Make x-ticks stick up
	if snr is not None:
		axs_1_text = r"$\,\tau_0 = %.2f\,\mathrm{ms}\\\mathrm{S/N} = %.2f$" % (tau_ms[0], snr)
	else:
		axs_1_text = r"$\,\tau_0 = %.2f\,\mathrm{ms}$" % (tau_ms[0])


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
	axs[2].yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

	if legend:
		axs[1].legend(loc='upper right')

	if info:
		axs[1].text(
		0.82, 0.90,  # x, y in axes fraction coordinates 
		axs_1_text,
		ha='right', va='top',
		transform=axs[1].transAxes,
		color='black',
		bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
		)

	if show_plots:
		plt.show()

	if save==True:
		fig.savefig(os.path.join(outdir, fname + f"_t_{tau_ms[0]}" + "_ILVPA." + extension), bbox_inches='tight', dpi=600)
		logging.info("Saved figure to %s \n" % (os.path.join(outdir, fname + f"_t_{tau_ms[0]}" + "_ILVPA." + extension)))


	#	----------------------------------------------------------------------------------------------------------
