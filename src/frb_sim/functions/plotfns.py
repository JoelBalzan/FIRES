#
#	Plotting functions
#
#								AB, August 2024
#
#	Function list
#

#	--------------------------	Import modules	---------------------------

import os
import sys

import matplotlib as mpl
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from scipy.optimize import curve_fit
from ..utils.utils import *
from .basicfns import *

mpl.rcParams['pdf.fonttype']	= 42
mpl.rcParams['ps.fonttype'] 	= 42
mpl.rcParams['savefig.dpi'] 	= 600
mpl.rcParams['font.family'] 	= 'sans-serif'
mpl.rcParams['font.size']		= 8

#	----------------------------------------------------------------------------------------------------------

def plot_stokes(fname, outdir, dspec4, iquvt, fmhzarr, tmsarr, xlim, save, figsize, show_plots):
	"""
	Plot Stokes IQUV profiles and dynamic spectra.
	Inputs:
		- fname, outdir: Directory to save the plot
		- dspec4: Dynamic spectrum array
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
	ax0.imshow(dspec4[0], aspect='auto', cmap='seismic', interpolation="none", vmin=-np.nanmax(np.abs(dspec4[0])), vmax=np.nanmax(np.abs(dspec4[0])), \
		extent=([np.amin(tmsarr), np.amax(tmsarr), (fmhzarr[-1] - chan_width_mhz) / 1.0e3, (fmhzarr[0] + chan_width_mhz) / 1.0e3]))
	ax0.text(0.95, 0.1, 'I', fontsize=10, fontweight='bold', transform=ax0.transAxes)
	ax0.set_xticklabels([])
	ax0.set_ylabel(r'$\nu$ (GHz)')
	ax0.yaxis.set_label_coords(-0.05, 0.5)
	
	ax1 = fig.add_axes([0.08, 0.38, 0.90, 0.16])
	ax1.tick_params(axis="both", direction="in", bottom=True, right=True, top=True, left=True)
	ax1.imshow(dspec4[1], aspect='auto', cmap='seismic', interpolation="none", vmin=-np.nanmax(np.abs(dspec4[1])), vmax=np.nanmax(np.abs(dspec4[1])), \
		extent=([np.amin(tmsarr), np.amax(tmsarr), (fmhzarr[-1] - chan_width_mhz) / 1.0e3, (fmhzarr[0] + chan_width_mhz) / 1.0e3]))
	ax1.text(0.95, 0.1, 'Q', fontsize=10, fontweight='bold', transform=ax1.transAxes)
	ax1.set_xticklabels([])
	ax1.set_ylabel(r'$\nu$ (GHz)')
	ax1.yaxis.set_label_coords(-0.05, 0.5)
	
	ax2 = fig.add_axes([0.08, 0.22, 0.90, 0.16])
	ax2.tick_params(axis="both", direction="in", bottom=True, right=True, top=True, left=True)
	ax2.imshow(dspec4[2], aspect='auto', cmap='seismic', interpolation="none", vmin=-np.nanmax(np.abs(dspec4[2])), vmax=np.nanmax(np.abs(dspec4[2])), \
		extent=([np.amin(tmsarr), np.amax(tmsarr), (fmhzarr[-1] - chan_width_mhz) / 1.0e3, (fmhzarr[0] + chan_width_mhz) / 1.0e3]))
	ax2.text(0.95, 0.1, 'U', fontsize=10, fontweight='bold', transform=ax2.transAxes)
	ax2.set_xticklabels([])
	ax2.set_ylabel(r'$\nu$ (GHz)')
	ax2.yaxis.set_label_coords(-0.05, 0.5)
	
	ax3 = fig.add_axes([0.08, 0.06, 0.90, 0.16])
	ax3.tick_params(axis="both", direction="in", bottom=True, right=True, top=True, left=True)
	ax3.imshow(dspec4[3], aspect='auto', cmap='seismic', interpolation="none", vmin=-np.nanmax(np.abs(dspec4[3])), vmax=np.nanmax(np.abs(dspec4[3])), \
		extent=([np.amin(tmsarr), np.amax(tmsarr), (fmhzarr[-1] - chan_width_mhz) / 1.0e3, (fmhzarr[0] + chan_width_mhz) / 1.0e3]))
	ax3.text(0.95, 0.1, 'V', fontsize=10, fontweight='bold', transform=ax3.transAxes)
	ax3.set_xlabel(r'Time (ms)')
	ax3.set_ylabel(r'$\nu$ (GHz)')
	ax3.yaxis.set_label_coords(-0.05, 0.5)
	
	if show_plots:
		plt.show()

	if save==True:
		fig.savefig(os.path.join(outdir, fname + "_iquv.pdf"), bbox_inches='tight', dpi=600)
		print("Saved figure to %s" % (os.path.join(outdir, fname + "_iquv.pdf")))


#	----------------------------------------------------------------------------------------------------------

def plot_dpa(fname, outdir, noistks, frbdat, tmsarr, ntp, save, figsize, show_plots):
	"""
	Plot PA profile and dPA/dt.
	Inputs:
		- fname, outdir: Directory to save the plot
		- noistks: Noise levels for each Stokes parameter
		- frbdat: FRB data object
		- tmsarr: Time array in ms
		- fsize: Figure size
		- ntp: Number of points for slope calculation
	"""
	print("Calculating dpa slope from %d points" % (2 * ntp + 1))

	phits = frbdat.phits
	dphits = frbdat.dphits

		
	dpadt = np.zeros(phits.shape, dtype=float)
	edpadt = np.zeros(phits.shape, dtype=float)	
	dpadt[:ntp] = np.nan
	edpadt[:ntp] = np.nan
	dpadt[-ntp:] = np.nan
	edpadt[-ntp:] = np.nan
	
	phits[frbdat.iquvt[0] < 10.0 * noistks[0]] = np.nan
	dphits[frbdat.iquvt[0] < 10.0 * noistks[0]] = np.nan
	
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
		
	print("Max (dPA/dt) = %.2f +/- %.2f deg/ms" % (dpadt[dpamax], edpadt[dpamax]))
		
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
		fig.savefig(os.path.join(outdir, fname + "_dpa.pdf"), bbox_inches='tight', dpi=600)
		print("Saved figure to %s" % (os.path.join(outdir, fname + "_dpa.pdf")))


#	----------------------------------------------------------------------------------------------------------

def plot_ilv_pa_ds(sc_dspec, freq_mhz, time_ms, save, fname, outdir, tsdata, noistks, figsize, scatter, show_plots):
	"""
		Plot I, L, V, dynamic spectrum and polarization angle.
		Inputs:
			- sc_dspec: Dynamic spectrum data
			- freq_mhz: Frequency array in MHz
			- time_ms: Time array in ms
			- rm: Rotation measure
			- save: Boolean indicating whether to save the plot
			- fname: Filename for saving the plot
			- outdir: Output directory for saving the plot
			- tsdata: Time series data object
			- noistks: Noise levels for each Stokes parameter
	"""
	
	tsdata.phits[tsdata.iquvt[0] < 10.0 * noistks[0]] = np.nan
	tsdata.dphits[tsdata.iquvt[0] < 10.0 * noistks[0]] = np.nan

	igood = np.where(tsdata.iquvt[0] > 10.0 * noistks[0])[0]
	
	lmax = np.argmax(tsdata.lfrac[igood])
	vmax = np.argmax(tsdata.vfrac[igood])
	pmax = np.argmax(tsdata.pfrac[igood])
		
	print("Max (L/I) = %.2f +/- %.2f" % (tsdata.lfrac[igood[lmax]], tsdata.elfrac[igood[lmax]]))
	print("Max (V/I) = %.2f +/- %.2f" % (tsdata.vfrac[igood[vmax]], tsdata.evfrac[igood[vmax]]))
	print("Max (P/I) = %.2f +/- %.2f" % (tsdata.pfrac[igood[pmax]], tsdata.epfrac[igood[pmax]]))


	# Linear polarisation
	L = np.sqrt(np.nanmean(sc_dspec[1,:], axis=0)**2 + np.nanmean(sc_dspec[2,:], axis=0)**2)

	
	fig, axs = plt.subplots(nrows=3, ncols=1, height_ratios=[0.5, 0.5, 1], figsize=(figsize[0], figsize[1]))
	fig.subplots_adjust(hspace=0.)


	# Plot polarisation angle
	#axs[0].errorbar(time_ms, tsdata.phits, tsdata.dphits, c='black', marker="*", markersize=1, lw=0.5, capsize=1, zorder=8)
	axs[0].plot(time_ms, tsdata.phits, c='black', lw=0.5, zorder=8)
	axs[0].fill_between(
		time_ms, 
		tsdata.phits - tsdata.dphits,  # Lower bound of the error
		tsdata.phits + tsdata.dphits,  # Upper bound of the error
		color='gray', 
		alpha=0.3,  # Transparency level
		label='Error'
	)
	axs[0].set_xlim(time_ms[0], time_ms[-1])
	axs[0].set_ylabel("PA [deg]")
	axs[0].set_xticklabels([])  # Hide x-tick labels for the first subplot
	axs[0].tick_params(axis='x', direction='in')  # Make x-ticks stick up
	
	# Plot the mean across all frequency channels (axis 0)
	axs[1].plot(time_ms, np.nanmean(sc_dspec[0,:], axis=0), markersize=2 ,label='I', color='Black')
	#axs[1].plot(time_ms, np.nanmean(np.sqrt(sc_dspec[1,:]**2 + sc_dspec[2,:]**2) + sc_dspec[3,:]**2, axis=0))
	axs[1].plot(time_ms, L, markersize=2, label='L', color='Red')
	#axs[1].plot(time_ms, np.nanmean(sc_dspec[1,:], axis=0), markersize=2, label='Q', color='Green')
	#axs[1].plot(time_ms, np.nanmean(sc_dspec[2,:], axis=0), markersize=2, label='U', color='Orange')
	axs[1].plot(time_ms, np.nanmean(sc_dspec[3,:], axis=0), markersize=2, label='V', color='Blue')
	axs[1].hlines(0, time_ms[0], time_ms[-1], color='Gray', lw=0.5)
	axs[1].yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
	
	axs[1].set_xlim(time_ms[0], time_ms[-1])
	axs[1].legend(loc='upper right')
	axs[1].set_ylabel("Flux Density (arb.)")
	axs[1].set_xticklabels([])  # Hide x-tick labels for the second subplot
	axs[1].tick_params(axis='x', direction='in')  # Make x-ticks stick up


	# Plot the 2D scattered dynamic spectrum
	## Calculate the mean and standard deviation of the dynamic spectrum
	mn = np.mean(sc_dspec[0,:], axis=(0, 1))
	std = np.std(sc_dspec[0,:], axis=(0, 1))
	## Set appropriate minimum and maximum values for imshow (Thanks to Dr. M. Lower)
	vmin = mn - 3*std
	vmax = mn + 7*std

	axs[2].imshow(sc_dspec[0], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
		vmin=vmin, vmax=vmax, extent=[time_ms[0], time_ms[-1], freq_mhz[0], freq_mhz[-1]])
	axs[2].set_xlabel("Time (ms)")
	axs[2].set_ylabel("Frequency (MHz)")
	axs[2].yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))


	if show_plots:
		plt.show()

	if save==True:
		fig.savefig(os.path.join(outdir, fname + f"_{scatter}" + "_dynspec.pdf"), bbox_inches='tight', dpi=600)
		print("Saved figure to %s" % (os.path.join(outdir, fname + f"_{scatter}" + "_dynspec.pdf")))


	#	----------------------------------------------------------------------------------------------------------

def plot_pa_rms_vs_scatter(scatter_timescales, pa_rms, dpa_rms, save, fname, outdir, figsize, show_plots):
	"""
	Plot the RMS of the polarization angle (PA) and its error bars vs the scattering timescale.
	
	Inputs:
		- scatter_timescales: Array of scattering timescales
		- pa_values: Array of polarization angle (PA) values
		- pa_errors: Array of errors associated with the PA values
		- save: Boolean indicating whether to save the plot
		- fname: Filename for saving the plot
		- outdir: Output directory for saving the plot
		- figsize: Tuple specifying the figure size (width, height)
	"""
	# Calculate RMS of PA and its error


	fig, ax = plt.subplots(figsize=figsize)
	print()

	# Plot the RMS of PA with error bars
	ax.errorbar(scatter_timescales, pa_rms, 
				yerr=dpa_rms, 
				fmt='o', capsize=3, color='blue', label='PA RMS')

	# Set plot labels and title
	ax.set_xlabel("Scattering Timescale (ms)")
	ax.set_ylabel("PA RMS (deg)")
	ax.set_title("RMS of Polarization Angle vs Scattering Timescale")
	ax.grid(True, linestyle='--', alpha=0.6)
	ax.legend()

	# Show the plot
	if show_plots:
		plt.show()

	# Save the plot if required
	if save:
		fig.savefig(os.path.join(outdir, fname + "_pa_rms_vs_scatter.pdf"), bbox_inches='tight', dpi=600)
		print(f"Saved figure to {os.path.join(outdir, fname + '_pa_rms_vs_scatter.pdf')}")