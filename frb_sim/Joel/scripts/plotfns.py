#
#	Plotting functions
#
#								AB, August 2024
#
#	Function list
#

#	--------------------------	Import modules	---------------------------

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mpc
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from utils import *

mpl.rcParams['pdf.fonttype']	= 42
mpl.rcParams['ps.fonttype'] 	= 42
mpl.rcParams['savefig.dpi'] 	= 600
mpl.rcParams['font.family'] 	= 'sans-serif'
mpl.rcParams['font.size']		= 8

#	----------------------------------------------------------------------------------------------------------

def plot_stokes(plotdir, dspec4, iquvt, fmhzarr, tmsarr, xlim, fsize):
    """
    Plot Stokes IQUV profiles and dynamic spectra.
    Inputs:
        - plotdir: Directory to save the plot
        - dspec4: Dynamic spectrum array
        - iquvt: IQUV time series array
        - fmhzarr: Frequency array in MHz
        - tmsarr: Time array in ms
        - xlim: X-axis limits for the plot
        - fsize: Figure size
    """
    chan_width_mhz = np.abs(fmhzarr[0] - fmhzarr[1])  # Calculate channel width in MHz
    
    fig = plt.figure(figsize=(fsize[0], fsize[1]))
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
    
    plt.show()

    return 0

#	----------------------------------------------------------------------------------------------------------

def plot_ilv_pa_ds(plotdir, noistks, dspec4, frbdat, fmhzarr, tmsarr, xlim, fsize):
    """
    Plot ILV and PA profiles, and I dynamic spectrum.
    Inputs:
        - plotdir: Directory to save the plot
        - noistks: Noise levels for each Stokes parameter
        - dspec4: Dynamic spectrum array
        - frbdat: FRB data object
        - fmhzarr: Frequency array in MHz
        - tmsarr: Time array in ms
        - xlim: X-axis limits for the plot
        - fsize: Figure size
    """
    igood = np.where(frbdat.iquvt[0] > 10.0 * noistks[0])[0]
    
    lmax = np.argmax(frbdat.lfrac[igood])
    vmax = np.argmax(frbdat.vfrac[igood])
    pmax = np.argmax(frbdat.pfrac[igood])
        
    print("Max (L/I) = %.2f +/- %.2f" % (frbdat.lfrac[igood[lmax]], frbdat.elfrac[igood[lmax]]))
    print("Max (V/I) = %.2f +/- %.2f" % (frbdat.vfrac[igood[vmax]], frbdat.evfrac[igood[vmax]]))
    print("Max (P/I) = %.2f +/- %.2f" % (frbdat.pfrac[igood[pmax]], frbdat.epfrac[igood[pmax]]))
    
    chan_width_mhz = np.abs(fmhzarr[0] - fmhzarr[1])
        
    fig = plt.figure(figsize=(fsize[0], fsize[1]))
    ax = fig.add_axes([0.14, 0.66, 0.85, 0.33])
    ax.tick_params(axis="both", direction="in", bottom=True, right=False, top=False, left=True)
    
    ax.axhline(c='c', ls='--', lw=0.5)
    ax.plot(tmsarr, frbdat.iquvt[0] / np.nanmax(frbdat.iquvt[0]), 'k-', label='I', lw=1.0, zorder=8)
    ax.plot(tmsarr, frbdat.lts / np.nanmax(frbdat.iquvt[0]), 'r-', label='L', lw=0.5)
    ax.plot(tmsarr, frbdat.iquvt[3] / np.nanmax(frbdat.iquvt[0]), 'b-', label='V', lw=0.5)
    ax.set_ylim(ymin=-0.3)
    ax.set_ylim(ymax=1.1)
    ax.set_xlim([np.amin(tmsarr), np.amax(tmsarr)])
    ax.legend(loc='upper right', ncol=1)
    ax.set_ylabel(r'Normalized flux density')
    ax.set_xticklabels([])
    ax.yaxis.set_label_coords(-0.1, 0.5)
    
    ax0 = fig.add_axes([0.14, 0.33, 0.85, 0.33])
    ax0.tick_params(axis="both", direction="in", bottom=True, right=False, top=False, left=True)
    ax0.imshow(dspec4[0], aspect='auto', cmap='seismic', interpolation="none", vmin=-np.nanmax(np.abs(dspec4[0])), vmax=np.nanmax(np.abs(dspec4[0])), \
        extent=([np.amin(tmsarr), np.amax(tmsarr), (fmhzarr[-1] - chan_width_mhz) / 1.0e3, (fmhzarr[0] + chan_width_mhz) / 1.0e3]))
    ax0.set_xticklabels([])
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax0.set_ylabel(r'$\nu$ (GHz)')
    ax0.yaxis.set_label_coords(-0.1, 0.5)
        
    ax1 = fig.add_axes([0.14, 0.08, 0.85, 0.25])
    ax1.tick_params(axis="both", direction="in", bottom=True, right=False, top=False, left=True)
    ax1.errorbar(tmsarr, frbdat.phits, frbdat.dphits, fmt='ro', markersize=2, lw=0.5, capsize=1, zorder=8)
    ax1.set_xlim([np.amin(tmsarr), np.amax(tmsarr)])
    ax1.set_ylabel(r'PA (deg)')
    ax1.set_xlabel(r'Time (ms)')
    ax1.yaxis.set_label_coords(-0.1, 0.5)
    
    plt.show()

    return 0

#	----------------------------------------------------------------------------------------------------------

def plot_dpa(plotdir, noistks, frbdat, tmsarr, fsize, ntp):
    """
    Plot PA profile and dPA/dt.
    Inputs:
        - plotdir: Directory to save the plot
        - noistks: Noise levels for each Stokes parameter
        - frbdat: FRB data object
        - tmsarr: Time array in ms
        - fsize: Figure size
        - ntp: Number of points for slope calculation
    """
    print("Calculating slope from %d points" % (2 * ntp + 1))
    
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
        
    fig = plt.figure(figsize=(fsize[0], fsize[1]))
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
    
    plt.show()

    return 0

#	----------------------------------------------------------------------------------------------------------