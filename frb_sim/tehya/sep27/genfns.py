#
#	Functions for simulating scattering 
#
#								AB, May 2024
#								TC, Sep 2024
#
#	Function list
#
#	gauspuls(fmhzarr, tmsarr, dfmhz, dtms, specind, peak, wms, locms, dmpccc):
#		Generate dynamic spectrum for a Gaussian pulse
#
#	scatter_dynspec(dspec, fmhzarr, tmsarr, dfmhz, dtms, taums, scindex):
#		Scatter a given dynamic spectrum
#
#	--------------------------	Import modules	---------------------------

import os, sys
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from genpars import *

mpl.rcParams['pdf.fonttype']	= 42
mpl.rcParams['ps.fonttype'] 	= 42
mpl.rcParams['savefig.dpi'] 	= 600
mpl.rcParams['font.family'] 	= 'sans-serif'
mpl.rcParams['font.size']		= 8
mpl.rcParams["xtick.major.size"]= 3
mpl.rcParams["ytick.major.size"]= 3

#	--------------------------	Analysis functions	-------------------------------
 
def gauspuls(fmhzarr, tmsarr, dfmhz, dtms, specind, peak, wms, locms, dmpccc, pa, l, v, dpadt, rm):
	
	#	Generate dynamic spectrum for Gaussian pulses
	
	gpdspec	=	np.zeros((4, fmhzarr.shape[0], tmsarr.shape[0]), dtype=float)
	ngp		=	len(specind) - 1
	plsarr	=	np.zeros(len(tmsarr), dtype=float)
	fmhzref	=	np.nanmedian(fmhzarr)
	lm2arr	=	(ccC*1.0e-8 / fmhzarr)**2
	lm20 	= 	np.nanmedian(lm2arr)

	for g in range(0,ngp):
	
                nrmarr	=	peak[g+1]*( (fmhzarr/np.nanmedian(fmhzarr))**specind[g+1] )
                plsarr	=	np.exp( -(tmsarr - locms[g+1])**2 / (2*(wms[g+1]**2)) )
                pa_arr 	= 	pa[g+1] + (tmsarr - locms[g+1])*dpadt[g+1]
		
                for c in range(0,len(fmhzarr)):
                    pa_farr 		= 	pa_arr + rm[g+1]*(lm2arr[c] - lm20)										# Faraday rotation
                    gpdspec[0, c]	=	gpdspec[0, c] + nrmarr[c]*plsarr	                        			
                    disdelms		=	4.15*dmpccc[g+1]*( ((1.0e3/fmhzarr[c])**2) - ((1.0e3/fmhzref)**2) )		# Dispersion delay
                    gpdspec[0, c]	=	np.roll(gpdspec[0, c], int(np.round(disdelms/dtms)))					# I
                    gpdspec[1, c]   = 	gpdspec[0, c] * l[g+1] * np.cos(2 * pa_farr)                            # Q
                    gpdspec[2, c]   =	gpdspec[0, c] * l[g+1] * np.sin(2 * pa_farr)                         	# U
                    gpdspec[3, c]   =   gpdspec[0, c] * v[g+1]    		                                        # V
                    	
	print("\nGenerating dynamic spectrum with %d Gaussian component(s)\n"%(ngp))
	plt.imshow(gpdspec[0, :], aspect='auto', interpolation='none', origin='lower', cmap='seismic', vmin=-np.nanmax(np.abs(gpdspec)), vmax=np.nanmax(np.abs(gpdspec)))
	plt.show()
	
	return gpdspec

#	--------------------------------------------------------------------------------

def scatter_dynspec(dspec, fmhzarr, tmsarr, dfmhz, dtms, taums, scindex):
        
    # Scatter a given dynamic spectrum
	
    scdspec = np.zeros(dspec.shape, dtype=float)
    taucms 	= taums * ((fmhzarr / np.nanmedian(fmhzarr)) ** scindex)
    
    for c in range(len(fmhzarr)):
        irfarr = np.heaviside(tmsarr, 1.0) * np.exp(-tmsarr / taucms[c]) / taucms[c]
        for stk in range(4): 
        	scdspec[stk, c] = np.convolve(dspec[stk, c], irfarr, mode='same')
	
    # add noise
    for stk in range(4): 
    	scdspec[stk] = scdspec[stk] + np.random.normal(loc=0.0, scale=1.0, size=(fmhzarr.shape[0], tmsarr.shape[0]))

    print(f"--- Scattering time scale = {taums:.2f} ms, {np.nanmin(taucms):.2f} ms to {np.nanmax(taucms):.2f} ms")
    
    fig, axs = plt.subplots(5, figsize=(10, 6))
    fig.suptitle('Scattered Dynamic Spectrum')
    
    # Plot the mean across all frequency channels (axis 0)
    axs[0].plot(np.nanmean(scdspec[0,:], axis=0), markersize=2 ,label='I')
    axs[0].plot(np.nanmean(scdspec[1,:], axis=0), markersize=2, label='Q')
    axs[0].plot(np.nanmean(scdspec[2,:], axis=0), markersize=2, label='U')
    axs[0].plot(np.nanmean(scdspec[3,:], axis=0), markersize=2, label='V')
    axs[0].set_title("Mean Scattered Signal over Time")
    axs[0].legend(loc='upper right')

    # Plot the 2D scattered dynamic spectrum
    axs[1].imshow(scdspec[0], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
    	vmin=-np.nanmax(np.abs(dspec[0])), vmax=np.nanmax(np.abs(dspec[0]))*5)
    axs[1].set_title("Mean Scattered Dynamic Spectrum Across Frequency Channels")

    axs[2].imshow(scdspec[1], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
    	vmin=-np.nanmax(np.abs(dspec[0])), vmax=np.nanmax(np.abs(dspec[0]))*2.5)

    axs[3].imshow(scdspec[2], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
    	vmin=-np.nanmax(np.abs(dspec[0])), vmax=np.nanmax(np.abs(dspec[0]))*2.5)

    axs[4].imshow(scdspec[3], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
   		vmin=-np.nanmax(np.abs(dspec[0])), vmax=np.nanmax(np.abs(dspec[0]))*5)
    axs[4].set_xlabel("Time (samples)")
    axs[4].set_ylabel("Frequency (MHz)")

    plt.tight_layout()
    plt.show()
    
    return scdspec


#	--------------------------------------------------------------------------------













































































