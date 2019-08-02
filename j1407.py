''' custom routines for reading in J1407 data'''

from astropy.io import ascii
import numpy as np
import pyfits

def j1407_photom():
    'read in J1407 photometry with errors'
    dfile = '1SWASP+J140747.93-394542.6-detrend.fits'
    with pyfits.open(dfile) as fpin:
        tgtdata = fpin[1].data

    fitcol = "SINVP_DETREND_010"

    polycol = "POLY1FIT"

    time = tgtdata['TIMEMHJD']
    flux = tgtdata['FLUX2_DECORR']/tgtdata[polycol]/tgtdata[fitcol]
    fluxe = tgtdata['FLUX2_ERR_DECORR']/tgtdata[polycol]/tgtdata[fitcol]
    camidx = np.r_[[int(i[:3]) for i in tgtdata['IMAGEID']]]
    return (time, flux, fluxe, camidx)

def j1407_photom_binned(fin, phot_tmin, phot_tmax):
    'read in binned j1407 photometry'
    print ('reading in j1407 photometry from %s' % fin)
    # load in J1407 binned photometry curve
    tin = ascii.read(fin)
    time = tin['time']
    flux = tin['flux']
    flux_err = tin['flux_rms']

    print ('restricting photometry to HJD range %.1f to %.1f' % (phot_tmin, phot_tmax))
    goodp = (time > phot_tmin) * (time < phot_tmax)

    flux = flux[goodp]
    time = time[goodp]
    flux_err = flux_err[goodp]

    print ('number of photometric points in J1407 light curve: %d' % time.size)

    return(time, flux, flux_err)

def j1407_gradients(fin):
    'read in gradients of j1407 light curve'
    print ('reading in gradients of light curve from %s' % fin)
    grad = ascii.read(fin)

    grad_time = grad['col1'] + 54222.
    grad_mag = np.abs(grad['col2'])
    grad_mag_norm = grad_mag/np.max(grad_mag)

    return(grad_time, grad_mag, grad_mag_norm)
