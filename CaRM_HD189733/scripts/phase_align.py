import numpy as np
from astropy.time import Time
from math import floor
import glob
from astropy.io import fits
from .bouchy_error import bouchy_err
import os
from .constants import storbitpar, pathlist, spect,ccfext
#from .constants import tepoch,P,r1, parguess, spect, pathlist
import subprocess
import scipy.optimize as op
from copy import deepcopy as dpcy
import matplotlib.pyplot as plt
import sys
import scipy.optimize as opt
import string
from .orders import orderl
#from joblib import Parallel, delayed
import multiprocessing
import emcee
from .priors import Gprior, Uprior


def get_data(path, st_type, orders):
    print(path)
    rvc = []
    rvc_err = []
    phase = []
    if spect == "HARPS":
        datafits = "/*ccf_"+str(st_type)+"_A.fits"
        dataspec = "/*e2ds_A.fits"

        data_path = sorted(glob.glob(path + datafits))
        dataspec_path = sorted(glob.glob(path + dataspec))
        dataspec_path = dpcy(dataspec_path[1:])

        # Calibration file
        cal_wav_ext = "/*wave_A.fits"
        cal_wav_file = sorted(glob.glob(path + cal_wav_ext))
        hdulist_cal_wav = fits.open(cal_wav_file[0])
        hdu_cal_wav = hdulist_cal_wav[0]
        datacal = hdu_cal_wav.data

    elif spect == "ESPRESSO" and len(pathlist) == 0:
        datafits = "/"+ccfext
        data_path = []
        data_path = np.append(data_path, glob.glob(path+datafits))
        data_path = sorted(data_path)
        print(data_path)
        datacal = None
    elif spect == "ESPRESSO" and len(pathlist) != 0:
        data_path = sorted(pathlist)
        print(data_path)
        datacal = None
    else:
        sys.exit("Spectrograph not recognized!")
    ldata = len(data_path)
    lorder = len(orders)
    rv = np.zeros((lorder, ldata))
    sigrv = np.zeros((lorder, ldata))
    phase = np.zeros((lorder, ldata))

    # Computes the wl ccfs from the individual orders
    if spect == "ESPRESSO":
        # Ver o valor do dataspec_path
        wl = np.matrix([readccf(data_path[k], None, [None for x in range(
            ldata)], st_type, [0, 170]) for k in range(ldata)]).T
        # plt.errorbar(np.array(wl[0])[0],np.array(wl[1])[0],yerr=np.array(wl[2])[0],fmt=".")
        # plt.show()
        # sys.exit()
        pool = multiprocessing.Pool()
        results = pool.map(getccf, ((order, data_path, None, [
                           None for x in range(ldata)], st_type) for order in orders))
        c = 0
    elif spect == "HARPS":
        wl = np.matrix([readccf(data_path[k], datacal, dataspec_path[k], st_type, [
                       0, 70]) for k in range(ldata)]).T
        pool = multiprocessing.Pool()
        results = pool.map(
            getccf, ((order, data_path, datacal, dataspec_path, st_type) for order in orders))
        c = 0
    for order in orders:
        rv[c] = np.array(results[c][1])[0]
        sigrv[c] = np.array(results[c][2])[0]
        phase[c] = np.array(results[c][0])[0]
        c += 1
    return(phase, rv, sigrv, np.array(wl[1])[0], np.array(wl[0])[0], np.array(wl[2])[0])


def RVerror(rv, ccf, eccf=1.):
    """
    Calculate the uncertainty on the radial velocity, following the same steps
    as the ESPRESSO DRS pipeline.

    Parameters
    ----------
    rv : array
    The velocity values where the CCF is defined.
    ccf : array
    The values of the CCF profile.
    eccf : array
    The errors on each value of the CCF profile.
    """

    ccf_slope = np.gradient(ccf, rv)
    ccf_sum = np.sum((ccf_slope / eccf)**2)
    return 1.0 / np.sqrt(ccf_sum)


def fgauss(x, a, mu, sigma2, c):
    return(c+a*np.exp(-(x-mu)**2./(2.*sigma2**2.)))


def getccf(k):
    order, data_path, datacal, dataspec_path, st_type = k
    ldata = len(data_path)
    f = [readccf(data_path[k], datacal, dataspec_path[k], st_type, order)
         for k in range(ldata)]
    f = np.matrix(f)
    f = f.T
    # plt.plot(f[1],f[2])
    # plt.show()
    return(f[0], f[1], f[2])


def readccf(filename, filecal, filespec, st_type, interval):
    ordi = orderl(interval)
    # CCF
    ccffile = fits.open(filename)
    header = ccffile[0].header
    # Leio RV do header:
    if spect == "ESPRESSO":
        rvheader = header["HIERARCH ESO QC CCF RV"]
        srvheader = header["HIERARCH ESO QC CCF RV ERROR"]
    else:
        None

    # Spectrum
    try:
        hdulistspec = fits.open(filespec)
    except:
        None
    if spect == "HARPS":
        hduspec = hdulistspec[0]
    elif spect == "ESPRESSO":
        None
    else:
        None
    try:
        data_spec = hduspec.data
    except:
        None

    # Time of observation MJD
    obs_time = header["MJD-OBS"]

    # Number of orbits since transit center epoch
    # tepoch=storbitpar["tepoch"]
    norb = (obs_time-storbitpar["tepoch"])/storbitpar["P"]

    # Number of full orbits
    nforb = round(norb)
    phase = norb-nforb
    if spect == "HARPS":
        dim = len(ccffile[0].data[:])
        ldim = len(ccffile[0].data[0])
    elif spect == "ESPRESSO":
        dim = len(ccffile[1].data[:])
        ldim = len(ccffile[1].data[0])
    else:
        None
    # Arrays for best fit radial velocity, error in the fit, order number
    rvp = np.zeros(dim)
    rv_err = np.zeros(dim)
    order = np.zeros(dim)
    ccfout = np.zeros(ldim)
    rvf = []
    ordsigrv = 0.
    ccferr = 0.
    if spect == "HARPS":
        ccc = 0
        for k in ordi:
            k = int(k)
            if int(sum(ccffile[0].data[k])) == 0:
                None
                #print("Warning: Some ccfs are not here!")
                # rvp[k],rv_err[k],order[k]=[0,0,0]
            else:
                ccf = ccffile[0].data[k]
                vstart, vstep = header['CRVAL1'], header['CDELT1']
                rv = np.linspace(vstart, vstart+vstep*(ccf.size-1), ccf.size)
                ccf = -ccf + abs(min(-ccf))*np.ones(len(ccf))
                ordsigrv += 1./RVerror(rv, ccf)**2.
                ccfout += ccf
                ccc += 1
    elif spect == "ESPRESSO":
        ccc = 0
        for k in ordi:
            k = int(k)
            if int(sum(ccffile[1].data[k])) == 0:
                None
                # Warns the user when some of the chosen orders are not present.
                print(("Warning: Some ccfs are not here!"+str(k)))
            else:
                ccf = ccffile[1].data[k]
                # print(k)
                # print(ccf)
                eccf = ccffile[2].data[k]
                vstart, vstep = header['HIERARCH ESO RV START'], header['HIERARCH ESO RV STEP']
                rv = np.linspace(vstart, vstart+vstep*(ccf.size-1), ccf.size)
                ccf = -ccf + abs(min(-ccf))*np.ones(len(ccf))
                """
				plt.plot(rv,ccf)
				plt.show(block=False)
				plt.pause(0.2)
				plt.close()
				"""
                ordsigrv += 1./RVerror(rv, ccf, eccf)**2.
                ccfout += ccf
                ccc += 1
    else:
        None
    #import sys
    # sys.exit("Done")
    rvout = rv
    sigrvccf = 1./np.sqrt(ordsigrv)
    mean = np.mean(rvout)
    sigma = np.sqrt(sum((rvout-mean)**2.)/(len(rvout)))
    guess = [max(ccfout), mean, sigma, 0.]
    fitpar, fitpar_cov = opt.curve_fit(fgauss, rvout, ccfout, guess)
    fitpar_err = np.sqrt(np.diag(fitpar_cov))
    """	
	print(fitpar,interval)
	print(phase,sigrvccf)
	plt.plot(rvout,ccfout,".")
	#print(fitpar)
	plt.plot(rvout,fgauss(rvout, fitpar[0], fitpar[1], fitpar[2],fitpar[3]))
	plt.show()
	"""
    return(phase, fitpar[1], sigrvccf, fitpar[2])
