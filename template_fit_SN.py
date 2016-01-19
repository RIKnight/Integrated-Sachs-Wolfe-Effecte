#! /usr/bin/env python
"""
  NAME:
    template_fit_SN.py
  PURPOSE:
    Program to calculate amplitude of template on CMB using non-direct methods
  USES:
    make_Cmatrix.py
    template_fit.py

  MODIFICATION HISTORY:
    Written by Z Knight, 2016.01.12

"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
#import astropy.io.fits as pf
#import time # for measuring duration

import make_Cmatrix as mcm
import template_fit as tf
import SN_mode_filter as sn


################################################################################
# testing code

def test(SNmin=1.0,nested=True):
    """
        Code for testing the SN filtering
    Args:
        SNmin:
            the minimum SN value to pass the filter
        nested:
            the NESTED vs RING parameter for healpy functions
    Returns:

    """

    print 'Starting template fitting on observed data... '
    # this line gets the test data and does 4 template fits for files specified within
    #cMatInv,mask,ISWvecs,modelVariances,ISWFiles,CMBFiles = tf.getTestData(doHighPass=True,nested=nested)

    # filenames
    CMBfile  = '/Data/PSG/planck_filtered_nomask.fits' # no bright star mask when anafast used (ring)
    ISWfile  = '/Data/PSG/hundred_point/ISWmap_RING_R010_hp11.fits' #(ring)
    # small mask
    maskFile = '/Data/PSG/ten_point/ISWmask_din1_R010.fits' #(nested)
    cMatrixFile = 'covar6110_R010.npy' #(nested)
    iCMatFile = 'invCovar_R010_cho.npy' #(nested)
    # big mask
    #maskFile = '/Data/PSG/hundred_point_bad/ISWmask2_din1_R160.fits'
    #cMatrixFile = 'covar9875_R160b.npy'
    #iCMatFile = 'invCovar_R160_RD.npy'

    # get rotation matrix
    SNrot,invRot = sn.getSNRot()

    # get SN filter and image mask
    #SNmin = 1.0
    SNFilter,mask = sn.SNavg(maskFile,SNrot, SNmin=SNmin)
    #mask = hp.read_map(maskFile,nest=nested)

    # get SMICA map and extract data vector
    # CMBFiles have unit microK, ISWFiles have unit K, and cMatrixFile has units K**2
    # SMICA map highpass filtered, beamsmoothed, and window smoothed:
    CMBmap = hp.read_map(CMBfile,nest=nested) * 1e-6 # convert microK to K
    CMBvec = CMBmap[np.where(mask)]
    ISWmap = hp.read_map(ISWfile,nest=nested)
    ISWvec = ISWmap[np.where(mask)]

    # rotate into SN frame
    CMBvecSNR = np.dot(SNrot,CMBvec)
    ISWvecSNR = np.dot(SNrot,ISWvec)

    # apply SN filter
    plt.plot(CMBvecSNR)
    CMBvecSNR *= SNFilter
    ISWvecSNR *= SNFilter
    plt.plot(CMBvecSNR)
    plt.title('rotated CMB data: unfiltered (blue) and filtered (green)')
    plt.xlabel('mode number')
    plt.ylabel('microK / microK')
    plt.show()

    # do template fit
    #cMatInv = np.load(iCMatFile)
    #amp,var = tf.templateFit(cMatInv,ISWvec,np.dot(invRot,CMBvecSNR))
    # in SN frame, template fit is simpler:
    var = 1/np.dot(ISWvecSNR,ISWvecSNR)
    amp = np.dot(CMBvecSNR,ISWvecSNR)*var
    print 'the amplitude of the fit: ',amp,' +- ',np.sqrt(var)
    print 'significance: ',amp/np.sqrt(var), ' sigma'

if __name__=='__main__':
    test()

