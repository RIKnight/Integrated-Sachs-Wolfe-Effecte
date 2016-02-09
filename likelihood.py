#! /usr/bin/env python
"""
  NAME:
    likelihood

  PURPOSE:
    Evaluate the CMB likelihood functions

  USES:
    SN_mode_filter.py
    ISW_template.py
    template_fit.py

  MODIFICATION HISTORY:
    Written by Z Knight, 2016.01.31

"""


import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
#from scipy.interpolate import interp1d
#from os import listdir
import time # for measuring duration

import SN_mode_filter as sn
import ISW_template as ISWt
import template_fit as tf


def logLikelihood(CMBvec,ISWvec,SNmin=1e-3,maskNum=1,invCovar=None):
    """
    Purpose:
        evaluate the log likelihood for the model T_ISWout = T_obs - T_ISWin
    Args:
        CMBvec: a vector of the observed CMB
        ISWvec: a vector of the ISW signal, equal to template*amplitude
        SNmin: the threshold for zeroing out modes that have less S/N than this
        maskNum: a number indicating which mask was used to create vectors and S/N
            rotation matrices.
            1: 6110 pixel (small) (Default)
            2: 9875 pixel (large)
            3: the difference between 2 and 1
        invCovar: the inverse covariance.  In the S/N frame, this is considered
            to be the identity.  For explicit calculations using the inverse
            covariance matrix, pass it to this variable.
            Default: invCovar is assumed to be the identity and omitted from calculation.

    Returns:
        the log likelihood
    """
    # load rotation matrices and get ready to make SN cut
    SNrot,invRot = sn.getSNRot(maskNum=maskNum, newRot=False)
    SNeigvals,rMatrix = sn.loadRot(maskNum) # has to be after getRot or simSN
    SNfilter = sn.getSNfilter(SNeigvals,SNmin) #return value is a mask for rotated data vectors

    # make rotations and SN cut
    CMBvecSNR = np.dot(SNrot,CMBvec)
    CMBvecSNR *= SNfilter
    ISWvecSNR = np.dot(SNrot,ISWvec)
    ISWvecSNR *= SNfilter

    # the difference between the observed CMB and the model
    CMBdiff = CMBvecSNR - ISWvecSNR

    #multiply
    if invCovar is None: # Assume invCovar is an identity matrix in SN frame
        logLike = np.dot(CMBdiff,CMBdiff)
    else: # use invCovar in calculation
        invCovarSNR = np.dot(SNrot,np.dot(invCovar,SNrot.T))
        logLike = np.dot(CMBdiff,np.dot(invCovarSNR,CMBdiff))

    return -0.5*logLike

def logLikelihood2(CMBmap,ISWobj,mask,ampParam,expParam,SNmin=1e-3,maskNum=1,invCovar=None):
    """
    Purpose: wrapper around the regular logLikelihood function.
        This version accepts a CMB healpix map and an ISW template object
        rather than data vectors.
    Args:
        CMBmap: a heaplix map of observed CMB
        ISWobj: an ISW_template.ISWtemplate object containing the template to fit
        mask: the mask indicating which pixels to use from CMBmap and ISWmap
        ampParam:
        expParam:
        masknum: set this to the number of the mask used to create the
            S/N rotation matrices
        **kwargs: extra keyword args to pass to logLikelihood

    Returns:
        the log likelihood
    """
    # don't want to keep loading the mask, so this is not included here
    # get mask
    #maskFile,cMatrixFile,iCMatFile,starMaskFile = tf.getMaskNames(maskNum=maskNum)
    #mask = hp.read_map(maskFile,nest=nested)

    # extract vectors
    CMBvec = CMBmap[np.where(mask)]
    ISWmap = ISWobj.getMap(ampParam,expParam)
    ISWvec = ISWmap[np.where(mask)]

    # modification for testing only
    #ISWmap = hp.read_map('/Data/PSG/hundred_point/ISWmap_RING_R060_hp12.fits',nest=False)
    #ISWmap = hp.read_map('/Data/PSG/hundred_point/ISWmap_RING_R100_hp12.fits',nest=False)
    #ISWvec = ISWmap[np.where(mask)]*ampParam


    return logLikelihood(CMBvec,ISWvec,maskNum=maskNum,SNmin=SNmin,invCovar=invCovar)



################################################################################
# testing code

def test(nested=False,maskNum=1,avgRc=30,avgRv=95):

    # create grid of points to test at
    nAmps = 5#11
    nExps = 6#11
    #amplitudes = np.linspace(0.5,1.5,nAmps)
    #exponents = np.linspace(-0.5,0.5,nExps)
    #amplitudes = np.linspace(1.25,1.35,nAmps) #for R060 test
    #amplitudes = np.linspace(0.25,0.45,nAmps) #for R100 test
    amplitudes = np.linspace(-0.4,0.4,nAmps)
    exponents = np.linspace(-0.1,0.5,nExps)
    results = np.zeros((nAmps,nExps))

    # get test data
    maskFile,cMatrixFile,iCMatFile,starMaskFile = tf.getMaskNames(maskNum=maskNum)
    mask = hp.read_map(maskFile,nest=nested)
    CMBFiles, ISWFiles = tf.getMapNames()
    CMBmap = hp.read_map(CMBFiles[1],nest=nested) *1e-6 # convert microK to K

    # get ISW template
    myTemplate = ISWt.ISWtemplate(avgRc=avgRc,avgRv=avgRv,newMaps=False)#,templateFile=templateFile)
    #ISWmap = myTemplate.getMap(1.0,0.0) #amplitude,exponent


    # do fitting
    startTime = time.time()
    for ampNum, amp in enumerate(amplitudes):
        for expNum, exp in enumerate(exponents):
            print 'starting amp = ',amp,', exp = ',exp
            #ISWmap = myTemplate.getMap(amp,exp)
            results[ampNum,expNum] = logLikelihood2(CMBmap,myTemplate,mask,amp,exp)

    print 'time elapsed for ',nAmps*nExps, 'likelihood evaluations: ',(time.time()-startTime)/60.,' minutes'
    print results

    # plots columns of 2d array as lines
    for ampNum, amp in enumerate(amplitudes):
        plt.plot(exponents,results[ampNum]*-2, #-2 to convert log likelihood to chi squared
                 label = str(amp))
    plt.xlabel('exponents')
    plt.title('Chi^2: each line for different amplitude value')
    plt.legend()
    plt.show()


if __name__=='__main__':
  test()



