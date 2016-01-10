#! /usr/bin/env python
"""
  NAME:
    KS_showdown.py
  PURPOSE:
    Evaluates KS statistic on an ensemble of CMB realizations
    Compares different methods of template fitting for ISW map
  USES:
    make_Cmatrix.py: to make covariance matrix
    template_fit.py: contains test code and file names for testing

  MODIFICATION HISTORY:
    Written by Z Knight, 2016.01.06
    Added nested parameter; ZK, 2016.01.09

"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
#import astropy.io.fits as pf
import time # for measuring duration

import make_Cmatrix as mcm
import template_fit as tf

################################################################################
# testing code

def test(nSkies=10,useInverse=True,nested=False):
    """
    Purpose: do the KS testing
    Input:
        nSkies: the number of simulated CMBs to generate for K-S test
        useInverse: set this to invert C matrix.  Otherwise, cInvT method is used
            Default: True.  The cInvT method is garbage; don't use it.
        nested: NESTED vs RING parameter for healpy functions
    Returns: nothing
    """

    doHighPass   = True
    useBigMask   = False
    newInverse   = False
    matUnitMicro = False # option for matrices newer than 2015.12.11

    # this line gets the test data and does 4 template fits for files specified within
    print 'Starting template fitting on observed data... '
    M,mask,ISWvecs,modelVariances,ISWFiles,CMBFiles = tf.getTestData(doHighPass=doHighPass,useBigMask=useBigMask,
                                                                     newInverse=newInverse,matUnitMicro=matUnitMicro,
                                                                     useInverse=useInverse,nested=nested)
    if useInverse:
        cMatInv = M
    else:
        cMatrix = M
    del M

    # get Cl for generating CMB skies
    # shold be the same file as used to make C matrices referred to in tf.getTestData
    ISWoutFile = 'ISWout_scalCls.fits'
    ell,temps = mcm.getCl(ISWoutFile)
    if doHighPass:
        highpass = 12 # the lowest ell not zeroed out
        temps = np.concatenate((np.zeros(highpass),temps[highpass:]))

    # get ISW map
    doISWnum = 1 # ISW file 0 has radius to 10%, file 1 has radius to 2%
    ISWmap = hp.read_map(ISWFiles[doISWnum],nest=nested)
    ISWvec = ISWmap[np.where(mask)]

    fineNSIDE = 1024 # will downgrade to 64 after creating at high resolution
    NSIDE = 64
    lmax = 250 #to supercede the default value 3*NSIDE+1 and match what was used in making cMatrix
    fwhmMin = 5. #120.
    fwhmRad = fwhmMin/60.*np.pi/180.

    # to match current (2016.01.06) cMatrices, need to beamsmooth with gb(120')/gb(5')
    mbeam = hp.gauss_beam(120./60*np.pi/180,lmax=lmax) # 120 arcmin to be below W_l
    pbeam = hp.gauss_beam(5./60*np.pi/180,lmax=lmax)   # 5 arcmin beam; SMICA already has
    B_l = mbeam/pbeam
    temps = temps[:lmax+1]*B_l**2

    actualAmps = (0.0,5.0,10.0,15.0)
    nAmps = actualAmps.__len__()
    fitAmps = np.empty((nSkies,nAmps))
    fitVars = np.empty(nAmps)

    # Here is the synfast command that I used to make sims in IDL:
    # isynfast, Clfile, 'simCMB01.fits', nside=1024, iseed=1, fwhm_arcmin=5, simul_type=1

    startTime = time.time()
    for skyNum in range(nSkies):
        print 'starting sim ',skyNum+1,' of ',nSkies,'... '
        # generate CMB sky and add ISW map

        # CMB map parameters should match those used in C matrix
        CMBmap = hp.synfast(temps,NSIDE,lmax=lmax,pixwin=True,verbose=False)
        #CMBmap = hp.synfast(temps,fineNSIDE,lmax=2*NSIDE,fwhm=fwhmRad,verbose=False)
        if nested:
            CMBmap = hp.reorder(CMBmap,r2n=True)
        # rebin to NSIDE=64
        #if nested:
        #    CMBmap = hp.ud_grade(CMBmap,NSIDE,order_in='NESTED',order_out='NESTED')
        #else:
        #    CMBmap = hp.ud_grade(CMBmap,NSIDE,order_in='RING',order_out='RING')

        CMBvec = CMBmap[np.where(mask)]
        #print 'CMBvec: ',CMBvec
        for ampNum in range(nAmps):
            myISWvec = ISWvec*actualAmps[ampNum]
            CMBsum = CMBvec + myISWvec
            if useInverse:
                amp,var = tf.templateFit(cMatInv,ISWvec,CMBsum)
            else: # NOOOO!!! it will eat you alive!
                amp,var = tf.templateFit2(cMatrix,ISWvec,CMBsum)
            fitAmps[skyNum,ampNum] = amp
            fitVars[ampNum] = var

    print 'time elapsed for '+str(nSkies)+' simulated skies: ',(time.time()-startTime)/60.,' minutes'

    # do K-S test of simulated amplitudes against expected normal distribution
    for ampNum,actual in enumerate(actualAmps):
        for iswNum in range(doISWnum): #well this for loop only has one time through for now
            myAmps = fitAmps[:,ampNum]
            sigma = np.sqrt(fitVars[ampNum])
            #print myAmps
            #print myAmps.shape
            KSresult = tf.KSnorm(myAmps,actual,sigma,showCDF=True)
            print 'K-S test result for simulated amplitude '+str(actual)+': ',KSresult


if __name__=='__main__':
    test()

