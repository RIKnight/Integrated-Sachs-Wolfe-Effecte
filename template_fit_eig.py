#! /usr/bin/env python
"""
  NAME:
    template_fit_eig.py
  PURPOSE:
    Program to calculate amplitude of template on CMB using non-direct methods
  USES:
    make_Cmatrix.py
    template_fit.py

  MODIFICATION HISTORY:
    Written by Z Knight, 2016.01.08
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

def test(nested=True):
    """
        Purpose: test the template fitting procedure
        Input:
            nested: set to True to use NESTED or False to use RING in healpy

        Returns: nothing
    """

    doHighPass = True
    # this line gets the test data and does 4 template fits for files specified within
    print 'Starting template fitting on observed data... '
    cMatInv,mask,ISWvecs,modelVariances,ISWFiles,CMBFiles = tf.getTestData(doHighPass=doHighPass,nested=nested)

    # get eigs of c matrix
    newEig = False#True
    if newEig:
        w,v = np.linalg.eigh(cMatrix)
        np.save('eigvecs.npy',v)
        np.save('eigvals.npy',w)
    else:
        v = np.load('eigvecs.npy')
        w = np.load('eigvals.npy')

    # get Cl for generating CMB skies
    # shold be the same file as used to make C matrices referred to in tf.getTestData
    ISWoutFile = 'ISWout_scalCls.fits'
    ell,temps = mcm.getCl(ISWoutFile)
    if doHighPass:
        highpass = 12 # the lowest C_l not zeroed out
        temps = np.concatenate((np.zeros(highpass),temps[highpass:]))

    # get ISW map
    doISWnum = 1 # ISW file 0 has radius to 10%, file 1 has radius to 2%
    ISWmap = hp.read_map(ISWFiles[doISWnum],nest=nested)
    ISWvec = ISWmap[np.where(mask)]

    NSIDE = 64
    lmax = 250 #to supercede the default value 3*NSIDE+1 and match what was used in making cMatrix
    fwhmMin = 5. #120.
    fwhmRad = fwhmMin/60.*np.pi/180.

    # to match current (2016.01.06) cMatrices, need to beamsmooth with gb(120')/gb(5')
    mbeam = hp.gauss_beam(120./60*np.pi/180,lmax=lmax) # 120 arcmin to be below W_l
    pbeam = hp.gauss_beam(5./60*np.pi/180,lmax=lmax)   # 5 arcmin beam; SMICA already has
    B_l = mbeam/pbeam
    temps = temps[:lmax+1]*B_l**2

    # use ensemble to check agreement with eigenvalues
    nSkies = 100
    CMBvecSqSum = np.zeros(np.sum(mask))
    for skyNum in range(nSkies):
        print 'starting sim ',skyNum+1,' of ',nSkies,'... '
        CMBmap = hp.synfast(temps,NSIDE,lmax=lmax,pixwin=True,verbose=False)
        if nested:
            CMBmap = hp.reorder(CMBmap,r2n=True)
        CMBvec = CMBmap[np.where(mask)]
        CMBvecRot = np.dot(v.T,CMBvec) #rotate CMBvec into frame where cMatrix is diagonal
        #print 'CMBvecRot: ',CMBvecRot
        CMBvecSqSum += CMBvecRot**2
    CMBvecSqAvg = CMBvecSqSum/nSkies

    """
    # plot eigenvalues unordered
    plt.plot(w)
    plt.xlabel('eigenvalue number')
    plt.title('unordered eigenvalues')
    plt.show()

    # plot eigenvalues unordered
    plt.plot(CMBvecSqAvg)
    plt.xlabel('eigenvalue number')
    plt.title('unordered expectation values')
    plt.show()

    # plot the result as a difference
    plt.plot(CMBvecSqAvg-w)
    plt.xlabel('eigenvalue number')
    plt.ylabel('<Tsquig^2> - lambda')
    plt.show()
    """

    # log plot of both together
    plt.semilogy(CMBvecSqAvg)
    plt.semilogy(w)
    plt.xlabel('eigenvalue number')
    plt.ylabel('eigenvalue')
    plt.title('<Tsquig^2> (blue) and lambda (green)')
    plt.show()

    # plot of ratio
    plt.plot(CMBvecSqAvg/w)
    plt.xlabel('eigenvalue number')
    plt.ylabel('<Tsquig^2> / lambda')
    plt.title('ratio of eigenvalues: ensemble <Tsquig^2> to lambda ')
    plt.show()


if __name__=='__main__':
    test()

