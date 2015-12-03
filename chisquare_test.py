#! /usr/bin/env python
"""
  NAME:
    chisquare_test.py
  PURPOSE:
    Program to check that T*C^-1*T follows chi square distribution as expected
  USES:
    make_Cmatrix.py
  MODIFICATION HISTORY:
    Written by Z Knight, 2015.11.30

"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time  # for measuring duration
import make_Cmatrix as mcm


def test(case = 0,nTrials=100):
    """
        function for testing the other functions in this file
        case: selects which set of files to use for test
        nTrials: the number of random skies to use
    """

    # get Cl
    ISWoutFile = 'ISWout_scalCls.fits'
    ell, temps = mcm.getCl(ISWoutFile)

    # show Cl
    # mcm.showCl(ell,temps)

    # mask to indicate which pixels to include in data array
    PSG = '/Data/PSG/'
    fileSets = {
    0:(PSG+'small_masks/ISWmask_din1_R010_trunc1000.fits','covar1000.npy','invCovar1000.npy'),  #used RD inv
    1:(PSG+'small_masks/ISWmask_din1_R010_trunc1000.fits','covar1000.npy','invCovar1000b.npy'), #used LU inv
    2:(PSG+'ten_point/ISWmask_din1_R010.fits','covar6110_R010_nhp.npy','invCovar_R010_nhp.npy')
        # try mask with clump near equator to check for latitude vs colatitude differences
    }

    #case = 0
    maskFile,saveMatrixFile,saveInvCMFile = fileSets.get(case)

    newMatrix = True
    useRD = False  #Testing indicates that RDInvert creates inverses that are not positive definite. please fix this.
    if newMatrix:
        startTime = time.time()
        print 'starting C matrix creation...'
        covMat = mcm.makeCmatrix(maskFile, ISWoutFile)
        mcm.symSave(covMat, saveMatrixFile)
        if useRD:
            print 'starting eigen decomposition...'
            w, v = np.linalg.eigh(covMat)
            print 'starting RD inversion...'
            invCMat = mcm.RDInvert(w, v)
        else:
            invCMat = np.linalg.inv(covMat)
        print 'time elapsed: ', int((time.time() - startTime) / 60), ' minutes'
        np.save(saveInvCMFile, invCMat)

    else:
        # covMat = mcm.symLoad(saveMatrixFile)
        invCMat = np.load(saveInvCMFile)

    # load the mask - nest=True for mask!
    mask = hp.read_map(maskFile, nest=True)

    #nTrials = 100
    chiSqResults = np.zeros(nTrials)
    for trial in range(nTrials):
        print 'starting trial ',trial+1,' of ',nTrials
        map = hp.synfast(temps, 64)
        Tvec = map[np.where(mask)]
        chiSqResults[trial] = np.dot(Tvec,np.dot(invCMat,Tvec))
    csqMean = np.mean(chiSqResults)
    print 'average of chiSquared results: ',csqMean
    plt.plot(chiSqResults,linestyle="none",marker='+')
    plt.title('chiSquare test average: '+str(csqMean))
    plt.xlabel('draw number')
    plt.ylabel('chi squared result')
    plt.show()

if __name__ == '__main__':
    test()
