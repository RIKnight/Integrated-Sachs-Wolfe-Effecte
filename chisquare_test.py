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


def test():
    """
        function for testing the other functions in this file
    """

    # get Cl
    ISWoutFile = 'ISWout_scalCls.fits'
    ell, temps = mcm.getCl(ISWoutFile)

    # show Cl
    # mcm.showCl(ell,temps)

    # mask to indicate which pixels to include in data array
    maskFile = '/Data/PSG/small_masks/ISWmask_din1_R010_trunc1000.fits'
    saveMatrixFile = 'covar1000.npy'
    saveInvCMFile = 'invCovar1000.npy'

    newMatrix = True
    if newMatrix:
        startTime = time.time()
        print 'starting C matrix creation...'
        covMat = mcm.makeCmatrix(maskFile, ISWoutFile)
        mcm.symSave(covMat, saveMatrixFile)
        print 'starting eigen decomposition...'
        w, v = np.linalg.eigh(covMat)
        print 'starting RD inversion...'
        invCMat = mcm.RDInvert(w, v)
        print 'time elapsed: ', int((time.time() - startTime) / 60), ' minutes'
        np.save(saveInvCMFile, invCMat)

    else:
        # covMat = mcm.symLoad(saveMatrixFile)
        invCMat = np.load(saveInvCMFile)

    # load the mask - nest=True for mask!
    mask = hp.read_map(maskFile, nest=True)

    nTrials = 1
    chiSqResults = np.zeros(nTrials)
    for trial in range(nTrials):
        map = hp.synfast(temps, 64)
        Tvec = map[np.where(mask)]


if __name__ == '__main__':
    test()
