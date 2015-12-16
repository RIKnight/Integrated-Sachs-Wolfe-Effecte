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
    Added C matrix correction routine; ZK 2015.12.08
    Switched to Cholesky Decomposition Inverse; ZK, 2015.12.11
    Added useMicro, doBeamSmooth, doPixWin parameters; ZK, 2015.12.11
    Added cInvT procedure; ZK, 2015.12.14

"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time  # for measuring duration
import make_Cmatrix as mcm

def cInvT_test(cInv,T):
    """
    Function for examining intermediate C**-1*T values

    Args:
        cInv:
        T:

    Returns:
        nothing
    """
    cInvT = np.dot(cInv,T)
    #print np.array(cInv).shape
    plt.plot(cInvT)
    plt.title('C**-1 *T testing')
    plt.show()

def cInvT(covMat,Tvec):
    """
    Purpose:
        calculates C**-1*T using "left inverse" of T (a row vector)
    Args:
        covMat: numpy array containing a covariance matrix of field statistics
        Tvec: numpy array containing a column vector of field values
    Note:
        this function is copied int template_fit.py
    Returns:
        numpy array of C**-1*T (a column vector)
    """
    TInv = Tvec.T/np.dot(Tvec,Tvec)    # create left inverse of Tvec
    TInvC = np.dot(TInv,covMat)        # left multiply
    return TInvC.T/np.dot(TInvC,TInvC) # return right inverse

def test(case = 10,nTrials=1000):
    """
        function for testing the expectation value <T*C**-1*T> = N_pix
        case: selects which set of files to use for test
        nTrials: the number of random skies to use
    """

    # get Cl
    ISWoutFile = 'ISWout_scalCls.fits'
    ell, temps = mcm.getCl(ISWoutFile) # temps has units K**2

    # show Cl
    # mcm.showCl(ell,temps)

    # dictionary of sets of file names
    PSG = '/Data/PSG/'
    fileSets = {
    100:(PSG+'ten_point/ISWmask_din1_R010.fits','covar6110_R010_nhp.npy','invCovar_R010_nhp.npy'), #LU; lmax250, with BW
    101:(PSG+'ten_point/ISWmask_din1_R010.fits','covar6110_R010_nhp.npy','invCovar_R010_RD.npy'), #RD; lmax250, with BW
    102:(PSG+'ten_point/ISWmask_din1_R010.fits','covar6110_R010_nhp.npy','invCovar_R010_cho.npy'), #cho; lmax250, with BW
    1:(PSG+'small_masks/ISWmask_din1_R010_trunc1000.fits','covar1000b.npy','invCovar1000b.npy'), #lmax250
    2:(PSG+'small_masks/ISWmask_din1_R010_trunc1000.fits','covar1000c.npy','invCovar1000c.npy'), #lmax2000
    3:(PSG+'small_masks/ISWmask_din1_R010_trunc1000.fits','covar1000d.npy','invCovar1000d.npy'), #lmax250, with W
    4:(PSG+'small_masks/ISWmask_din1_R010_trunc1000.fits','covar1000e.npy','invCovar1000e.npy'), #lmax250, with B
    5:(PSG+'small_masks/ISWmask_din1_R010_trunc1000.fits','covar1000f.npy','invCovar1000f.npy'), #lmax250, no BW
    7:(PSG+'small_masks/ISWmask_din1_R060_trunc1.fits','covar478a.npy','invCovar478a.npy'), #lmax2000, no BW
    8:(PSG+'small_masks/ISWmask_din1_R060_trunc2.fits','covar525a.npy','invCovar525a.npy'), #lmax2000, no BW
    9:(PSG+'small_masks/ISWmask_din1_R060_trunc3.fits','covar500a.npy','invCovar500a.npy'), #lmax2000, no BW
    10:(PSG+'small_masks/ISWmask_din1_R060_trunc1.fits','covar478b.npy','invCovar478b.npy'), #lmax250, no BW
    11:(PSG+'small_masks/ISWmask_din1_R060_trunc2.fits','covar525b.npy','invCovar525b.npy'), #lmax250, no BW
    12:(PSG+'small_masks/ISWmask_din1_R060_trunc3.fits','covar500b.npy','invCovar500b.npy'), #lmax250, no BW
    13:(PSG+'small_masks/ISWmask_din1_R060_trunc1.fits','covar478c.npy','invCovar478c.npy'), #lmax250, with B
    14:(PSG+'small_masks/ISWmask_din1_R060_trunc2.fits','covar525c.npy','invCovar525c.npy'), #lmax250, with B
    15:(PSG+'small_masks/ISWmask_din1_R060_trunc3.fits','covar500c.npy','invCovar500c.npy'), #lmax250, with B
    16:(PSG+'small_masks/ISWmask_din1_R060_trunc1.fits','covar478d.npy','invCovar478d.npy'), #lmax250, no B, with W
    17:(PSG+'small_masks/ISWmask_din1_R060_trunc2.fits','covar525d.npy','invCovar525d.npy'), #lmax250, no B, with W
    18:(PSG+'small_masks/ISWmask_din1_R060_trunc3.fits','covar500d.npy','invCovar500d.npy'), #lmax250, no B, with W
    19:(PSG+'small_masks/ISWmask_din1_R060_trunc1.fits','covar478e.npy','invCovar478e.npy'), #lmax250, with BW
    20:(PSG+'small_masks/ISWmask_din1_R060_trunc2.fits','covar525e.npy','invCovar525e.npy'), #lmax250, with BW
    21:(PSG+'small_masks/ISWmask_din1_R060_trunc3.fits','covar500e.npy','invCovar500e.npy') #lmax250, with BW

    }
    BWcontrol = {
        102:(True,True),
        10:(False,False),
        11:(False,False),
        12:(False,False),
        13:(True,False),
        14:(True,False),
        15:(True,False),
        16:(False,True),
        17:(False,True),
        18:(False,True),
        19:(True,True),
        20:(True,True),
        21:(True,True)
    }


    #case = 0
    maskFile,saveMatrixFile,saveInvCMFile = fileSets.get(case)
    doBeamSmooth,doPixWin = BWcontrol.get(case)

    newMatrix = False#True
    useInverse = False # set to True to use matrix inversion, False to use cInvT method
    useMicro = False
    #doBeamSmooth = True#False
    #doPixWin = False
    useRD = False#True # overrides useCho
    useCho = True # overrides default: use LU
    if newMatrix:
        startTime = time.time()
        print 'starting C matrix creation...'
        covMat = mcm.makeCmatrix(maskFile, ISWoutFile, highpass=0, beamSmooth=doBeamSmooth,
                                 pixWin=doPixWin, lmax=250, useMicro=useMicro)
        #covMat = mcm.cMatCorrect(covMat) #correction due to estimating mean from sample
        mcm.symSave(covMat, saveMatrixFile)
        if useInverse:
            if useRD:
                print 'starting eigen decomposition...'
                w, v = np.linalg.eigh(covMat)
                print 'starting RD inversion...'
                invCMat = mcm.RDInvert(w, v)
            elif useCho:
                print 'starting Cholesky inversion...'
                invCMat = mcm.choInvert(covMat)
            else: # use LU
                print 'starting LU inversion...'
                invCMat = np.linalg.inv(covMat)
            print 'time elapsed: ', int((time.time() - startTime) / 60), ' minutes'
            np.save(saveInvCMFile, invCMat)
    else:
        if useInverse:
            invCMat = np.load(saveInvCMFile)
        else:
            covMat = mcm.symLoad(saveMatrixFile)

    # load the mask - nest=True for mask!
    mask = hp.read_map(maskFile, nest=True)

    # apply gaussbeam before synfast?
    #lmax = 250
    #Wpix = hp.pixwin(64)
    #W_l = Wpix[:lmax+1]
    #mbeam = hp.gauss_beam(120./60*np.pi/180,lmax=lmax) # 120 arcmin to be below W_l
    #B_l = mbeam[:lmax+1]
    #temps = temps[:lmax+1]*B_l**2 #*W_l**2

    #nTrials = 1000
    NSIDE=64
    lmax=250
    fwhmMin = 120.
    fwhmRad = fwhmMin/60.*np.pi/180.

    chiSqResults = np.zeros(nTrials)
    for trial in range(nTrials):
        print 'starting trial ',trial+1,' of ',nTrials
        if doBeamSmooth:
            if doPixWin:
                map = hp.synfast(temps, NSIDE, lmax=lmax, fwhm=fwhmRad, pixwin=True)#, verbose=False)
            else:
                map = hp.synfast(temps, NSIDE, lmax=lmax, fwhm=fwhmRad)#, verbose=False)
        else:
            if doPixWin:
                map = hp.synfast(temps, NSIDE, lmax=lmax, pixwin=True)#, verbose=False)
            else:
                map = hp.synfast(temps, NSIDE, lmax=lmax)#, verbose=False)
        Tvec = map[np.where(mask)] #apply mask
        if useMicro:
            Tvec = Tvec*1e6 #convert K to microK
        if useInverse:
            chiSqResults[trial] = np.dot(Tvec,np.dot(invCMat,Tvec))
            #cInvT_test(invCMat,Tvec)
        else:
            chiSqResults[trial] = np.dot(Tvec,cInvT(covMat,Tvec))
    csqMean = np.mean(chiSqResults)
    print 'average of chiSquared results: ',csqMean
    plt.plot(chiSqResults,linestyle="none",marker='+')
    plt.title('chiSquare test average: '+str(csqMean))
    plt.xlabel('draw number')
    plt.ylabel('chi squared result')
    plt.show()

if __name__ == '__main__':
    test()
