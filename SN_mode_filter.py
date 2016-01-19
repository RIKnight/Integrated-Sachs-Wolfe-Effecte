#! /usr/bin/env python
"""
  NAME:
    SN_mode_filter.py
  PURPOSE:
    Uses covariance matrices for signal (s) and noise (n) with T=s+n model
        and a mask for selecting a data vector from a healpix map to find
        a rotation such that covariance matrices <s_i*S_j>, <n_i*n_j>
        are diagonal
    Creates an ensemble of simulations from power spectra for s and n,
        extracts data vector from these, rotates into diagonal frame,
        squares and averages these to find (S+N)/N eigenvalues.
  USES:

  MODIFICATION HISTORY:
    Written by Z Knight, 2016.01.11
    Fixed omission of signal in calculation of (S+N)/N; ZK, 2016.01.15

"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time  # for measuring duration
import make_Cmatrix as mcm


def mSqrt(matrix):
    """
    Purpose:
        calculate the square root of a positive semi-definite matrix
    Args:
        matrix: a positive semi-definite matrix; a symmetric matrix
    Returns:
        the square root of the matrix
    """
    w,v = np.linalg.eigh(matrix)
    sqrtW = np.diag(np.sqrt(w)) #square root of eigenvalues on diagonal
    return np.dot(v,np.dot(sqrtW,v.T))

def invSqrt(matrix):
    """
    Purpose:
        calculate the inverse square root of a positive semi-definite matrix
        Designed for use on covariance matrices
    Args:
        matrix: a positive semi-definite matrix; a symmetric matrix
    Returns:
        the inverse square root of the matrix
    """
    w,v = np.linalg.eigh(matrix)
    sqrtDInv = np.diag(1/np.sqrt(w)) #inverse of square root of eigenvalues on diagonal
    return np.dot(v,np.dot(sqrtDInv,v.T))

def getRot(signalCov,noiseCov):
    """

    Args:
        signalCov: a numpy array containing a signal covariance matrix
        noiseCov: same for noise

    Returns:
        w: a numpy array containing the S/N eigenvalues.
        v: a numpy array containing the S/N eigenvectors.
            This is also a matrix for rotating into a frame in which signal and noise
            covariance matrices are diagonal
    """
    noiseInvSqrt = invSqrt(noiseCov) #symmetric so no need to transpose in next line
    toRotate = np.dot(noiseInvSqrt,np.dot(signalCov,noiseInvSqrt))
    print 'calculating eigenvalues and eigenvectors of N^(-1/2).S.N^(-1/2)...'
    w,v = np.linalg.eig(toRotate)
    return w,v

def getSNRot(newRot=False):
    """
    Purpose:
        Creates rotation matrices based on theoretical covariance matrices
        The forward rotation transforms a CMB data vector into a basis
            in which the covariance matrix is diagonal with value (S+N)/N
    Args:
        newRot: set this to recalculate the rotation matrices and save
            them to files.
            otherwise, they will be loaded from files
    Uses:
        Covariance matrix files indicated in function
    Returns:
        the matrix for rotating into the SN frame and its inverse
    """

    # the files used
    nMatrixFile = 'covar6110_R010.npy'  # 'noise': the primaryCMB + ISWout
    sMatrixFile = 'covar6110_R010_ISWin_hp12.npy' # 'signal': ISWin

    if newRot:
        startTime = time.time()
        # load covariance matrices that were created using the same mask as CMBvec:
        # these use highpass=12, beamSmooth=True, pixWin=True, lmax=250, nested=True
        print 'loading noise covariance matrix from file ',nMatrixFile
        nMatrix = mcm.symLoad(nMatrixFile)
        print 'loading signal covariance matrix from file ',sMatrixFile
        sMatrix = mcm.symLoad(sMatrixFile)

        # define rotation function and transform the observational data vector:
        print 'creating rotation matrices...'
        nSqrt = mSqrt(nMatrix)
        nInvSqrRoot = invSqrt(nMatrix)
        SNeigvals,rMatrix = getRot(sMatrix,nMatrix)
        myRot = np.dot(rMatrix.T,nInvSqrRoot)
        invRot = np.dot(nSqrt,rMatrix)
        np.save('rot_6110_inv.npy',invRot)
        np.save('rot_6110.npy',myRot)
        print 'time elapsed for rotation matrices creation: ',(time.time()-startTime)/60.,' minutes'
        # this took 35.1 min for 6110 pixel mask
    else:
        print 'loading rotation matrices...'
        invRot = np.load('rot_6110_inv.npy')
        myRot = np.load('rot_6110.npy')
    print 'rotation matrices ready'

    return myRot,invRot


def SNavg(maskFile,SNrot,SNmin=1,nSkies=1000,nested=True,newSim=False):
    """
    Purpose:
        Creates nSkies simulated CMB skies using power spectrum, extracts a vector
            from each according to the mask,
    Args:
        maskFile: The name of the file containing the mask that was used
            to create the covariance matrices
        SNrot:
            the matrix for rotating a vector into the S/N frame
        SNmin:
            the minimum value of the SN transformed vector not be zeroed out
        nSkies:
            the number of simulated skies to create
                Default: 1000
        nested:
            the NESTED vs RING parameter for healpy functions
            IMPORTANT: must match parameter used in covariance matrices
        newSim:
            set this to create new simulations for calculation of <Tsquig**2>.
            Otherwise, this will be loaded from file.
                Default: False
    Uses:
        power spectrum file
    Returns:
        mask for zeroing out low S/N modes, mask for selecting data vector from image
    """

    mask = hp.read_map(maskFile,nest=nested)

    # get Cl
    ISWoutFile   = 'ISWout_scalCls.fits'
    ISWinFile    = 'ISWin_scalCls.fits'
    ell,tempsOut = mcm.getCl(ISWoutFile)
    ell,tempsIn  = mcm.getCl(ISWinFile)
    temps = tempsOut+tempsIn

    # to match current (2016.01.06) cMatrices, need to beamsmooth with gb(120')/gb(5')
    lmax=250
    NSIDE=64
    mbeam = hp.gauss_beam(120./60*np.pi/180,lmax=lmax) # 120 arcmin to be below W_l
    pbeam = hp.gauss_beam(5./60*np.pi/180,lmax=lmax)   # 5 arcmin beam; SMICA already has
    B_l = mbeam/pbeam
    temps = temps[:lmax+1]*B_l**2

    doHighPass=True
    if doHighPass:
        highpass = 12 # the lowest ell not zeroed out
        temps = np.concatenate((np.zeros(highpass),temps[highpass:]))

    if newSim:
        #nSkies = 1000
        tRotSum = np.zeros(np.sum(mask))
        startTime = time.time()
        for skyNum in range(nSkies):
            print 'starting sim ',skyNum+1,' of ',nSkies,'... '
            # CMB map parameters should match those used in C matrix
            CMBmap = hp.synfast(temps,NSIDE,lmax=lmax,pixwin=True,verbose=False)
            if nested:
                CMBmap = hp.reorder(CMBmap,r2n=True)
            CMBvec = CMBmap[np.where(mask)]
            #CMBvecSNR = snRotate(CMBvec)
            CMBvecSNR = np.dot(SNrot,CMBvec)
            tRotSum += CMBvecSNR**2
        CMBvecSNRsqrAvg = tRotSum/nSkies

        print 'time elapsed for '+str(nSkies)+' simulated skies: ',(time.time()-startTime)/60.,' minutes'
        # took about 1.8 minutes for 1000 skies
        np.save('CMBsimSNR.npy',CMBvecSNRsqrAvg)
    else:
        CMBvecSNRsqrAvg = np.load('CMBsimSNR.npy')


    squigFilter = np.ones(np.sum(mask))             # all ones to start
    squigFilter[np.where(CMBvecSNRsqrAvg < SNmin)] = 0 # zero out values below threshold

    doPlot=True
    if doPlot:
        sortedIndices = np.argsort(CMBvecSNRsqrAvg)
        plt.plot(CMBvecSNRsqrAvg[sortedIndices])
        plt.plot((CMBvecSNRsqrAvg*squigFilter)[sortedIndices])
        plt.title('S/N +1 variance of '+str(nSkies)+' sky simulations over SDSS region')
        plt.show()

    return squigFilter,mask



def test(nested=True,SNmin=1.0):
    """
    Purpose:
        test the other functions in this file
    Returns:

    """

    # test invSqrt
    mSize = 3
    nMatrix = np.random.rand(mSize,mSize)
    nMatrix = np.dot(nMatrix,nMatrix.T) #creates positive semi-definite matrix
    print 'nMatrix: '
    print nMatrix
    nInvSqrRoot = invSqrt(nMatrix)
    print 'check for identity: '
    print np.dot(nInvSqrRoot,np.dot(nMatrix,nInvSqrRoot))

    # test getRot
    sMatrix = np.random.rand(mSize,mSize)
    sMatrix = np.dot(sMatrix,sMatrix.T) #creates positive semi-definite matrix
    print 'sMatrix: '
    print sMatrix
    SNeigvals,rMatrix = getRot(sMatrix,nMatrix)
    print 'check for diagonal: '
    print np.dot(rMatrix.T,np.dot(nInvSqrRoot,np.dot(sMatrix,np.dot(nInvSqrRoot,rMatrix))))

    # test mSqrt
    nSqrt = mSqrt(nMatrix)
    print 'check for zeros: '
    print np.dot(nSqrt,nSqrt)-nMatrix

    # test S/N rotation combinations
    myRot = np.dot(rMatrix.T,nInvSqrRoot)
    invRot = np.dot(nSqrt,rMatrix)
    print 'check for identity: '
    print np.dot(myRot,invRot)

    # test getSNRot
    SNrot,invRot = getSNRot()
    snRotate = lambda vecIn: np.dot(SNrot,vecIn)
    snInvRot = lambda vecIn: np.dot(invRot,vecIn)


    # compare S/N from covariance matrices to S/N from sims:

    # open CMBsimSNRsqrAvg from file
    CMBvecSNRsqrAvg = np.load('CMBsimSNR.npy')

    # get eigenvalues of S/N matrix
    nMatrixFile = 'covar6110_R010.npy'  # 'noise': the primaryCMB + ISWout
    sMatrixFile = 'covar6110_R010_ISWin_hp12.npy' # 'signal': ISWin
    # load covariance matrices that were created using the same mask as CMBvec:
    # these use highpass=12, beamSmooth=True, pixWin=True, lmax=250, nested=True
    print 'loading noise covariance matrix from file ',nMatrixFile
    nMatrix = mcm.symLoad(nMatrixFile)
    print 'loading signal covariance matrix from file ',sMatrixFile
    sMatrix = mcm.symLoad(sMatrixFile)
    SNeigvals,rMatrix = getRot(sMatrix,nMatrix)

    #plot together
    plt.plot(SNeigvals)
    plt.plot(CMBvecSNRsqrAvg-1)
    plt.xlabel('eigenvalue number')
    plt.ylabel('S/N')
    plt.title('S/N eigenvalues (blue), <Tsquig^2>-1 (green)')
    plt.show()


    # work on observational data:
    print 'starting on observational data: '

    CMBfile  = '/Data/PSG/planck_filtered_nomask.fits' # no bright star mask when anafast used (ring)
    maskFile = '/Data/PSG/ten_point/ISWmask_din1_R010.fits' #(nested)

    squigFilter,mask = SNavg(maskFile,SNrot, SNmin=1,nSkies=1000,nested=True,newSim=False)

    # get SMICA map and extract data vector
    # CMBFiles have unit microK, ISWFiles have unit K, and cMatrixFile has units K**2
    # SMICA map highpass filtered, beamsmoothed, and window smoothed:
    CMBmap = hp.read_map(CMBfile,nest=nested) * 1e-6 # convert microK to K
    CMBvec = CMBmap[np.where(mask)]

    CMBvecSNR = snRotate(CMBvec)

    # look at variances in rotated frame
    rotVariances = CMBvecSNR**2
    sortedIndices = np.argsort(rotVariances)
    #plt.plot(rotVariances[sortedIndices])
    #plt.title('S/N +1 of (B+W)smoothed, hp12 SMICA over SDSS region')
    #plt.show()

    plt.plot(CMBvecSNR[sortedIndices])

    # zero out low S/N values and transform back and save
    CMBvecSNR  *= squigFilter
    CMBvecFiltered = snInvRot(CMBvecSNR)

    plt.plot(CMBvecSNR[sortedIndices])
    plt.title('unfiltered and filtered Tsquggle')
    plt.show()

    sortedIndices = np.argsort(CMBvec)
    plt.plot(CMBvecFiltered[sortedIndices])
    plt.plot(CMBvec[sortedIndices])
    plt.title('filtered and unfiltered T')
    plt.show()





if __name__ == '__main__':
    test()
