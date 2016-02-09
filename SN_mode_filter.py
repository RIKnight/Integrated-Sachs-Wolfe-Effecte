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
    template_fit.py
  MODIFICATION HISTORY:
    Written by Z Knight, 2016.01.11
    Fixed omission of signal in calculation of (S+N)/N; ZK, 2016.01.15
    Renamed mask and cMatrix files for easier use,
        changed nested default to False; ZK, 2016.01.19
    Consolidated filenames into getFilenames; split getRot into
        getRot and loadRot; removed squigFilter from SNavg; ZK, 2016.01.23
    Added third mask to list in getFilenames:
        difference between 9875 and 6110 pixel masks;
        added same options to getRot, loadRot; ZK, 2016.01.26
    Added masks 11-18 for clusters only, voids only, top 25, second 25,
        top 12, second 12, third 12, fourth 12; ZK, 2016.02.03
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time  # for measuring duration
import make_Cmatrix as mcm
import template_fit as tf


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

def getRot(signalCov,noiseCov,maskNum=0):
    """

    Args:
        signalCov: a numpy array containing a signal covariance matrix
        noiseCov: same for noise
        masknum: the mask to get matrices for.  See getFilenames.
            Default: 0.  This is not actually as masknum, and indicates
                that no file should be saved.
    Returns:
        w: a numpy array containing the S/N eigenvalues.
        v: a numpy array containing the S/N eigenvectors.
            This is also a matrix for rotating into a frame in which signal and noise
            covariance matrices are diagonal
    """
    # should be the same filenames as in loadRot
    covDir = '/Data/covariance_matrices/'

    eigValsFile = covDir+'SNeigvals'+str(maskNum)+'.npy'
    eigVecsFile = covDir+'SNeigvecs'+str(maskNum)+'.npy'

    startTime = time.time()
    noiseInvSqrt = invSqrt(noiseCov) #symmetric so no need to transpose in next line
    toRotate = np.dot(noiseInvSqrt,np.dot(signalCov,noiseInvSqrt))
    print 'calculating eigenvalues and eigenvectors of N^(-1/2).S.N^(-1/2)...'
    w,v = np.linalg.eig(toRotate)
    if maskNum != 0:
        np.save(eigValsFile,w)
        np.save(eigVecsFile,v)
    print 'time elapsed for eigen decomposition: ',(time.time()-startTime)/60.,' minutes'
    # this took 4.8 minutes (ring) for 3765 pixel mask
    # this took ... min (ring) for 6110 pixel mask
    # this took 85 minutes (ring) for 9875 pixel mask
    return w,v

def loadRot(maskNum):
    """
    Purpose:
        loads the S/N eigenvalues and eigenmodes from files
    Note:
        the files must have been already created using getRot or simSN
    Args:
        masknum: the mask number to load files for.  See getFilenames.
    Returns:

    """
    # must be the same filenames as in getRot
    covDir = '/Data/covariance_matrices/'
    eigValsFile = covDir+'SNeigvals'+str(maskNum)+'.npy'
    eigVecsFile = covDir+'SNeigvecs'+str(maskNum)+'.npy'
    if maskNum == 0:
        print 'no such mask'
        return 0
    w = np.load(eigValsFile)
    v = np.load(eigVecsFile)
    return w,v


def getFilenames(maskNum):
    """

    Args:
        maskNum: to select which mask to use.  These should match the
            filenames used in template_fit.getFilenames:
            1: the ~5 degree 6110 pixel mask (Default)
            2: the ~10 degree 9875 pixel mask
            3: the difference between the 9875 and 6100 pixel masks

            11: 50 voids
            12: 50 clusters
            13: top 25+25
            14: second 25+25
            15: top 12+12
            16: second 12+12
            17: third 12+12
            18: fourth 12+12
    Returns:
        maskFile,nMatrixFile,sMatrixFile,saveRotFile,saveRotInvFile
    """
    covDir = '/Data/covariance_matrices/'
    if maskNum == 1:
        maskFile       = covDir+'ISWmask6110_RING.fits'
        nMatrixFile    = covDir+'covar6110_ISWout_bws_hp12_RING.npy'
        sMatrixFile    = covDir+'covar6110_ISWin_bws_hp12_RING.npy'
        saveRotFile    = covDir+'SNrot_6110.npy'
        saveRotInvFile = covDir+'SNrot_6110_inv.npy'
    elif maskNum == 2:
        maskFile       = covDir+'ISWmask9875_RING.fits'
        nMatrixFile    = covDir+'covar9875_ISWout_bws_hp12_RING.npy'
        sMatrixFile    = covDir+'covar9875_ISWin_bws_hp12_RING.npy'
        saveRotFile    = covDir+'SNrot_9875.npy'
        saveRotInvFile = covDir+'SNrot_9875_inv.npy'
    elif maskNum == 3:
        maskFile       = covDir+'ISWmask9875minus6110_RING.fits'
        nMatrixFile    = covDir+'covar9875minus6110_ISWout_bws_hp12_RING.npy'
        sMatrixFile    = covDir+'covar9875minus6110_ISWin_bws_hp12_RING.npy'
        saveRotFile    = covDir+'SNrot_9875minus6110.npy'
        saveRotInvFile = covDir+'SNrot_9875minus6110_inv.npy'

    elif maskNum == 11:
        maskFile       = covDir+'ISWmask_voids_05.0deg_3612pix.fits'
        nMatrixFile    = covDir+'covar_voids_05.0deg_3612_ISWout.npy'
        sMatrixFile    = covDir+'covar_voids_05.0deg_3612_ISWin.npy'
        saveRotFile    = covDir+'SNrot_voids_05.0deg_3612.npy'
        saveRotInvFile = covDir+'SNrot_voids_05.0deg_3612_inv.npy'
    elif maskNum == 12:
        maskFile       = covDir+'ISWmask_clusters_05.0deg_3821pix.fits'
        nMatrixFile    = covDir+'covar_clusters_05.0deg_3821_ISWout.npy'
        sMatrixFile    = covDir+'covar_clusters_05.0deg_3821_ISWin.npy'
        saveRotFile    = covDir+'SNrot_clusters_05.0deg_3821.npy'
        saveRotInvFile = covDir+'SNrot_clusters_05.0deg_3821_inv.npy'
    elif maskNum == 13:
        maskFile       = covDir+'ISWmask_top25_05.0deg_3795pix.fits'
        nMatrixFile    = covDir+'covar_top25_05.0deg_3795_ISWout.npy'
        sMatrixFile    = covDir+'covar_top25_05.0deg_3795_ISWin.npy'
        saveRotFile    = covDir+'SNrot_top25_05.0deg_3795.npy'
        saveRotInvFile = covDir+'SNrot_top25_05.0deg_3795_inv.npy'
    elif maskNum == 14:
        maskFile       = covDir+'ISWmask_second25_05.0deg_3733pix.fits'
        nMatrixFile    = covDir+'covar_second25_05.0deg_3733_ISWout.npy'
        sMatrixFile    = covDir+'covar_second25_05.0deg_3733_ISWin.npy'
        saveRotFile    = covDir+'SNrot_second25_05.0deg_3733.npy'
        saveRotInvFile = covDir+'SNrot_second25_05.0deg_3733_inv.npy'
    elif maskNum == 15:
        maskFile       = covDir+'ISWmask_top12_05.0deg_2056pix.fits'
        nMatrixFile    = covDir+'covar_top12_05.0deg_2056_ISWout.npy'
        sMatrixFile    = covDir+'covar_top12_05.0deg_2056_ISWin.npy'
        saveRotFile    = covDir+'SNrot_top12_05.0deg_2056.npy'
        saveRotInvFile = covDir+'SNrot_top12_05.0deg_2056_inv.npy'
    elif maskNum == 16:
        maskFile       = covDir+'ISWmask_second12_05.0deg_2126pix.fits'
        nMatrixFile    = covDir+'covar_second12_05.0deg_2126_ISWout.npy'
        sMatrixFile    = covDir+'covar_second12_05.0deg_2126_ISWin.npy'
        saveRotFile    = covDir+'SNrot_second12_05.0deg_2126.npy'
        saveRotInvFile = covDir+'SNrot_second12_05.0deg_2126_inv.npy'
    elif maskNum == 17:
        maskFile       = covDir+'ISWmask_third12_05.0deg_2032pix.fits'
        nMatrixFile    = covDir+'covar_third12_05.0deg_2032_ISWout.npy'
        sMatrixFile    = covDir+'covar_third12_05.0deg_2032_ISWin.npy'
        saveRotFile    = covDir+'SNrot_third12_05.0deg_2032.npy'
        saveRotInvFile = covDir+'SNrot_third12_05.0deg_2032_inv.npy'
    elif maskNum == 18:
        maskFile       = covDir+'ISWmask_fourth12_05.0deg_2130pix.fits'
        nMatrixFile    = covDir+'covar_fourth12_05.0deg_2130_ISWout.npy'
        sMatrixFile    = covDir+'covar_fourth12_05.0deg_2130_ISWin.npy'
        saveRotFile    = covDir+'SNrot_fourth12_05.0deg_2130.npy'
        saveRotInvFile = covDir+'SNrot_fourth12_05.0deg_2130_inv.npy'

    else:
        print 'no such mask'
        return 0
    return maskFile,nMatrixFile,sMatrixFile,saveRotFile,saveRotInvFile


def getSNRot(newRot=False,maskNum = 1):
    """
    Purpose:
        Creates rotation matrices based on theoretical covariance matrices
        The forward rotation transforms a CMB data vector into a basis
            in which the covariance matrix is diagonal with value (S+N)/N
    Args:
        newRot: set this to recalculate the rotation matrices and save
            them to files.
            otherwise, they will be loaded from files
        maskNum: to select which mask to use
            1: the ~5 degree 6110 pixel mask (Default)
            2: the ~10 degree 9875 pixel mask
            3: the difference between the 9875 and 6100 pixel masks
    Uses:
        Covariance matrix files indicated in function
    Returns:
        the matrix for rotating into the SN frame and its inverse
    """

    # get filenames
    maskFile,nMatrixFile,sMatrixFile,saveRotFile,saveRotInvFile = getFilenames(maskNum)

    if newRot:
        startTime = time.time()
        # load covariance matrices that were created using the same mask as CMBvec:
        print 'loading noise covariance matrix from file ',nMatrixFile
        nMatrix = mcm.symLoad(nMatrixFile)
        print 'loading signal covariance matrix from file ',sMatrixFile
        sMatrix = mcm.symLoad(sMatrixFile)

        # define rotation function and transform the observational data vector:
        print 'creating rotation matrices...'
        # maybe I should combine these next two functions into one, since the each call eigh(nMatrix)
        nSqrt = mSqrt(nMatrix)
        nInvSqrRoot = invSqrt(nMatrix)
        SNeigvals,rMatrix = getRot(sMatrix,nMatrix,maskNum=maskNum) #SNeigvals not used in this function
        myRot = np.dot(rMatrix.T,nInvSqrRoot)
        invRot = np.dot(nSqrt,rMatrix)
        np.save(saveRotInvFile,invRot)
        np.save(saveRotFile,myRot)
        print 'time elapsed for rotation matrices creation: ',(time.time()-startTime)/60.,' minutes'
        # this took 7.8 min (ring) for 3765 pixel mask
        # this took 35.1 min (nested) and 33.2 min (ring) for 6110 pixel mask
        # this took 135 min. (ring) for 9875 pixel mask
    else:
        print 'loading rotation matrices...'
        invRot = np.load(saveRotInvFile)
        myRot = np.load(saveRotFile)
    print 'rotation matrices ready'

    return myRot,invRot


def SNavg(maskFile,SNrot,nSkies=1000,nested=False,newSim=False,doPlot=False):
    """
    Purpose:
        Creates nSkies simulated CMB skies using power spectrum, extracts a vector
            from each according to the mask, rotates into the SN frame,
            calculates <Tsquig**2>, and creates a mask for zeroing out
            low S/N modes in this frame
    Args:
        maskFile: The name of the file containing the mask that was used
            to create the covariance matrices
        SNrot:
            the matrix for rotating a vector into the S/N frame
        nSkies:
            the number of simulated skies to create
                Default: 1000
        nested:
            the NESTED vs RING parameter for healpy functions
            IMPORTANT: must match parameter used in covariance matrices
                Default: False
        newSim:
            set this to create new simulations for calculation of <Tsquig**2>.
            Otherwise, this will be loaded from file.
                Default: False
        doPlot:
            set this to plot S/N+1 from simulations
                Default: False
    Uses:
        power spectrum file
    Returns:
        CMBvecSNRsqrArray: a numpy array(nSkies,nPixels) containing <Tsquig**2>
            for each of nSkies skies
    """

    CMBvecSNRsqrArrayFile = '/Data/CMBvecSNRsqrArray.npy'

    if newSim:
        # get mask
        mask = hp.read_map(maskFile,nest=nested)

        # get Cl
        ISWoutFile   = 'ISWout_scalCls.fits'
        ISWinFile    = 'ISWin_scalCls.fits'
        ell,tempsOut = mcm.getCl(ISWoutFile)
        ell,tempsIn  = mcm.getCl(ISWinFile)
        temps = tempsOut+tempsIn

        lmax=250
        NSIDE=64
        # to match old (before 2016.01.18) cMatrices, need to beamsmooth with gb(120')/gb(5')
        oldMatrix = False
        mbeam = hp.gauss_beam(120./60*np.pi/180,lmax=lmax) # 120 arcmin to be below W_l
        if oldMatrix:
            pbeam = hp.gauss_beam(5./60*np.pi/180,lmax=lmax)   # 5 arcmin beam; SMICA already has
            B_l = mbeam/pbeam
        else:
            B_l = mbeam
        temps = temps[:lmax+1]*B_l**2

        doHighPass=True
        if doHighPass:
            highpass = 12 # the lowest ell not zeroed out
            temps = np.concatenate((np.zeros(highpass),temps[highpass:]))

        startTime = time.time()
        CMBvecSNRsqrArray = np.zeros((nSkies,np.sum(mask)))
        for skyNum in range(nSkies):
            print 'starting sim ',skyNum+1,' of ',nSkies,'... '
            # CMB map parameters should match those used in C matrix
            CMBmap = hp.synfast(temps,NSIDE,lmax=lmax,pixwin=True,verbose=False)
            if nested:
                CMBmap = hp.reorder(CMBmap,r2n=True)
            CMBvec = CMBmap[np.where(mask)]
            CMBvecSNR = np.dot(SNrot,CMBvec)
            CMBvecSNRsqrArray[skyNum] = CMBvecSNR**2

        print 'time elapsed for '+str(nSkies)+' simulated skies: ',(time.time()-startTime)/60.,' minutes'
        # took about 1.8 (nested) or 1.5 (ring) minutes for 1000 skies
        np.save(CMBvecSNRsqrArrayFile,CMBvecSNRsqrArray)
    else:
        CMBvecSNRsqrArray = np.load(CMBvecSNRsqrArrayFile)
    #CMBvecSNRsqrAvg = np.average(CMBvecSNRsqrArray,axis=0)
    #CMBvecSNRsqrStd = np.std(CMBvecSNRsqrArray,axis=0)

    """
    squigFilter = np.ones(np.sum(mask))             # all ones to start
    squigFilter[np.where(CMBvecSNRsqrAvg < SNmin)] = 0 # zero out values below threshold

    #doPlot=True
    if doPlot:
        sortedIndices = np.argsort(CMBvecSNRsqrAvg)
        plt.plot(CMBvecSNRsqrAvg[sortedIndices])
        plt.plot((CMBvecSNRsqrAvg*squigFilter)[sortedIndices])
        plt.title('S/N +1 variance of '+str(nSkies)+' sky simulations over SDSS region')
        plt.show()

    return squigFilter,mask
    """
    return CMBvecSNRsqrArray


def simSN(SNrot,InvRot,nSkies=1000,newSim=True,maskNum=1):
    """

    Args:
        SNrot: the matrix to rotate into the SN frame
        InvRot: the inverse of the SNrot matrix
        nSkies: the number of skies to simulate for averaging
            file size:
                4.9 Mb for nSkies = 1e2, 48.9 Mb for 1e3, 488.8 for 1e4;
                4.89 Gb for 1e5 (ran into VFAT file size limit and program crashed; needed 2.5 hrs to re-do)
            Default: 1000
        maskNum: to select which mask to use
            1: the ~5 degree 6110 pixel mask (Default)
            2: the ~10 degree 9875 pixel mask
            3: the difference between the 9875 and 6100 pixel masks
    Returns:
        average and standard deviation of nSkies
    """

    # get filenames
    maskFile,nMatrixFile,sMatrixFile,saveRotFile,saveRotInvFile = getFilenames(maskNum)

    newRot=False # set to True if new covariance matrices are being used
    if newRot:
        print 'loading noise covariance matrix from file ',nMatrixFile
        nMatrix = mcm.symLoad(nMatrixFile)
        print 'loading signal covariance matrix from file ',sMatrixFile
        sMatrix = mcm.symLoad(sMatrixFile)
        SNeigvals,rMatrix = getRot(sMatrix,nMatrix,maskNum=maskNum)
    else:
        SNeigvals,rMatrix = loadRot(maskNum)

    # open CMBsimSNRsqrArray from file.  Create file first with newSim if needed
    CMBvecSNRsqrArray = SNavg(maskFile,SNrot,nSkies=int(nSkies),nested=False,newSim=newSim)
    CMBvecSNRsqrAvg = np.average(CMBvecSNRsqrArray,axis=0)
    CMBvecSNRsqrStd = np.std(CMBvecSNRsqrArray,axis=0)
    return CMBvecSNRsqrAvg,CMBvecSNRsqrStd


def getSNfilter(SNeigvals,SNmin):
    """

    Args:
        SNeigvals:
        SNmin: the threshold value.

    Returns:
        SNfilter, a numpy array of the same length as SNeigvals that equals 1 at
            indices with values >= SNmin, and 0 otherwise
    """
    SNfilter = np.ones(SNeigvals.size)       # all ones to start
    SNfilter[np.where(SNeigvals <= SNmin)] = 0  # zero out values below threshold
    return SNfilter



################################################################################
# testing code

def test(nested=False,SNmin=1e-3,maskNum=1,newRot=False):
    """
    Purpose:
        test the other functions in this file
    Args:
        nested: the NESTED vs RING parameter for healpy functions
            Default: False
        SNmin: the minimum S/N to pass the filter when creating SNfilter
            Default: 1e-3
        maskNum: number indicating the mask and associated fileset to use.
            1: the 6110 pixel mask (Default)
            2: the 9875 pixel mask
        newRot: set this to find a new rotation matrix, eg, for a new mask.

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
    SNeigvals,rMatrix = getRot(sMatrix,nMatrix) #no save
    print 'check for diagonal: '
    diag = np.dot(rMatrix.T,np.dot(nInvSqrRoot,np.dot(sMatrix,np.dot(nInvSqrRoot,rMatrix))))
    print diag
    print 'check for zeroes: '
    print np.diag(diag)-SNeigvals

    # test mSqrt
    nSqrt = mSqrt(nMatrix)
    print 'check for zeros: '
    print np.dot(nSqrt,nSqrt)-nMatrix

    # test S/N rotation combinations
    myRot = np.dot(rMatrix.T,nInvSqrRoot)
    invRot = np.dot(nSqrt,rMatrix)
    print 'check for identity: '
    print np.dot(myRot,invRot)


    # start testing with simulated CMB data:

    # get rotation matrices for transforming into and out of SN frame
    SNrot,invRot = getSNRot(maskNum=maskNum, newRot=newRot)
    snRotate = lambda vecIn: np.dot(SNrot,vecIn)
    snInvRot = lambda vecIn: np.dot(invRot,vecIn)

    """
    # compare S/N from covariance matrices to S/N from sims:
    nSkies=100
    avg,std = simSN(SNrot,invRot,nSkies=nSkies,maskNum=maskNum)
    CMBvecSNRsqrAvg = avg
    CMBvecSNRsqrStd = std

    # get eigvals, eigvecs of S/N matrix: N**(-1/2).S.N**(-1/2)
    SNeigvals,rMatrix = loadRot(maskNum) # has to be after getSNrot(newRot=True), getRot, or simSN

    #plot together
    xVals = np.arange(CMBvecSNRsqrAvg.size)
    plt.errorbar(xVals,CMBvecSNRsqrAvg-1,yerr=CMBvecSNRsqrStd/np.sqrt(nSkies),
                 fmt='o',markersize=2)
    plt.plot(SNeigvals)  #these seem to be (semi)sorted without me having asked for it.  Why?
    plt.xlabel('eigenvalue number')
    plt.ylabel('S/N')
    plt.title('S/N eigenvalues (green), <Tsquig^2>-1 (blue); nSkies='+str(nSkies))
    plt.show()

    # subplot
    plt.errorbar(xVals,CMBvecSNRsqrAvg-1,yerr=CMBvecSNRsqrStd/np.sqrt(nSkies),
                 fmt='o',markersize=2)
    plt.plot(SNeigvals)  #these seem to be sorted without me having asked for it.  Why?
    plt.xlabel('eigenvalue number')
    plt.ylabel('S/N')
    plt.title('S/N eigenvalues (green), <Tsquig^2>-1 (blue); nSkies='+str(nSkies))
    plt.xlim((0,250))
    plt.ylim((-0.05,0.05))
    plt.show()
    """



    # start testing with observed CMB data:
    print 'starting on observational data: '

    # if not done in section above:
    SNeigvals,rMatrix = loadRot(maskNum) # has to be after getRot or simSN

    plt.semilogy(np.sort(SNeigvals))
    horizontal = np.ones(SNeigvals.size)*SNmin
    plt.plot(horizontal)
    plt.xlabel('SN eigenvalue number')
    plt.ylabel('eigenvalue')
    plt.title('sorted eigenvalues for mask number '+str(maskNum))
    plt.show()

    # this part is set up for masknum=1 only.

    # get CMBmap, ISWmap; extract vectors and rotate to SN frame
    CMBFiles,ISWFiles,maskFile,cMatrixFile,iCMatFile,starMaskFile = tf.getFilenames(maskNum=maskNum)
        #CMBFiles[0]: mask with anafast; CMBFiles[1]: no mask with anafast
        #ISWFiles[0]: R010 r10%; ISWFiles[1]: R010 r02%;
        #ISWFiles[2]: PSGplot060 r02%; ISWFiles[3]: R060 r02%;
        #ISWFiles[4]: PSGplot100 r02%; ISWFiles[5]: R100 r02%;
    CMBnum = 1
    ISWnum = 1
    CMBmap = hp.read_map(CMBFiles[CMBnum],nest=nested) *1e-6 # convert microK to K
    ISWmap = hp.read_map(ISWFiles[ISWnum],nest=nested)
    print 'loading mask from file ',maskFile
    mask = hp.read_map(maskFile,nest=nested)
    CMBvec = CMBmap[np.where(mask)]
    ISWvec = ISWmap[np.where(mask)]
    CMBvecSNR = snRotate(CMBvec)
    ISWvecSNR = snRotate(ISWvec)

    SNfilter = getSNfilter(SNeigvals,SNmin) #return value is a mask for rotated data vectors
    sortedIndices = np.argsort(SNeigvals)

    # in rotated frame square of vector is variance
    CMBrotVar = CMBvecSNR**2
    ISWrotVar = ISWvecSNR**2


    # lots of plots
    print 'here are 6 plots.'

    # is this the right way to apply SNfilter?  Does there need to be sorting done?  Depends on how filter was made.

    # plot CMB variance before and after filtering
    plt.plot(CMBrotVar[sortedIndices])
    plt.plot((CMBrotVar*SNfilter)[sortedIndices])
    plt.xlabel('eigenmode number')
    plt.title('CMB variance in S/N frame; unfiltered (blue), filtered (green)')
    plt.show()

    # plot ISW variance before and after filtering
    plt.plot(ISWrotVar[sortedIndices])
    plt.plot((ISWrotVar*SNfilter)[sortedIndices])
    plt.xlabel('eigenmode number')
    plt.title('ISW variance in S/N frame; unfiltered (blue), filtered (green)')
    plt.show()

    # plot CMB temperature before and after filtering
    plt.plot(CMBvecSNR[sortedIndices])
    plt.plot((CMBvecSNR*SNfilter)[sortedIndices])
    plt.xlabel('eigenmode number')
    plt.title('CMB vector rotated into S/N frame; unfiltered (blue), filtered (green)')
    plt.show()

    # plot ISW temperature before and after filtering
    plt.plot(ISWvecSNR[sortedIndices])
    plt.plot((ISWvecSNR*SNfilter)[sortedIndices])
    plt.xlabel('eigenmode number')
    plt.title('ISW vector rotated into S/N frame; unfiltered (blue), filtered (green)')
    plt.show()

    # plot filtered data after rotating back into original frame
    CMBvecSNRfilt = CMBvecSNR*SNfilter
    CMBvecSNRdeRot = snInvRot(CMBvecSNR)
    CMBvecSNRfiltDeRot = snInvRot(CMBvecSNRfilt)
    plt.plot(CMBvecSNRdeRot*1e6)
    plt.plot(CMBvecSNRfiltDeRot*1e6)
    plt.xlabel('pixel number')
    plt.ylabel('temperature (microK)')
    plt.title('unfiltered and filtered CMB temperature in original frame')
    plt.show()

    # plot filtered data after rotating back into original frame
    ISWvecSNRfilt = ISWvecSNR*SNfilter
    ISWvecSNRdeRot = snInvRot(ISWvecSNR)
    ISWvecSNRfiltDeRot = snInvRot(ISWvecSNRfilt)
    plt.plot(ISWvecSNRdeRot*1e6)
    plt.plot(ISWvecSNRfiltDeRot*1e6)
    plt.xlabel('pixel number')
    plt.ylabel('temperature (microK)')
    plt.title('unfiltered and filtered ISW temperature in original frame')
    plt.show()




if __name__ == '__main__':
    test()
