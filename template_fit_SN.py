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
    Completely revised; ZK, 2016.01.26
    Added starMaskFiles; ZK, 2016.01.27
    Added more mask sets; ZK, 2016.02.03

"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
#import astropy.io.fits as pf
#import time # for measuring duration
from scipy.interpolate import interp1d

import make_Cmatrix as mcm
import template_fit as tf
import SN_mode_filter as sn


def templateFitSN(CMBmap,ISWmap,SNmin=1e-3,nested=False,maskNum=1,newRot=False):
    """
    Purpose:
        Function to do a template fit including S/N filtering
    Args:
        CMBmap:
            a numpy array containing the healpix map for the CMB signal
        ISWmap:
            a numpy array containing the healpix map for the ISW signal
            or, this can be an array of maps, where the first index indicates
                a map and the second is a pixel in that map.
        SNmin:
            the minimum SN value to pass the filter
            Use SN_mode_filter to search for optimal value.
            Default: 1e-3
        nested:
            the NESTED vs RING parameter for healpy functions
            Default: False
        CMBnum, ISWnum:
            file numbers indicated in template_fit.py
            Defaults: both 1
        maskNum:
            Indicates which mask to use, as in SN_mode_filter.py
        newRot:
            set this to calculate a new rotation matrices for the mask
            indicated by maskNum.
            Default: False
    Returns:
        amp,var: the amplitude of the fit and its variance
    """
    # get mask file names
    maskFile,cMatrixFile,iCMatFile,starMaskFile = tf.getMaskNames(doHighPass=True, maskNum=maskNum)

    # get rotation matrices for transforming into and out of SN frame
    SNrot,invRot = sn.getSNRot(maskNum=maskNum, newRot=newRot)
    #snRotate = lambda vecIn: np.dot(SNrot,vecIn)
    #snInvRot = lambda vecIn: np.dot(invRot,vecIn)

    # get eigvals, eigvecs of S/N matrix: N**(-1/2).S.N**(-1/2)
    SNeigvals,rMatrix = sn.loadRot(maskNum) # has to be after getSNrot(newRot=True), getRot, or simSN

    # get SN filter and image mask
    mask = hp.read_map(maskFile,nest=nested)
    SNfilter = sn.getSNfilter(SNeigvals,SNmin)

    # extract data vectors, rotate into SN frame, and apply SN filter
    CMBvec = CMBmap[np.where(mask)]
    CMBvecSNR = np.dot(SNrot,CMBvec)
    CMBvecSNR *= SNfilter

    if ISWmap.ndim == 1:
        nMaps = 1
        ISWmap = np.array([ISWmap]) #adding extra layer of indexing to look like 2d array
    else:
        nMaps = ISWmap.shape[0]
    amplitudes = np.zeros(nMaps)
    variances = np.zeros(nMaps)
    for mapNum in range(nMaps):
        ISWvec = ISWmap[mapNum,np.where(mask)]
        ISWvecSNR = np.dot(SNrot,ISWvec[0]) # 0 since ISWvec has useless first index
        ISWvecSNR *= SNfilter

        # do template fit
        #cMatInv = np.load(iCMatFile)
        #amp,var = tf.templateFit(cMatInv,ISWvec,np.dot(invRot,CMBvecSNR))
        # in SN frame, template fit is simpler:
        variances[mapNum] = 1/np.dot(ISWvecSNR,ISWvecSNR)
        amplitudes[mapNum] = np.dot(CMBvecSNR,ISWvecSNR)*variances[mapNum]

    if nMaps == 1:
        return amplitudes[0],variances[0]
    else:
        return amplitudes,variances

def getLabels(setNumber):
    """

    Args:
        setNumber: controls which set of masks to evaluate
            1: for checking the effects of the size of the mask
            2: for voids vs clusters
            3: for first vs second 25
            4: for quartiles

    Returns:
        maskNumSet,maskLabels,tfSN_savefile,testName
    """
    if setNumber == 1:
        maskNumSet = np.array([      1,      2,      3])
        maskLabels = np.array(['small','large','lg.-sm.'])
        tfSN_savefile = 'tfSN_mask_size.npy'
        testName = 'effect of mask size'
    elif setNumber == 2:
        maskNumSet = np.array([      1,     11,     12])
        maskLabels = np.array(['cl+vo','voids','clusters'])
        tfSN_savefile = 'tfSN_voids_clusters.npy'
        testName = 'voids vs. clusters'
    elif setNumber == 3:
        maskNumSet = np.array([      1,     13,     14])
        maskLabels = np.array(['50+50','top25+25','next25+25'])
        tfSN_savefile = 'tfSN_halves.npy'
        testName = 'top 25+25 vs. next 25+25'
    elif setNumber == 4:
        maskNumSet = np.array([      1,     15,     16,     17,     18])
        maskLabels = np.array(['50+50','top12+12','2nd12+12','3rd12+12','4th12+12'])
        tfSN_savefile = 'tfSN_quarters.npy'
        testName = 'top 12+12 vs. 2nd, 3rd, 4th 12+12'
    else:
        print 'no such set of masks.'
        return 0
    return maskNumSet,maskLabels,tfSN_savefile,testName


def plotFits(ampSig,maskLabels,ISWlabels,testName=''):
    """
    Purpose:
        plot the results of a set of template fits
    Args:
        ampSig:
            numpy array of [[[amp,var]]],
                first index is mask number: eg. 0: m6110, 1: m9875, 2: mDelta
                second index is ISW map number: eg. 10,40,...
        maskLabels:
            numpy string array the same length as the first index of ampSig
        ISWlabels:
            numpy integer array the same length as the second index of ampSig
        testName:
            the name of the test to be used in plot title
    Returns:

    """
    plt.figure(1)
    plt.subplot(211)
    for maskNum in range(maskLabels.__len__()):
        plt.semilogy(ISWlabels,ampSig[maskNum,:,0],marker='.',
                     label=maskLabels[maskNum]) #amplitudes
    plt.ylabel('amplitude of fit')
    plt.title('S/N filtered template fit results: '+testName)
    plt.legend()
    plt.subplot(212)
    for maskNum in range(maskLabels.__len__()):
        plt.plot(ISWlabels,ampSig[maskNum,:,0]/ampSig[maskNum,:,1],marker='.'
                 ,label=maskLabels[maskNum]) #amplitude/standard deviations
    plt.ylabel('significance of fit')
    plt.xlabel('R [Mpc/h]')
    #plt.legend()
    plt.show()



################################################################################
# testing code

def test(SNmin=1e-3,nested=False,setNum=1,newRot=False):
    """

    Args:
        SNmin:
        nested:
        setNum: controls which set of masks to evaluate
            1: for checking the effects of the size of the mask
            2: for voids vs clusters
            3: for first vs second 25
            4: for quartiles
        newRot: set to False if rotation matrices have already been calculated
            Default: False

    Returns:

    """

    # select which combinations will be evaluated
    CMBnum = 1 # for the non-masked anafast filtered map
    #ISWnumSet = np.array([2,3,4,5]) # Z vs PSG testing
    ISWnumSet = np.linspace(1,16,16)+5
    ISWlabels = np.linspace(1,16,16)*10

    maskNumSet,maskLabels,tfSN_savefile,testName = getLabels(setNum)


    results = np.zeros((maskNumSet.size,ISWnumSet.size,2)) #2 for [amp,stddev]

    print 'Starting template fitting on observed data... '
    # get filenames and CMB map
    CMBFiles,ISWFiles = tf.getMapNames(doHighPass=True)
    CMBmap = hp.read_map(CMBFiles[CMBnum],nest=nested) *1e-6 # convert microK to K

    ISWmapSet = np.zeros((ISWnumSet.size,CMBmap.size)) #CMBmap has same size as ISW maps
    for maskIndex,maskNum in enumerate(maskNumSet):
        for ISWindex,ISWnum in enumerate(ISWnumSet):
            ISWmapSet[ISWindex,:] = hp.read_map(ISWFiles[ISWnum],nest=nested)
        amplitudes,variances = templateFitSN(CMBmap,ISWmapSet,SNmin=SNmin,nested=nested,maskNum=maskNum,newRot=newRot)
        stddevs = np.sqrt(variances)
        results[maskIndex,:,0]=amplitudes
        results[maskIndex,:,1]=stddevs

        for ISWindex,ISWnum in enumerate(ISWnumSet):
            print 'Mask number ',maskNum,', Map number ',ISWnum,':'
            amp   = results[maskIndex,ISWindex,0]
            sigma = results[maskIndex,ISWindex,1]
            print 'the amplitude of the fit: ',amp,' +- ',sigma,' (',amp/sigma,' sigma)'

    np.save(tfSN_savefile,results)
    print 'table of results: ([[[amp,stddev]]])'
    print results
    print 'done'

    plotFits(results,maskLabels,ISWlabels,testName=testName)

if __name__=='__main__':
    test()

