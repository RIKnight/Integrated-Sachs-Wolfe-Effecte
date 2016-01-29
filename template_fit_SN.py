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

def plotFits(ampSig,maskLabels,ISWlabels):
    """
    Purpose:
        plot the results of a set of template fits
    Args:
        ampSig:
            numpy array of [[[amp,var]]],
                first index is mask number: eg. 0: m6110, 1: m9875, 2: mDelta
                second index is ISW map number: eg. 10,40,...
        maskLabels:
            numpy string array the same length as the first index of ampVar
        ISWlabels:
            numpy integer array the same length as the second index of ampVar
    Returns:

    """
    plt.figure(1)
    plt.subplot(211)
    for maskNum in range(maskLabels.__len__()):
        plt.semilogy(ISWlabels,ampSig[maskNum,:,0],marker='.') #amplitudes
    plt.ylabel('amplitude of fit')
    plt.title('S/N filtered template fit results: small (blue), large (green), diff (red)')
    plt.subplot(212)
    for maskNum in range(maskLabels.__len__()):
        plt.plot(ISWlabels,ampSig[maskNum,:,0]/ampSig[maskNum,:,1],marker='.') #amplitude/standard deviations
    plt.ylabel('significance of fit')
    plt.xlabel('R [Mpc/h]')
    plt.show()



################################################################################
# testing code

def test(SNmin=1e-3,nested=False):

    # select which combinations will be evaluated
    CMBnum = 1 # for the non-masked anafast filtered map
    #ISWnumSet = np.array([2,3,4,5]) # Z vs PSG testing
    ISWnumSet = np.array([ 6, 7, 8, 9, 10, 11, 12])
    ISWlabels = np.array([10,40,60,80,100,120,160])
    maskNumSet = np.array([      1,      2,      3])
    maskLabels = np.array(['small','large','delta'])

    results = np.zeros((maskNumSet.size,ISWnumSet.size,2)) #2 for [amp,var]

    print 'Starting template fitting on observed data... '
    # get CMBmap, ISWmap
    #CMBFiles,ISWFiles,maskFile,cMatrixFile,iCMatFile,starMaskFile = tf.getFilenames(doHighPass=True, maskNum=maskNum)
    CMBFiles,ISWFiles = tf.getMapNames(doHighPass=True)
        #CMBFiles[0]: mask with anafast; CMBFiles[1]: no mask with anafast
        #ISWFiles[0]: R010 r10%; ISWFiles[1]: R010 r02%;
        #ISWFiles[2]: PSGplot060 r02%; ISWFiles[3]: R060 r02%;
        #ISWFiles[4]: PSGplot100 r02%; ISWFiles[5]: R100 r02%;
    CMBmap = hp.read_map(CMBFiles[CMBnum],nest=nested) *1e-6 # convert microK to K
    #ISWmap = hp.read_map(ISWFiles[ISWnum],nest=nested)

    newRot=False # set to False if rotation matrices have already been calculated
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

    np.save('tfSN_result.npy',results)
    print 'table of results: ([[[amp,stddev]]])'
    print results
    print 'done'

    plotFits(results,maskLabels,ISWlabels)

if __name__=='__main__':
    test()

