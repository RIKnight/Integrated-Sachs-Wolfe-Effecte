#! /usr/bin/env python
"""
  NAME:
    mask_check.py
  PURPOSE:
    Program to calculate amplitude of template on CMB using various masks
  USES:
    make_Cmatrix.py
    template_fit.py
    fits file containing mask indicating the set of pixels to use

  MODIFICATION HISTORY:
    Written by Z Knight, 2016.01.20
    Added starMaskFiles variable, but not using it yet; ZK, 2016.01.27

"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time # for measuring duration
from os import listdir

import make_Cmatrix as mcm
import template_fit as tf

def getFiles(findDir,toFind):
    """
    Note:
        mask files should have been created with make_ISW_map.makeMask
            and contain 'deg_' in the filename
    Args:
        findDir:
        toFind:

    Returns:

    """
    #findDir = '/Data/PSG/hundred_point/'
    findFiles = listdir(findDir)
    findFiles = [findDir+file for file in findFiles if 'deg_' in file and toFind in file]
    findFiles = np.sort(findFiles)
    return findFiles

def getMaskFiles(findDir):
    return getFiles(findDir,'mask_')
def getCMatFiles(findDir):
    return getFiles(findDir,'covar')
def getICMatFiles(findDir):
    return getFiles(findDir,'invCovar')



def makeMatrices(nested=False, maskDir = '/Data/PSG/hundred_point/'):
    """
    Purpose:
        This creates C and inverse C matrices for the list of masks
    Args:
        nested: the NESTED vs RING parameter for healpy functions
        maskDir: the directory to look for mask files and create matrices in
    Returns:
        nothing
    """

    useBigMask = False#True
    covDir = '/Data/covariance_matrices/'
    if useBigMask:
        bigCmatrixFile = covDir+'covar19917_ISWout_bws_hp12_RING.npy' #1.46 Gb file
        bigMaskFile = covDir+'ISWmask19117_RING.fits'
    else:
        bigCmatrixFile = covDir+'covar9875_ISWout_bws_hp12_RING.npy' #390.1 Mb file, should be good through ~10.5 degrees
        bigMaskFile = covDir+'ISWmask9875_RING.fits'
    print 'loading C matrix from file ',bigCmatrixFile
    bigCmatrix = mcm.symLoad(bigCmatrixFile)
    bigMask = hp.read_map(bigMaskFile,nest=nested)

    #maskDir = '/Data/PSG/hundred_point/'
    strLength = maskDir.__len__()+"ISWmask_".__len__() # masks created by make_ISW_map.makeMasks
    maskFiles  = getMaskFiles(maskDir)
    cMatFiles  = getCMatFiles(maskDir)
    iCMatFiles = getICMatFiles(maskDir)
    for maskFile in maskFiles:
        mask = hp.read_map(maskFile,nest=nested)
        cMatFile = 'covar_'+maskFile[strLength+1:-5]+'.npy' #starting after "ISWmask_", remove ".fits"
        iCMatFile = 'invCovar_'+maskFile[strLength+1:-5]+'.npy' #starting after "ISWmask_", remove ".fits"
        startTime = time.time()
        if cMatFile in cMatFiles:
            print cMatFile,' found'
        else:
            print 'creating C matrix for mask ',maskFile
            cMatrix = mcm.subMatrix2(mask,bigMask,bigCmatrix,nested=nested)
            mcm.symSave(cMatrix,maskDir+cMatFile)
        if iCMatFile in iCMatFiles:
            print iCMatFile,' found'
        else:
            print 'creating inverse C matrix for mask ',maskFile
            if cMatFile in cMatFiles:
                cMatrix = mcm.symLoad(maskDir+cMatFile)
            invCmat = mcm.choInvert(cMatrix)
            np.save(maskDir+iCMatFile,invCmat)
        print 'time elapsed for matrices creation: ',(time.time()-startTime)/60.,' minutes'



################################################################################
# testing code

def test(nested=False,ISWnum=1,CMBnum=1):
    """
    Purpose:
        This function loads a list of mask files, each created with a different
        size aperture around the GNS coordinates.  It then does a template
        fit on the specified ISW and CMB files using each of these masks
        and their associated inverse covariance matrices.  No mode filtering
        has been done on the files specified and none is done in this program.
    Args:
        nested: the NESTED vs RING parameter for healpy functions
        ISWnum: selects which ISW file set
        CMBnum: selects which CMB file set
    Returns:

    """
    maskDir = '/Data/PSG/hundred_point/'
    #makeMatrices(nested=nested,maskDir=maskDir)

    # get inverse covariance matrix filenames
    iCMatFiles = getICMatFiles(maskDir)
    maskFiles = getMaskFiles(maskDir)

    # get CMBmap, ISWmap
    CMBFiles,ISWFiles,maskFile,cMatrixFile,iCMatFile,starMaskFile = tf.getFilenames(doHighPass=True,maskNum=1)
        #CMBFiles[0]: mask with anafast; CMBFiles[1]: no mask with anafast
        #ISWFiles[0]: R010 r10%; ISWFiles[1]: R010 r02%;
        #ISWFiles[2]: PSGplot060 r02%; ISWFiles[3]: R060 r02%;
        #ISWFiles[4]: PSGplot100 r02%; ISWFiles[5]: R100 r02%;
    CMBmap = hp.read_map(CMBFiles[CNBnum],nest=nested) *1e-6 # convert microK to K
    ISWmap = hp.read_map(ISWFiles[ISWnum],nest=nested)

    nFiles = iCMatFiles.__len__()
    amps=np.zeros(nFiles)
    vars=np.zeros(nFiles)
    for fileNum,iCMatFile in enumerate(iCMatFiles):
        print 'opening file ',fileNum+1,' of ',nFiles,': ',iCMatFile
        invCmat = np.load(iCMatFile)
        mask = hp.read_map(maskFiles[fileNum])
        CMBvec = CMBmap[np.where(mask)]
        ISWvec = ISWmap[np.where(mask)]
        amp,var = tf.templateFit(invCmat,ISWvec,CMBvec)
        amps[fileNum]=amp
        vars[fileNum]=var
        print 'amp: ',amp,'+-',np.sqrt(var),' (',amp/np.sqrt(var),'sigma )'

    print 'done'





if __name__=='__main__':
  test()
