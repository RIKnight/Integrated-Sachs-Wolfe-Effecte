#! /usr/bin/env python
"""
  NAME:
    template_fit.py
  PURPOSE:
    Program to calculate amplitude of template on CMB
  USES:
    make_Cmatrix.py
    fits file containing mask indicating the set of pixels to use

  MODIFICATION HISTORY:
    Written by Z Knight, 2015.09.24
    Fixed Kelvin vs. microKelvin problem; ZK, 2015.09.29
    Added fit verification testing; ZK, 2015.10.04
    Added option for cMatrix units in microK**2; ZK, 2015.12.11
    Added useInverse flag and cInvT function; ZK, 2015.12.15
    Added K-S test for sim amplitudes vs. normal dist.; ZK, 2016.01.05
    Moved test data from test function to getTestData function; ZK, 2016.01.06
    Fixed nested vs ring misconception in hp.read_map; ZK, 2016.01.06
    Added nested as a passable parameter to 2 functions; ZK, 2016.01.09
    Broke getTestData apart into smaller pieces; ZK, 2016.01.20
    Modified ISW file names; ZK, 2016.01.21

"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time # for measuring duration
from os import listdir
from scipy import stats # for K-S test

import make_Cmatrix as mcm

def templateFit(cMatInv,ISW,CMB):
  """
    calculates amplitude and variance of amplitude of template
    INPUTS:
      cMatInv: (numpy array) inverse of the covariance matrix in Kelvin**-2
        where covariance matrix was calculated from C_l in Kelvin**2
      ISW: (numpy vector) the ISW template in Kelvin
      CMB: (numpy vector) the observed CMB in Kelvin
    Alternately:
      all Kelvin units can be replaced by microKelvin
    RETURNS:
      amp,var: the amplitude and variance of the fit
  """
  cInvISW = np.dot(cMatInv,ISW)
  var = (np.dot(ISW,cInvISW) )**(-1)
  amp = (np.dot(CMB,cInvISW) )*var
  return amp,var

def templateFit2(cMatrix,ISW,CMB):
  """
  IMPORTANT!!! this function is BS.  Don't use it.
  Purpose: same as templateFit but avoids matrix inversion by using vector inversion
  Args:
      cMatrix: a numpy array containing the covariance matrix (in K**2)
      ISW: a numpy vector containing the ISW template (in K)
      CMB: a numpy vector containing the observed CMB (in K)
  note:
      all Kelvin units can be replaced by microKelvin
  Uses:
      cInvT function
  Returns:
      amp,var: the amplitude and variance of the fit
  """
  cInvISW = cInvT(cMatrix,ISW)
  var = (np.dot(ISW,cInvISW) )**(-1)
  amp = (np.dot(CMB,cInvISW) )*var
  return amp,var

def cInvT(covMat,Tvec):
    """
    IMPORTANT!!! this function is BS.  Don't use it.
    Purpose:
        calculates C**-1*T using "left inverse" of T (a row vector)
    Args:
        covMat: numpy array containing a covariance matrix of field statistics
        Tvec: numpy array containing a column vector of field values
    Note:
        this function is copied in chisquare_test.py
    Returns:
        numpy array of C**-1*T (a column vector)
    """
    TInv = Tvec.T/np.dot(Tvec,Tvec)    # create left inverse of Tvec
    TInvC = np.dot(TInv,covMat)        # left multiply
    return TInvC.T/np.dot(TInvC,TInvC) # return right inverse

def KSnorm(rvs,loc,sigma,nBins=10,showPDF=False,showCDF=False):
  """

  Args:
      rvs: random variables to compare to normal distribution
      loc: the center of the normal distribution
      sigma: the standard deviation of the normal distribution
      nBins: the number of bins to use for PDF visual comparison. Does not affect KS test results
      showPDF: set this to show a PDF of the KS comparison
      showCDF: set this to show a CDF of the KS comparison
  Uses:
      scipy.stats.kstest, scipy.stats.norm
  Returns:
      the result of the KS test

  """
  KSresult = stats.kstest(rvs,'norm',args=(loc,sigma))
  if showPDF:
    h=plt.hist(rvs,bins=nBins,normed=True)
    x=np.linspace(stats.norm.ppf(0.01,loc=loc,scale=sigma),stats.norm.ppf(0.99,loc=loc,scale=sigma),100)
    plt.plot(x,stats.norm.pdf(x,loc=loc,scale=sigma))
    plt.title('K-S PDF at amplitude '+str(loc)+'; K-S statistic = '+str(KSresult[0]))
    plt.show()
  if showCDF:
    h=plt.hist(rvs,bins=rvs.__len__()*10,normed=True,cumulative=True,histtype='step')
    x=np.linspace(stats.norm.ppf(0.01,loc=loc,scale=sigma),stats.norm.ppf(0.99,loc=loc,scale=sigma),100)
    plt.plot(x,stats.norm.cdf(x,loc=loc,scale=sigma))
    plt.title('K-S CDF at amplitude '+str(loc)+'; K-S statistic = '+str(KSresult[0]))
    plt.show()
  return KSresult

def getFilenames(doHighPass=True, useBigMask=False):
  """
  Purpose:
      Load filenames into variables
  Note:
      CMBFiles have unit microK, ISWFiles have unit K, and cMatrixFile has units K**2 or microK**2
  Args:
      doHighPass: set this to use the highpass filtered CMB, ISW, cMatrix, and invCMat files
      useBigMask: there are two mask sizes specified in this function.
        Set this to use the large size, otherwise, the small one will be used.
  Returns:
      CMBFiles,ISWFiles,maskFile,cMatrixFile,iCMatFile
        CMBFiles[0]: mask with anafast; CMBFiles[1]: no mask with anafast
        ISWFiles[0]: r10%; ISWFiles[1]: r02%; ISWRiles[2]: PSGplot r02%
        CMBFiles have unit microK, ISWFiles have unit K, and cMatrixFile has units K**2

  """
  PSG = '/Data/PSG/'
  if doHighPass:
    ISWFiles = np.array([PSG+'hundred_point/ISWmap_RING_r10_R010_hp12.fits',  #radius to 10% max (ring)
                         PSG+'hundred_point/ISWmap_RING_R010_hp12.fits',      #radius to  2% max (ring)
                         PSG+'hundred_point/ISWmap_RING_PSGplot_hp12.fits'])  # from PSG fig.1 overmass
    CMBFiles = np.array([PSG+'planck_filtered.fits',             # used mask with anafast (ring)
                         PSG+'planck_filtered_nomask.fits'])     # no mask with anafast   (ring)
  else:
    ISWFiles = np.array([PSG+'hundred_point/ISWmap_RING_r10_R010_nhp.fits',   #radius to 10% max (ring)
                         PSG+'hundred_point/ISWmap_RING_R010_nhp.fits',       #radius to  2% max (ring)
                         PSG+'hundred_point/ISWmap_RING_PSGplot_nhp.fits'])   # from PSG fig.1 overmass
    CMBFiles = np.array([PSG+'planck_filtered_nhp.fits',         # used mask with anafast (ring)
                         PSG+'planck_filtered_nomask_nhp.fits']) # no mask with anafast   (ring)
  covDir = '/Data/covariance_matrices/'
  if useBigMask:
    maskFile = covDir+'ISWmask9875_RING.fits'
    if doHighPass:
      cMatrixFile = covDir+'covar9875_ISWout_bws_hp12_RING.npy'
      iCMatFile = covDir+'invCovar9875_cho_hp12_RING.npy'
    else:
      cMatrixFile = covDir+'covar9875_ISWout_bws_nhp_RING.npy' #haven't made this yet
      iCMatFile = covDir+'invCovar9875_cho_nhp_RING.npy'       #haven't made this yet
  else:
    maskFile = covDir+'ISWmask6110_RING.fits'
    if doHighPass:
      cMatrixFile = covDir+'covar6110_ISWout_bws_hp12_RING.npy'
      iCMatFile = covDir+'invCovar6110_cho_hp12_RING.npy'
    else:
      cMatrixFile = covDir+'covar6110_ISWout_bws_nhp_RING.npy' #haven't made this yet
      iCMatFile = covDir+'invCovar6110_cho_nhp_RING.npy'       #haven't made this yet
  return CMBFiles,ISWFiles,maskFile,cMatrixFile,iCMatFile

def getInverse(cMatrixFile,iCMatFile,type=3,newInverse=True,noSave=False):
  """
  Purpose:
      get the inverse of the covariance matrix
  Args:
      cMatrixFile: the filename storing a numpy covariance matrix
      iCMatFile: the filename to store a new inverse matrix to or load an existing matrix from
      type: the type of inversion to do
        1: LU inverse (lower,upper triangular decomposition)
        2: RD inverse (eigen decomposition)
        3: Cho inverse (Cholesky decomposition)
      newInverse: set this to calculate a new inverse matrix
        otherwise, no matrix will be inverted and one will be loaded from file
      noSave: set this to omit saving the new inverse C matrix (does nothing if newInverse=False)
        Default: True (new matrix will be saved to file iCMatFile)
  Returns:
      the inverse of the C matrix
  """
  if newInverse:
    print 'loading C matrix from file ',cMatrixFile  #may want to move this outside this function
    cMatrix = mcm.symLoad(cMatrixFile)
    # 2015.12.11: new matrices may have units microK**2 or K**2. Older matrices are all in K**2

    startTime = time.time()
    if   type == 1: # use LU
      print 'starting matrix (LU) inversion...'
      cMatInv = np.linalg.inv(cMatrix)
    elif type == 2: # use RD
      print 'calculating eigen decomposition...'
      w,v = np.linalg.eigh(cMatrix)
      print 'starting matrix (eigen) inversion...'
      cMatInv = mcm.RDinvert(w,v)
    elif type == 3: # use Cho
      print 'starting matrix (Cholesky) inversion...'
      cMatInv = mcm.choInvert(cMatrix)
    else:
      print 'no type ',type
      return 0
    print 'time elapsed for inversion: ',(time.time()-startTime)/60.,' minutes'
      #took about 2 minutes for np.linalg.inv on 6110**2 matrix
    if not noSave:
      np.save(iCMatFile,cMatInv)
  else:
    print 'loading inverse C matrix from file ',iCMatFile
    cMatInv = np.load(iCMatFile)
  return cMatInv


def getTestData(doHighPass=True, useBigMask=False, newInverse=False, matUnitMicro=False,
                useInverse=True, nested=False):
  """
  Purpose:
      get the (covariance matrix or inverse covariance matrix), mask, ISW vectors, and model variances
      The program also does some template fitting for two test CMB files and two test ISW files.
        (This last functionality doesn't really belong here and should be moved back to test function)
  Args:
      doHighPass: set this to use files that have been high pass filtered
      useBigMask: set this to use the 9000 pixel mask. Otherwise, the 6000 pixel mask is used
      newInverse: set this to create a new inverse if useInverse is also flagged
      matUnitMicro: set this if cMatrix or cMatInv has units of microKelvin**2.  Otherwise, K**2 is assumed.
      useInverse: set this to invert cMatrix.  Otherwise, left and right inverses are used via cInvT function.
        If used, inverse covariance matrix is returned.  Otherwise, covariance matrix is returned.
        Default: True.  please don't change to False.  The cInvT method is garbage.
      nested: the NESTED vs RING parameter to pass to healpy functions

  Returns:
      matrix,mask,ISWvecs,modelVariances,ISWFiles,CMBFiles
        matrix: a numpy array containing the covariance matrix or the inverse covariance matrix,
          depending on the value of useInverse
        mask: a numpy vector containing the mask
        ISWvecs: a numpy array containing a vector for each ISW file specified in the code below.
        modelVariances: the template fitting variance for each ISW file specified below.
        ISWFiles: list of ISW files
        CMBFiles: list of CMB files
  """
  # file names
  CMBFiles,ISWFiles,maskFile,cMatrixFile,iCMatFile = getFilenames(doHighPass=doHighPass,useBigMask=useBigMask)

  if useInverse:
    cMatInv = getInverse(cMatrixFile,iCMatFile,type=3,newInverse=newInverse,noSave=False)
  else: # do not invert and use cInvT
    print 'loading C matrix from file ',cMatrixFile
    cMatrix = mcm.symLoad(cMatrixFile)
    # 2015.12.11: new matrices may have units microK**2 or K**2. Older matrices are all in K**2

  # load the mask
  mask = hp.read_map(maskFile,nest=nested)

  modelVariances = np.empty(ISWFiles.size) # to store the variance expected in model
  maskSize = np.sum(mask)
  ISWvecs = np.zeros((ISWFiles.size,maskSize))
  CMBvecs = np.zeros((CMBFiles.size,maskSize))
  for iIndex,ISWfile in enumerate(ISWFiles):
    ISW = hp.read_map(ISWfile,nest=nested)
    ISW = ISW[np.where(mask)]
    for cIndex,CMBfile in enumerate(CMBFiles):
      print 'starting with ',ISWfile,' and ',CMBfile
      CMB = hp.read_map(CMBfile,nest=nested)
      CMB = CMB[np.where(mask)]

      if useInverse:
        if matUnitMicro:
          amp,var = templateFit(cMatInv,ISW*1e6,CMB) # ISW from K to microK
        else:
          amp,var = templateFit(cMatInv,ISW,CMB*1e-6) # CMB from microK to K
      else: # use cInvT  ... c'mon really? cInvT will swallow your soul.
        if matUnitMicro:
          amp,var = templateFit2(cMatrix,ISW*1e6,CMB) # ISW from K to microK
        else:
          amp,var = templateFit2(cMatrix,ISW,CMB*1e-6) # CMB from microK to K
      print 'amplitude: ',amp,', variance: ',var
      CMBvecs[cIndex] = CMB
    ISWvecs[iIndex] = ISW
    modelVariances[iIndex] = var # independant of CMB map

  if useInverse:
    return cMatInv,mask,ISWvecs,modelVariances,ISWFiles,CMBFiles
  else:
    return cMatrix,mask,ISWvecs,modelVariances,ISWFiles,CMBFiles



################################################################################
# testing code

def test(useInverse=True, nested=False):
  """
    Purpose: test the template fitting procedure
    Input:
      useInverse:  set this to True to invert the C matrix,
        False to use vector inverses via cInvT
        Default: True. please don't change to False.  The cInvT method is garbage.
      nested: the NESTED vs RING parameter to pass to healpy functions
    Returns: nothing
  """

  doHighPass   = True  # having this false lets more cosmic variance in
  useBigMask   = False
  newInverse   = False
  matUnitMicro = False # option for matrices newer than 2015.12.11

  # this line gets the test data and does 4 template fits for files specified within
  print 'Starting template fitting on observed data... '
  M,mask,ISWvecs,modelVariances,ISWFiles,CMBFiles = getTestData(doHighPass=doHighPass,useBigMask=useBigMask,
                                                                newInverse=newInverse,matUnitMicro=matUnitMicro,
                                                                useInverse=useInverse,nested=nested)
  if useInverse:
    cMatInv = M
  else:
    cMatrix = M
  del M

  # the rest of this file has been outdated.  Use KS-showdown instead
  """

  # testing for verification of template fitting method
  # all of the maps have been created with highpass filtering and beamsmoothing
  # each of 10 random realizations of C_l(ISWout) is added to each of 10
  # amplitudes times the ISW map... see make_sims.py

  newFit = True
  print 'Starting verification testing... '

  # collect filenames
  if doHighPass:
    simDirectory = '/shared/Data/sims/highpass/'
  else:
    simDirectory = '/shared/Data/sims/'
  simFiles = listdir(simDirectory)
  CMBFiles = [simDirectory+file for file in simFiles if '.fits' in file] #'b120_N64' in file]
  CMBFiles = np.sort(CMBFiles)

  # just do one file for now
  #CMBFiles = [file for file in CMBFiles if '01a' in file]
  
  # nested vs ring parameter for loading data
  #nested = False

  # load the mask
  #mask = hp.read_map(maskFile,nest=nested) #mask file has nested ordering


  if newFit:
    # array for storing results
    results = np.zeros([len(ISWFiles),len(CMBFiles),2]) # 2 for amp,var
  
    # it took about 2 minutes to do 110 files
    for iIndex,ISWfile in enumerate(ISWFiles):
      ISW = hp.read_map(ISWfile,nest=nested)
      ISW = ISW[np.where(mask)]
      for cIndex,CMBfile in enumerate(CMBFiles):
        print 'starting with ',ISWfile,' and ',CMBfile
        CMB = hp.read_map(CMBfile,nest=nested)
        CMB = CMB[np.where(mask)]

        if useInverse:
          if matUnitMicro:
            amp,var = templateFit(cMatInv,ISW*1e6,CMB*1e6) #ISW and CMB from K to microK
          else:
            amp,var = templateFit(cMatInv,ISW,CMB)
        else: # use cInvT... beware!  cInvT will destroy your career and melt your brain
          if matUnitMicro:
            amp,var = templateFit2(cMatrix,ISW*1e6,CMB*1e6) #ISW and CMB from K to microK
          else:
            amp,var = templateFit2(cMatrix,ISW,CMB)
        print 'amplitude: ',amp,', variance: ',var

        results[iIndex,cIndex,:] = [amp,var]

    np.save('verification_results',results)
  else:
    results = np.load('verification_results.npy')
    #results = np.load('verification_results.noBSisw.npy')
    #results = np.load('verification_results.cosVariance.npy')
  
  nISW = 2
  nAmps = 10
  nRealizations = 10
  actualAmps = np.logspace(-1,2,nAmps) # must match values in make_sims.py
  actualAmps = np.append([0],actualAmps) # extra [0]
  ampTags = ['0','a','b','c','d','e','f','g','h','i','j'] # extra '0'
  realizationNum = np.arange(nRealizations)
  ampNum = np.arange(nAmps+1)
  indices = np.arange(len(CMBFiles))

  # reshape the results
  reshResults = np.reshape(results,[2,10,11,2]) #10 realizations, 11 amplitudes


  # plot by amplitudes
  for actInd,actual in enumerate(actualAmps):
    for iswNum in [1]: #range(nISW):
      plt.figure()
      sigma = np.sqrt(reshResults[iswNum,:,actInd,1])
      plt.errorbar(realizationNum,reshResults[iswNum,:,actInd,0],yerr=sigma,fmt='o')
      actualAmp = np.ones(10)*actual
      plt.plot(realizationNum,actualAmp)
      plt.xlabel('10 realizations of ISWout CMB')
      plt.ylabel('amplitude derived from simulation')
      plt.title('trying to recover amplitude = '+str(actual)+'with ISW'+str(iswNum+1))
      plt.show()
  
  # plot by realizations
  for realInd,realization in enumerate(realizationNum):
    for iswNum in [1]: #range(nISW):
      plt.figure()
      sigma = np.sqrt(reshResults[iswNum,realization,:,1])
      plt.errorbar(ampNum,reshResults[iswNum,realization,:,0],yerr=sigma,fmt='o')
      plt.plot(ampNum,actualAmps)
      plt.xlabel('11 amplitudes of ISW template')
      plt.ylabel('amplitude derived from simulation')
      plt.title('template fitting with CMB realization #'+str(realInd))
      plt.show()

  # plot mean with errors
  for iswNum in [1]: #range(nISW):
    plt.figure()
    amplitudes = reshResults[iswNum,:,:,0]
    means = np.mean(amplitudes,axis = 0)
    stds = np.std(amplitudes,axis=0)
    plt.errorbar(ampNum,means,yerr=stds,fmt='o')
    plt.plot(ampNum,actualAmps)
    plt.xlabel('amplitude number of ISW template')
    plt.ylabel('mean amplitude of 10 simulated CMBs')
    if doHighPass:
      plt.title('template fitting compared to actual amplitudes; no l < 12')
    else:
      plt.title('template fitting compared to actual amplitudes; no highpass')
    plt.show()


  # do K-S test of simulated amplitudes against expected normal distribution
  for actInd,actual in enumerate(actualAmps):
    for iswNum in [1]:
      myAmp = reshResults[iswNum,:,actInd,0]
      # this is the wrong variance fix this.  Or just use KS_showdown instead.
      sigma = np.sqrt(modelVariances[iswNum])
      KSresult = KSnorm(myAmp,actual,sigma,nBins=5,showCDF=True)
      print 'K-S test result for simulated amplitude '+str(actual)+': ',KSresult

  """

if __name__=='__main__':
  test()

