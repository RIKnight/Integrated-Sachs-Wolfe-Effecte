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

"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
#import pyfits as pf
#from numpy.polynomial.legendre import legval
#from scipy.special import legendre
import time # for measuring duration
from os import listdir

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
    Purpose:
        calculates C**-1*T using "left inverse" of T (a row vector)
    Args:
        covMat: numpy array containing a covariance matrix of field statistics
        Tvec: numpy array containing a column vector of field values
    Note:
        this function is copied int chisquare_test.py
    Returns:
        numpy array of C**-1*T (a column vector)
    """
    TInv = Tvec.T/np.dot(Tvec,Tvec)    # create left inverse of Tvec
    TInvC = np.dot(TInv,covMat)        # left multiply
    return TInvC.T/np.dot(TInvC,TInvC) # return right inverse

################################################################################
# testing code

def test(useInverse = False):
  """
    Purpose: test the template fitting procedure
    Input:
      useInverse:  set this to True to invert the C matrix,
        False to use vector inverses
        Default: False
    Returns: nothing
  """

  doHighPass = True
  useBigMask = False
  newInverse = False
  matUnitMicro = False # option for matrices newer than 2015.12.11
  
  # file names
  PSG = '/Data/PSG/'
  if doHighPass:
    # testing with 2 variations on ISW map and 2 variations on CMB map
    ISWFiles = np.array([PSG+'hundred_point/ISWmap_RING_r10_R010_hp11.fits',#radius to 10% max (ring)
                         PSG+'hundred_point/ISWmap_RING_R010_hp11.fits'])   #radius to  2% max (ring)
    CMBFiles = np.array([PSG+'planck_filtered.fits',         # used mask with anafast (ring)
                         PSG+'planck_filtered_nomask.fits']) # no mask with anafast   (ring)
  else:
    # testing with 2 variations on ISW map and 2 variations on CMB map
    ISWFiles = np.array([PSG+'hundred_point/ISWmap_RING_r10_R010.fits',   #radius to 10% max (ring)
                         PSG+'hundred_point/ISWmap_RING_R010.fits'])      #radius to  2% max (ring)
    CMBFiles = np.array([PSG+'planck_filtered_nhp.fits',         # used mask with anafast (ring)
                         PSG+'planck_filtered_nomask_nhp.fits']) # no mask with anafast   (ring)

  if useBigMask:
    maskFile = PSG+'hundred_point_bad/ISWmask2_din1_R160.fits' #(nested)
    if doHighPass:
      cMatrixFile = 'covar9875_R160b.npy'
      #iCMatFile = 'invCovar_R160.npy'
      iCMatFile = 'invCovar_R160_RD.npy'
    else:
      cMatrixFile = 'covar9875_R160b_nhp.npy' #haven't made this yet
      iCMatFile = 'invCovar_R160_nhp.npy' #haven't made this yet
  else:
    maskFile = PSG+'ten_point/ISWmask_din1_R010.fits' #(nested)
    if doHighPass:
      cMatrixFile = 'covar6110_R010.npy' 
      #iCMatFile = 'invCovar_R010.npy'
      iCMatFile = 'invCovar_R010_RD.npy'
    else:
      cMatrixFile = 'covar6110_R010_nhp.npy'
      iCMatFile = 'invCovar_R010_nhp.npy'
  # CMBFiles have unit microK, ISWFiles have unit K, and cMatrixFile has units K**2 or microK**2


  # nested vs ring parameter for loading data
  nested = False

  if useInverse:
    # invert CMatrix
    useRD = False#True # Overrides useCho
    useCho = True#False # Overrides default: use LU
    if newInverse:
      print 'loading C matrix from file ',cMatrixFile
      cMatrix = mcm.symLoad(cMatrixFile)
      # 2015.12.11: new matrices may have units microK**2 or K**2. Older matrices are all in K**2

      startTime = time.time()
      if useRD:
        print 'calculating eigen decomposition...'
        w,v = np.linalg.eigh(cMatrix)
        print 'starting matrix (eigen) inversion...'
        cMatInv = mcm.RDinvert(w,v)
      elif useCho:
        print 'starting matrix (Cholesky) inversion...'
        cMatInv = mcm.choInvert(cMatrix)
      else: # use LU
        print 'starting matrix (LU) inversion...'
        cMatInv = np.linalg.inv(cMatrix)
      print 'time elapsed for inversion: ',(time.time()-startTime)/60.,' minutes'
        #took about 2 minutes for np.linalg.inv on 6110**2 matrix
      np.save(iCMatFile,cMatInv)
    else:
      print 'loading inverse C matrix from file ',iCMatFile
      cMatInv = np.load(iCMatFile)
  else: # do not invert and use cInvT
    print 'loading C matrix from file ',cMatrixFile
    cMatrix = mcm.symLoad(cMatrixFile)
    # 2015.12.11: new matrices may have units microK**2 or K**2. Older matrices are all in K**2

  # load the mask - nest=True for mask!
  mask = hp.read_map(maskFile,nest=True)
  
  for ISWfile in ISWFiles:
    ISW = hp.read_map(ISWfile,nest=nested)
    ISW = ISW[np.where(mask)]
    for CMBfile in CMBFiles:
      print 'starting with ',ISWfile,' and ',CMBfile
      CMB = hp.read_map(CMBfile,nest=nested)
      CMB = CMB[np.where(mask)]

      if useInverse:
        if matUnitMicro:
          amp,var = templateFit(cMatInv,ISW*1e6,CMB) # ISW from K to microK
        else:
          amp,var = templateFit(cMatInv,ISW,CMB*1e-6) # CMB from microK to K
      else: # use cInvT
        if matUnitMicro:
          amp,var = templateFit2(cMatrix,ISW*1e6,CMB) # ISW from K to microK
        else:
          amp,var = templateFit2(cMatrix,ISW,CMB*1e-6) # CMB from microK to K
      print 'amplitude: ',amp,', variance: ',var

  # testing for verification of template fitting method
  # all of the maps have been created with highpass filtering and beamsmoothing
  # each of 10 random realizations of C_l(ISWout) is added to each of 10
  # amplitudes times the ISW map... see make_sims.py

  newFit = False#True
  print 'Starting verification testing... '

  # collect filenames
  if doHighPass:
    simDirectory = '/shared/Data/sims/highpass/'
  else:
    simDirectory = '/shared/Data/sims/'
  simFiles = listdir(simDirectory)
  CMBFiles = [simDirectory+file for file in simFiles if 'b120_N64' in file]
  CMBFiles = np.sort(CMBFiles)

  # just do one file for now
  #CMBFiles = [file for file in CMBFiles if '01a' in file]
  
  # nested vs ring parameter for loading data
  #nested = False

  # load the mask
  #mask = hp.read_map(maskFile,nest=True) #nested)  #mask must have nest=True


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
        else: # use cInvT
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
  

if __name__=='__main__':
  test()

