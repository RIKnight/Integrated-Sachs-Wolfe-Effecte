#! /usr/bin/env python
"""
  NAME:
    make_Cmatrix.py
  PURPOSE:
    Program to create a covariance matrix for a given set of HEALpix pixels.
  USES:
    fits file containing mask indicating the set of pixels to use
    fits file containing ISWout power spectrum
      where ISWout indicates no ISW for 0.4 < z < 0.75
  MODIFICATION HISTORY:
    Written by Z Knight, 2015.09.14
    Added subMatrix; ZK, 2015.09.21
    Added symSave, symLoad; ZK, 2015.09.29
    Added RDInvert; ZK, 2015.10.29
    Switched pyfits to astropy.io.fits; ZK, 2015.11.09
    Switched showCl scaling to l(l+1) from l(l-1); ZK, 2015.11.13
    Added beamSmooth, pixWin parameters to makeCmatrix;
      changed highpass parameter default to 0; ZK, 2015.12.04
    Added cMatCorrect function; ZK, 2015.12.08
    Added choInvert function; ZK, 2015.12.09

"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
#import pyfits as pf
import astropy.io.fits as pf
from numpy.polynomial.legendre import legval
from scipy.special import legendre
import time # for measuring duration

def symSave(saveMe,saveFile='symsave'):
  """
    Turns symmetric matrix into a vector and saves it to disk.
    INPUTS:
      saveMe: the symmetric numpy 2d array to save
      saveFile: the file name to save.  .npy will be appended if not already present
        default: symsave.npy
  """
  indices = np.triu_indices_from(saveMe)
  np.save(saveFile,saveMe[indices])

def symLoad(loadFile='symsave.npy'):
  """
    loads a numpy .npy array from file and transforms it into a symmetric array
    INPUTS:
      loadFile: the name of the file to load
        default: symsave.npy
    Returns a symmetric 2d numpy array
  """
  loaded = np.load(loadFile)
  n = -1/2.+np.sqrt(1/4.+2*loaded.size)
  array = np.zeros([n,n])
  indices = np.triu_indices(n)
  array[indices]=loaded
  return array+np.transpose(array)-np.diag(np.diag(array))


def getCl(filename):
  """
    opens a CAMB FITS file and extracts the Power spectrum
    filename: the name of the FITS file to open
    returns two arrays: one of ell, one of C_l
  """
  powSpec = pf.getdata(filename,1)
  temps = powSpec.field('TEMPERATURE')
  ell = np.arange(temps.size)
  return ell,temps

def showCl(ell,temps,title='CAMB ISWout power spectrum'):
  """
    create a plot of power spectrum
    ell: the multipole number
    temps: the temperature power in Kelvin**2
    title : the title for the plot
    uses ell*(ell+1)/2pi scaling on vertical axis
  """
  plt.plot(ell,temps*ell*(ell+1)/(2*np.pi) *1e12) #1e12 to convert to microK**2
  plt.xlabel('multipole moment l')
  plt.ylabel('l(l+1)C_l/(2pi) [microK**2]')
  plt.title(title)
  plt.show()

def makeLegendreTable(lmax,saveFile='legtab'):
  """
    function to create a table of legendre polynomial coefficients and save it to file
    table will be a square array
    columns correspond to powers of x starting with x^0 in the left most column
    INPUTS:
      lmax: the highest l value to be calculated
      saveFile: the name of the numpy save file.  If not already present, '.npy' will
        be appended.
        Default: legtab.npy

  # this works fine for small lmax values, but ell=86 and higher have problems
  #   possibly due to exceeding the maximum size of a float64 dtype
  """
  coefs = np.zeros((lmax+1,lmax+1))
  for ell in range(lmax+1):
    cs = legendre(ell)
    coefs[ell,:ell+1] = cs.coeffs[::-1]
  symSave(coefs,saveFile)

def powerArray(x,powMax):
  """
    function to create an array of powers of x, starting from x^0
    returns a powMax+1 length numpy array
  """
  pows = np.array([1])
  for power in range(1,powMax):
    pows = np.concatenate((pows,np.array([pows[power-1]*x])))
  return pows


def makeCmatrix(maskFile, powerFile, highpass = 0, beamSmooth = True, pixWin = True):
  """
    function to make the covariance matrix
    maskFile is a healpix fits file name that contains 1 where a pixel is to be included
      in covariance matrix and 0 otherwise
      Must be Nested and NSIDE=64
    powerFile is a CAMB CMB power spectrum file
    highpass is the lowest multipole l to not be zeroed out.
      Default is 0
    beamSmooth determines whether to use beamsmoothing on C_l,
      also control lmax
      Default is True, with lmax = 250
    pixWin determines whether to use the pixel window,
      also controls lmax
      Default is True, with lmax = 250
    If both beamSmooth and pixWin are false, then lmax = 2000
    returns the covariance matrix
  """
  # read mask file
  mask = hp.read_map(maskFile,nest=True)

  # read power spectrum file
  ell,C_l = getCl(powerFile)

  # read coordinates file
  coordsFile64 = '/shared/Data/pixel_coords_map_nested_galactic_res6.fits'
  gl,gb = hp.read_map(coordsFile64,(0,1),nest=True)

  # isolate pixels indicated by mask
  #myGl = np.array([gl[index] for index in range(gl.size) if mask[index] == 1])
  #myGb = np.array([gb[index] for index in range(gb.size) if mask[index] == 1])
  myGl = gl[np.where(mask)]
  myGb = gb[np.where(mask)]
  #print 'mask size: ',myGl.size,' or ',myGb.size

  # convert to unit vectors
  unitVectors = hp.rotator.dir2vec(myGl,myGb,lonlat=True)
  print unitVectors.shape
  
  # create half (symmetric) matrix of cosine of angular separations
  # this takes about 67 seconds for 6110 point mask
  vecSize = myGl.size
  cosThetaArray = np.zeros([vecSize,vecSize])
  for row in range(vecSize): #or should this be called the column?
    cosThetaArray[row,row] = 1.0 # the diagonal
    for column in range(row+1,vecSize):
      cosThetaArray[row,column] = np.dot(unitVectors[:,row],unitVectors[:,column])
      #if cosThetaArray[row,column] != cosThetaArray[row,column]:
      #  print 'NaN at row: ',row,', column: ',column

  print cosThetaArray
  
  # create beam and pixel window expansions and other factor
  lmax = 2000
  print "lmax cutoff imposed at l=",lmax
  if pixWin:
    lmax = 250
    Wpix = hp.pixwin(64)
    W_l = Wpix[:lmax+1]
  else:
    W_l = 1.0
  if beamSmooth:
    lmax = 250
    forSMICA = False # if using C matrix with SMICA map, set to True.  if using with sims, set to False
    mbeam = hp.gauss_beam(120./60*np.pi/180,lmax=lmax) # 120 arcmin to be below W_l
    if forSMICA:
        pbeam = hp.gauss_beam(5./60*np.pi/180,lmax=lmax)   # 5 arcmin beam; SMICA already has
        B_l = mbeam/pbeam
    else:
        B_l = mbeam
  else:
    B_l = 1.0
  fac_l = (2*ell[:lmax+1]+1)/(4*np.pi)
  C_l = np.concatenate((np.zeros(highpass),C_l[highpass:]))
  preFac_l = fac_l *B_l**2 *W_l**2 *C_l[:lmax+1]

  # should I convert C_l units from K to \mu K?

  # evaluate legendre series with legval
  covArray = np.zeros([vecSize,vecSize])
  for row in range(vecSize):
    print 'starting row ',row
    for column in range(row,vecSize):
      covArray[row,column] = legval(cosThetaArray[row,column],preFac_l)
    for column in range(row-1):
      covArray[row,column] = covArray[column,row]

  return covArray

def cMatCorrect(cMatrix):
    """
    Purpose:
        corrects C matrix for effects of estimating the mean of the sample from the sample iteslf
        Follows Granett, Nerynck, and Szapudi 2009, section 4.1
    Args:
        cMatrix:  a numpy array containing the covariance matrix to correct

    Returns:
        a numpy array containing the corrected C matrix
    """
    size = cMatrix.shape[0] #array is square
    c1 = np.sum(cMatrix,axis=0) #adds column of printed matrix, row should remain
    c2 = np.sum(cMatrix,axis=1) #adds row of printed matrix, column should remain
    c3 = np.sum(cMatrix)        #adds entire matrix, is a number
    # c1 is ok for subtraction via broadcasting rules; c2 needs modification
    c2mat = np.reshape(np.repeat(c2,size),(size,size))

    cMatrixCorrected = cMatrix - c1/size - c2mat/size + c3/size**2
    return cMatrixCorrected


def subMatrix(maskFile,bigMaskFile,cMatrixFile):
  """
    function to extract a C matrix from a matrix made for a larger set of pixels.
    
    (please modify this to accept a list of maskFiles and return a list of C matrices)

    INPUTS:
      maskFile: FITS file containg a mask indicating which pixels to use
      bigMaskFile: FITS file containing a mask that corresponds to the pixels
        used to create the cMatrix stored in cMatrixFile
      cMatrixFile: numpy file containing a symSave C matrix
    OUTPUTS:
      returns a C matrix
  """
  # read mask files
  mask = hp.read_map(maskFile,nest=True)
  bigMask = hp.read_map(bigMaskFile,nest=True)
  # read C matrix file
  print 'loading C matrix from file...'
  cMatrix = symLoad(cMatrixFile)

  maskVec = np.where(mask)[0] #array of indices
  bigMaskVec = np.where(bigMask)[0] #array of indices
  print 'looping through indices to create sub-matrix...'
  subVec = [bigI for bigI in range(bigMaskVec.size) for subI in range(maskVec.size) if maskVec[subI] == bigMaskVec[bigI] ]
  print 'done'

  subCmat = cMatrix[np.meshgrid(subVec,subVec)]  #.transpose() # transpose not needed for symmetric
  
  return subCmat

def RDInvert(eigVals, eigVecs):
  """
    function to calculate symmetric inverse of Covariance matrix using
      eigen decomposition of covariance matrix
    eigVals: 
      numpy array of eigenvalues of C matrix
    eigVecs:
      numpy array of eigenvectors of C matrix
    returns:
      the inverse of the Covariance matrix

  """
  diagInv = np.diag(eigVals**-1)
  eigVecsTranspose = np.transpose(eigVecs)
  cInv = np.dot(np.dot(eigVecs,diagInv),eigVecsTranspose)
  return cInv

def choInvert(cMatrix):
  """
  Purpose:
      Function to find inverse of symmetric matrix using Cholesky decomposition
  Args:
      cMatrix: the matrix to invert

  Returns:
      The inverse of cMatrix
  """
  L = np.linalg.cholesky(cMatrix) # defined for real symmetric matrix: cMatrix = L * L.T
  Linv = np.linalg.inv(L)  # uses LU decomposition, but L is already L
  cMatInv = np.dot(Linv.T,Linv)
  return cMatInv


def test():
  """
    code for testing the other functions in this module
  """
  # test getCl
  ISWoutFile = 'ISWout_scalCls.fits'
  ell,temps = getCl(ISWoutFile)

  """
  # test showCl
  showCl(ell,temps)

  # test makeLegendreTable
  # this works fine for small lmax values, but ell=86 and higher have problems
  #   possibly due to exceeding the maximum size of a float64 dtype
  makeLegendreTable(9,'testTable.npy')
  table = symLoad('testTable.npy')
  print table

  # test powerArray
  powers = powerArray(2,9)
  print powers

  # test makeCmatrix
  # measured time: 4.25 hrs for 6110 point mask
  startTime = time.time()
  #maskFile = '/shared/Data/PSG/ten_point/ISWmask_din1_R010.fits'
  #saveMatrixFile = 'covar6110_R010_lowl.npy'
  #saveMatrixFile = 'covar6110_R010.npy'
  maskFile = '/shared/Data/PSG/hundred_point/ISWmask2_din1_R160.fits'
  saveMatrixFile = 'covar9875_R160b.npy'
  
  covMat = makeCmatrix(maskFile, ISWoutFile)
  print 'time elapsed: ',int((time.time()-startTime)/60),' minutes'
  symSave(covMat,saveMatrixFile)

  # test subMatrix
  subMask = '/shared/Data/PSG/small_masks/ISWmask_din1_R010_trunc0500.fits'
  subCmat = subMatrix(subMask,maskFile,saveMatrixFile)
  print 'time elapsed: ',int((time.time()-startTime)/60),' minutes'
  """

if __name__=='__main__':
  test()



