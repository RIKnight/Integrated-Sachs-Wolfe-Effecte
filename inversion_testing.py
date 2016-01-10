#! /usr/bin/env python
"""
  NAME:
    inversion_testing.py
  PURPOSE:
    Program to explore covariance matrix inversion
  USES:
    make_Cmatrix.py 
    fits file containing mask indicating the set of pixels to use
    fits file containing ISWout power spectrum
      where ISWout indicates no ISW for 0.4 < z < 0.75
      Default: ISWout_scalCl.fits
  MODIFICATION HISTORY:
    Written by Z Knight, 2015.09.19
    Added plotting, ZK, 2015.09.21
    Added more tests and plots, ZK, 2015.09.29
    Switched to make_Cmatrix.symSave file format; ZK, 2015.09.29
    Added nested parameter; ZK, 2016.01.09

"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import make_Cmatrix as mcm
import time

def checkEigs(maskFile,ISWoutFile,covMatFile,Load=False,subMatrix=False):
  """
    function to check the eigenvalues and eigenvectors of a covariance matrix
    designed to help look for invertibility problems
    INPUTS:
      maskFile: FITS file containing HEALpix mask
      ISWoutFile: FITS file containing CAMB power spectrum
      covMatFile: numpy file to save or load symmetric covariance matrix to or from
      Load: set this to true to load covMat from covMatFile
        default = False (create matrix and save in covMatFile)
      subMatrix: set this to create a C matrix as a sub-matrix
        default = False
    USES:
      make_Cmatrix
    RETURNS:
      eigenvectors, eigenvalues, inverse matrix (or error messages)

  """
  bigMaskFile = '/shared/Data/PSG/ten_point/ISWmask_din1_R010.fits'
  cMatrixFile = 'covar6110_R010.npy'

  if not Load:
    if not subMatrix:
      covMat = mcm.makeCmatrix(maskFile,ISWoutFile)
    else:
      covMat = mcm.subMatrix(maskFile,bigMaskFile,cMatrixFile)
    mcm.symSave(covMat,covMatFile)
  else:
    covMat = mcm.symLoad(covMatFile)

  w,v     = np.linalg.eig(covMat) #w=eigvals,v=eigvecs
  invCmat = np.linalg.inv(covMat)

  return w,v,invCmat

def plotEigs(maskFile,maskName,eigvals,eigvecs,nested=False):
  """
    function to compare eigenvectors dotted with data vectors (called RT)
      against sqrt(eigenvalues) 
    INPUTS:
      maskFile: FITS file containing HEALpix mask
      maskName: name of the mask for plot title
      eigvals: eigenvalues of C matrix
      eigvecs: eigenvectors of C matrix
      nested: the NESTED vs RING parameter to pass to healpy functions
    OUTPUTS:
      plots to pyplot window
  """
  # load maps
  mapFile  = '/shared/Data/PSG/ten_point/ISWmap_din1_R060.fits'
  ISWmap   = hp.read_map(mapFile,nest=nested)
  ISWmask = hp.read_map(maskFile,nest=nested)
  mapVec  = ISWmap[np.where(ISWmask)]
  RT = np.dot(eigvecs,mapVec)

  plt.plot(RT,np.sqrt(eigvals)*1e5,'b+')
  plt.xlabel('C(p1,p2) eigenvectors * ISWmap values')
  plt.ylabel('sqrt(C) eigenvalues [10^-5]')
  plt.title('eigenvalue comparison for '+maskName)
  plt.show()

"""

ISWoutFile = 'ISWout_scalCls.fits'

maskFile1 = '/shared/Data/sparsemask_1.fits'
maskFile2 = '/shared/Data/sparsemask_2.fits'
maskFile3 = '/shared/Data/sparsemask_3.fits'

maskFile4 = '/shared/Data/PSG/small_masks/ISWmask_din1_R060_trunc1.fits'
maskFile5 = '/shared/Data/PSG/small_masks/ISWmask_din1_R060_trunc2.fits'
maskFile6 = '/shared/Data/PSG/small_masks/ISWmask_din1_R060_trunc3.fits'

maskFile7 = '/shared/Data/PSG/small_masks/ISWmask_din1_R010_trunc0500.fits'
maskFile8 = '/shared/Data/PSG/small_masks/ISWmask_din1_R010_trunc1000.fits'
maskFile9 = '/shared/Data/PSG/small_masks/ISWmask_din1_R010_trunc1500.fits'

maskFile10 = '/shared/Data/PSG/small_masks/ISWmask_din1_R010_trunc2000.fits'
maskFile11 = '/shared/Data/PSG/small_masks/ISWmask_din1_R010_trunc2500.fits'
maskFile12 = '/shared/Data/PSG/small_masks/ISWmask_din1_R010_trunc3000.fits'

maskFile13 = '/shared/Data/PSG/ten_point/ISWmask_din1_R010.fits'

# With submatrix=True, it took 74 seconds for file7, 95 for file8, 126 for file9,
#   147 for file10, 206 for file11, 283 for file12
startTime = time.time()
#w1,v1,invCMat1 = checkEigs(maskFile1,ISWoutFile,'covar_sparse1.txt',Load=True)
#w2,v2,invCMat1 = checkEigs(maskFile2,ISWoutFile,'covar_sparse2.txt',Load=True)
#w3,v3,invCMat3 = checkEigs(maskFile3,ISWoutFile,'covar_sparse3.txt',Load=True)
#w4,v4,invCMat4 = checkEigs(maskFile4,ISWoutFile,'covar_sparse4.txt',Load=True)
#w5,v5,invCMat5 = checkEigs(maskFile5,ISWoutFile,'covar_sparse5.txt',Load=True)
#w6,v6,invCMat6 = checkEigs(maskFile6,ISWoutFile,'covar_sparse6.txt',Load=True)
#w7,v7,invCMat7 = checkEigs(maskFile7,ISWoutFile,'covar_sparse7.txt',Load=True)#subMatrix=True)
#w8,v8,invCMat8 = checkEigs(maskFile8,ISWoutFile,'covar_sparse8.txt',Load=True)#subMatrix=True)
#w9,v9,invCMat9 = checkEigs(maskFile9,ISWoutFile,'covar_sparse9.txt',Load=True)#subMatrix=True)
#w10,v10,invCMat10 = checkEigs(maskFile10,ISWoutFile,'covar_sparse10.txt',Load=True)#subMatrix=True)
#w11,v11,invCMat11 = checkEigs(maskFile11,ISWoutFile,'covar_sparse11.txt',Load=True)#subMatrix=True)
#w12,v12,invCMat12 = checkEigs(maskFile12,ISWoutFile,'covar_sparse12.txt',Load=True)#subMatrix=True)
#w13,v13,invCmat13 = checkEigs(maskFile13,ISWoutFile,'covar6110_R010.npy',Load=True)
print 'time elapsed: ',int((time.time()-startTime)), ' seconds'
# with Load=True: 1s:f7, 11s:f8, 28s:f9, 63s:f10, 111s:f11, 172:f12
#   looks like time increases proportional to npix**2.5 => t(6110) ~= 1042 sec. or 17.3 min
#   Actual: t(6110) = 1296 s = 21.6 min

#np.save('eigenvalues.npy',w13)
#np.save('eigenvectors.npy',v13)
#np.save('invCovar_R010.npy',invCmat13)

cMat13 = mcm.symLoad('covar6110_R010.npy')
w13 = np.load('eigenvalues.npy')
v13 = np.load('eigenvectors.npy')
#invCmat13 = mcm.symLoad('invCovar_R010_sym.npy')
# testing shows that even though this matrix should be diagonal, it's not,
#  so symSave and symLoad should not be used on it.
invCmat13 = np.load('invCovar_R010.npy')

# make plots
#plotEigs(maskFile4,'mask 4',w4,v4)
#plotEigs(maskFile5,'mask 5',w5,v5)
#plotEigs(maskFile6,'mask 6',w6,v6)
#plotEigs(maskFile7,'mask 7',w7,v7)
#plotEigs(maskFile8,'mask 8',w8,v8)
#plotEigs(maskFile9,'mask 9',w9,v9)
#plotEigs(maskFile10,'mask 10',w10,v10)
#plotEigs(maskFile11,'mask 11',w11,v11)
#plotEigs(maskFile12,'mask 12',w12,v12)
plotEigs(maskFile13,'mask 13',w13,v13)


# sort and plot eigenvalues
wSort = np.sort(w13)
plt.semilogy(wSort)
plt.xlabel('6110 pixels over ISW mask')
plt.ylabel('eigenvalues of covariance matrix')
plt.title('Eigenvalues sorted in ascending order')
plt.show()

# multiply matrices
testProduct = np.dot(cMat13,invCmat13)
testDiag = np.diag(testProduct)
plt.plot(testDiag)
plt.xlabel('matrix column number')
plt.ylabel('value on diagonal')
plt.title('c*c^-1 : values should be 1')
plt.show()

# test diagonalization
CR = np.dot(cMat13,v13)
lambdaR = w13*v13
plt.plot(CR-lambdaR)
plt.xlabel('matrix column number')
plt.ylabel('C*R-lambda*R')
plt.title('test of eigen equation; should equal 0')
plt.show()


"""



