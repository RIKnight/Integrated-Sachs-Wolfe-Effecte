#! /usr/bin/env python
"""
  NAME:
    CovMatrix.py
  PURPOSE:
    Program to create a covariance matrix for a given set of HEALpix pixels
      and angular power spectrum.
  USES:
    fits file containing mask indicating the set of pixels to use
    fits file containing ISWout power spectrum
      where ISWout indicates no ISW for 0.4 < z < 0.75
  MODIFICATION HISTORY:
    Written by Z Knight, 2015.09.14
    This is a rewritten version of the 2016.01.20 version of make_Cmatrix.py
      In this version, covariance matrices are created, stored, and manipulated 
      by objects of the CovMatrix class;  by Z Knight, 2019.01.28

"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legval
from scipy.special import legendre
#import time # for measuring duration
import healpy as hp
import astropy.io.fits as pf

class CovMatrix:
    """
    Name: 
        CovMatrix
    Purpose:
        Create, contain, and manipulate a covariance matrix. 
        This is meant to be used for the theoretical covariance of a Guassian random field with:
            a given power spectrum, specified locations on a spherical map, 
            and observational effects of a telescope beam and pixelization.
    Uses:
        healpy: the python implementation of HEALpix
        astropy.io.fits: to read and write FITS files
        numpy, scipy, matplotlib: standard python libraries
    """
    
    def __init__(self, loadFile=None, maskFile=None, powerFile=None, highpass = 0, beamSmooth = True, 
                 pixWin = True, lmax=2000, useMicro=False, nested=False):
        """
        Class constructor.  Initialize with loadFile to load a covariance matrix from file,
            or Initialize with maskFile and powerFile to create a covariance matrix from those.
                If you try to do both, the matrix will be loaded, not calculated.
            Values of all parameters are saved to object variables, unless loadFile, maskFile, and 
                powerFile are all None.
        Inputs:
            loadFile: name of a CovMatrix save file that contains a covariance matrix.  
              If this is used, the values of all other parameters are ignored during init, 
              but stored in object.
            maskFile: name of a healpix fits file that contains 1 where a pixel is to be included
              in covariance matrix and 0 otherwise
              Must be NSIDE=64
            powerFile: a CAMB CMB power spectrum file with units K**2
            highpass: the lowest multipole l to not be zeroed out.
              Default is 0
            beamSmooth: determines whether to use beamsmoothing on C_l,
              also controls lmax
              Default is True, with lmax = 250
            pixWin: determines whether to use the pixel window,
              also controls lmax
              Default is True, with lmax = 250
            lmax: maximum l value in Legendre series.
              Note: this value is overridden by beamSmooth and pixWin lmax settings
              Default is 2000
            useMicro: converts power spectrum units from K**2 to microK**2 before calculating matrix
              Default is False
            nested: NESTED vs RING parameter to be used with healpy functions
              IMPORTANT!!! Note that nested parameter used to create C matrix must match that used
              in every function that uses the C matrix
              Default is False
        """
        # initialize based on inputs:
        if loadFile is not None:
            self.covLoad(loadFile=loadFile)
            
            # save the file names and other parameters
            self._loadFile = loadFile
            self._maskFile = maskFile
            self._powerFile = powerFile
            self._highpass = highpass
            self._beamSmooth = beamSmooth
            self._pixWin = pixWin
            self._lmax = lmax
            self._useMicro = useMicro
            self._nested = nested
            print 'Loaded covariance matrix from file.'
            
        elif maskFile is not None and powerFile is not None:
            self._covMat = self.makeCmatrix(maskFile, powerFile, highpass=highpass, 
                                            beamSmooth=beamSmooth, pixWin=pixWin, lmax=lmax, 
                                            useMicro=useMicro, nested=nested)
            # save the file names and other parameters
            self._loadFile = loadFile
            self._maskFile = maskFile
            self._powerFile = powerFile
            self._highpass = highpass
            self._beamSmooth = beamSmooth
            self._pixWin = pixWin
            self._lmax = lmax
            self._useMicro = useMicro
            self._nested = nested
        else:
            # This case is just for catching errors... may not otherwise be useful.
            print 'Initializing with no covariance matrix.'
            self._covMat = None
            
        # don't calculate the inverse covariance matrix yet
        self._eigVals = None
        self._eigVecs = None
        self._invMat = None
        
        
    
    
    ###############################################################################
    # the methods for saving and loading matrices
    
    def symSave(self,saveMe,saveFile='symsave.npy'):
        """
        Turns symmetric matrix into a vector and saves it to disk.
        Inputs:
            saveMe: the symmetric numpy 2d array to save
            saveFile: the file name to save.  .npy will be appended if not already present
                default: symsave.npy
        """
        indices = np.triu_indices_from(saveMe)
        np.save(saveFile,saveMe[indices])
    
    def covSave(self,saveFile='symsave.npy'):
        """
        Saves the covariance matrix to an array using symSave
        Inputs:
            saveFile: the file name to save.  .npy will be appended if not already present
                default: symsave.npy
        """
        self.symSave(self._covMat,saveFile=saveFile)
        
    def invSave(self,saveFile='symsave.npy'):
        """
        Saves the inverse covariance matrix to an array using symSave
        Inputs:
            saveFile: the file name to save.  .npy will be appended if not already present
                default: symsave.npy
        """
        self.symSave(self._invMat,saveFile=saveFile)
    
    def symLoad(self,loadFile='symsave.npy'):
        """
        Loads a numpy .npy array from file and transforms it into a symmetric array
        Inputs:
            loadFile: the name of the file to load
                default: symsave.npy
        Returns a symmetric 2d numpy array
        """
        loaded = np.load(loadFile)
        n = int(-1/2.+np.sqrt(1/4.+2*loaded.size))
        array = np.zeros([n,n])
        indices = np.triu_indices(n)
        array[indices]=loaded
        return array+np.transpose(array)-np.diag(np.diag(array))
            
    def covLoad(self,loadFile='symsave.npy'):
        """
        Loads a numpy .npy array from file and transforms it into a symmetric array,
            and stores it in the object's covariance matrix variable
        Inputs:
            loadFile: the name of the file to load
                default: symsave.npy
        """
        self._covMat = self.symLoad(loadFile=loadFile)
        
    def invLoad(self,loadFile='symsave.npy'):
        """
        Loads a numpy .npy array from file and transforms it into a symmetric array,
            and stores it in the object's inverse covariance matrix variable
        Inputs:
            loadFile: the name of the file to load
                default: symsave.npy
        """
        self._invMat = self.symLoad(loadFile=loadFile)
    
    
    ###############################################################################
    # the methods for creating the covariance matrix
    
    def getCl(self,filename):
        """
        Opens a CAMB FITS file and extracts the Power spectrum
        Inputs:
            filename: the name of the FITS file to open
        Returns two arrays: one of ell, one of C_l
        """
        powSpec = pf.getdata(filename,1)
        temps = powSpec.field('TEMPERATURE')
        ell = np.arange(temps.size)
        return ell,temps


    def makeCmatrix(self, maskFile, powerFile, highpass = 0, beamSmooth = True, pixWin = True,
                    lmax=2000, useMicro=False, nested=False):
        """
        Method to make the covariance matrix
        Inputs:
            maskFile: name of a healpix fits file that contains 1 where a pixel is to be included
              in covariance matrix and 0 otherwise
              Must be NSIDE=64
            powerFile: a CAMB CMB power spectrum file with units K**2
            highpass: the lowest multipole l to not be zeroed out.
              Default is 0
            beamSmooth: determines whether to use beamsmoothing on C_l,
              also controls lmax
              Default is True, with lmax = 250
            pixWin: determines whether to use the pixel window,
              also controls lmax
              Default is True, with lmax = 250
            lmax: maximum l value in Legendre series.
              Note: this value is overridden by beamSmooth and pixWin lmax settings
              Default is 2000
            useMicro: converts power spectrum units from K**2 to microK**2 before calculating matrix
              Default is False
            nested: NESTED vs RING parameter to be used with healpy functions
              IMPORTANT!!! Note that nested parameter used to create C matrix must match that used
              in every function that uses the C matrix
              Default is False
        Returns the covariance matrix, with units K**2 or microK**2, depending on value of useMicro parameter
        """
        # read mask file
        mask = hp.read_map(maskFile,nest=nested)
      
        # read power spectrum file
        ell,C_l = self.getCl(powerFile)
      
        # read coordinates file
        if nested:
          coordsFile64 = '/Data/pixel_coords_map_nested_galactic_res6.fits'
        else:
          coordsFile64 = '/Data/pixel_coords_map_ring_galactic_res6.fits'
        gl,gb = hp.read_map(coordsFile64,(0,1),nest=nested)
      
        # isolate indices of pixels indicated by mask
        myGl = gl[np.where(mask)]
        myGb = gb[np.where(mask)]
        #print 'mask size: ',myGl.size,' or ',myGb.size
      
        # convert to unit vectors
        unitVectors = hp.rotator.dir2vec(myGl,myGb,lonlat=True)
        print "Shape of unitVectors: ",unitVectors.shape
        
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
        if pixWin:
          lmax = 250
          Wpix = hp.pixwin(64)
          W_l = Wpix[:lmax+1]
        else:
          W_l = 1.0
        if beamSmooth:
          lmax = 250
          B_l = hp.gauss_beam(120./60*np.pi/180,lmax=lmax) # 120 arcmin to be below W_l
        else:
          B_l = 1.0
        print "lmax cutoff imposed at l=",lmax
      
        fac_l = (2*ell[:lmax+1]+1)/(4*np.pi)
        C_l = np.concatenate((np.zeros(highpass),C_l[highpass:]))
        if useMicro: # convert C_l units from K**2 to muK**2:
          C_l = C_l * 1e12
        preFac_l = fac_l *B_l**2 *W_l**2 *C_l[:lmax+1]
      
        # evaluate legendre series with legval
        covArray = np.zeros([vecSize,vecSize])
        for row in range(vecSize):
          print 'starting row ',row
          for column in range(row,vecSize):
            covArray[row,column] = legval(cosThetaArray[row,column],preFac_l)
          #for column in range(row):
          #  covArray[row,column] = covArray[column,row]
        covArray = covArray + covArray.T - np.diag(np.diag(covArray))
      
        return covArray

    
    
    ###############################################################################
    # methods for extracting submatrices
    
    # depreciated.  May remove
    def subMatrix(self,maskFile,bigMaskFile,cMatrixFile,nested=False):
        """
        Purpose:
            function to extract a C matrix from a matrix made for a larger set of pixels.
        Args:
            maskFile: FITS file containg a mask indicating which pixels to use
                Must be a submask of the mask used to create the larger covariance matrix
            bigMaskFile: FITS file containing a mask that corresponds to the pixels
                used to create the cMatrix stored in cMatrixFile
            cMatrixFile: numpy file containing a symSave C matrix
            nested: NESTED vs RING parameter to be used with healpy functions
        Uses:
            submatrix2
        Returns:
            returns a numpy array containing a C matrix
        """
      
        mask = hp.read_map(maskFile,nest=nested)
        bigMask = hp.read_map(bigMaskFile,nest=nested)
        print 'loading C matrix from file ',cMatrixFile
        cMatrix = symLoad(cMatrixFile)
        return self.subMatrix2(mask,bigMask,cMatrix,nested=nested)
    
    
    def subMatrix2(self,mask,bigMask,cMatrix,nested=False):
        """
        Purpose:
            function to extract a C matrix from a matrix made for a larger set of pixels.
        Args:
            mask: a mask indicating which pixels to use
            bigMask: a mask that corresponds to the pixels used to create the cMatrix
            cMatrix: a numpy array containing a C matrix
            nested: NESTED vs RING parameter to be used with healpy functions
      
        Returns:
            returns a numpy array containing a C matrix
        """
      
        maskVec = np.where(mask)[0] #array of indices
        bigMaskVec = np.where(bigMask)[0] #array of indices
        print 'looping through indices to create sub-matrix...'
        # check for mask pixels outside bigmask:
        for pixel in maskVec:
          if pixel not in bigMaskVec:
            print 'error: small mask contains pixel outside of big mask.'
            return 0
        subVec = [bigI for bigI in range(bigMaskVec.size) for subI in range(maskVec.size) if maskVec[subI] == bigMaskVec[bigI] ]
        print 'done'
      
        subCmat = cMatrix[np.meshgrid(subVec,subVec)]  #.transpose() # transpose not needed for symmetric
        
        return subCmat

    # this one is new for the CovMatrix class
    def getSubMatrix(self,maskFile):
        """
        Purpose:
            function to extract a C matrix from a matrix made for a larger set of pixels.
        Args:
            maskFile: FITS file containg a mask indicating which pixels to use
                Must be a submask of the mask used to create the larger covariance matrix
        Uses:
            submatrix2
        Returns:
            returns a numpy array containing a C matrix
        """
      
        mask = hp.read_map(maskFile,nest=self._nested)
        bigMask = hp.read_map(self._maskFile,nest=self._nested)
        return self.subMatrix2(mask,bigMask,self._covMat,nested=self._nested)
    

    ###############################################################################
    # methods for calcuating inverse covariance matrix
    
    def makeEigs(self):
        """
        calculates the eigenvalues and eigenvectors of the covariance matrix,
            and stores them.
        """
        if self._eigVals is None:
            self._eigVals, self._eigVecs = np.linalg.eigh(self._covMat)
    
    def RDInvert(self):
        """
        function to calculate symmetric inverse of Covariance matrix using
            eigen decomposition of covariance matrix
            Result is stored in self._invMat
        """
        self.makeEigs()
        diagInv = np.diag(self._eigVals**-1)
        eigVecsTranspose = np.transpose(self._eigVecs)
        cMatInv = np.dot(np.dot(self._eigVecs,diagInv),eigVecsTranspose)
        self._invMat = cMatInv
        #return cInv
    
    def choInvert(self):
        """
        Purpose:
            Function to find inverse of symmetric matrix using Cholesky decomposition
            Result is stored in self._invMat
        """
        L = np.linalg.cholesky(self._covMat) # defined for real symmetric matrix: cMatrix = L * L.T
        Linv = np.linalg.inv(L)  # uses LU decomposition, but L is already L
        cMatInv = np.dot(Linv.T,Linv)
        self._invMat = cMatInv
        #return cMatInv
        
    def LUInvert(self):
        """
        Purpose:
            Function to find inverse of symmetric matrix using LU decomposition
                (standard method used by numpy invert)
            Result is stored in self._invMat
        """
        cMatInv = np.linalg.inv(self._covMat)
        self._invMat = cMatInv
        


    ###############################################################################
    # methods for displaying object data

    def showCl(self,ell,temps,title='CAMB ISWout power spectrum'):
        """
          create a plot of power spectrum
          Inputs:
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
    
    
    def getCovMat(self):
        return self._covMat
    def getInvMat(self):
        return self._invMat
    
    
################################################################################
# testing code

def test(newMatrix=False,doPlots=True):
    """
    a rudimentary test function that checks some of the basic functionality
    Inputs:
        newMatrix: set to True to calculate a new matrix from 3000 pixels
        doPlots: set to True to plot Mollwiede projections of the test masks
    
    """
    import time # for measuring duration
    
    # Create covMat from mask and power spectrum
    # tiny masks, nside=64, 100,100,190 pixels
    maskFile1 = '/Data/sparsemask_1.fits'
    maskFile2 = '/Data/sparsemask_2.fits'
    maskFile3 = '/Data/sparsemask_3.fits'
    maskFile = maskFile2
    #saveMatrixFile = '/Data/covar100_test.npy'
    
    ISWoutFile = 'ISWout_scalCls.fits'
    ISWinFile = 'ISWin_scalCls.fits'
      
    myCovMat = CovMatrix(maskFile=maskFile, powerFile=ISWinFile, highpass=12, beamSmooth=True, pixWin=True, nested=True)
    
    print myCovMat.getCovMat()

    
    # testing the getSubMatrix routine

    # need mask and submask
    maskFile4 = '/Data/PSG/small_masks/ISWmask_din1_R010_trunc3000.fits'  #47 minutes to make covMat for this
    maskFile5 = '/Data/PSG/small_masks/ISWmask_din1_R010_trunc2500.fits'
    # This is not a submask.  It contains pixels outside the bigger mask.  
    # That makes it good for checking the execution of the error condition.
    
    nested = True
    mask = hp.read_map(maskFile4,nest=nested)
    print mask.shape
    print np.sum(mask)
    
    if doPlots:
        hp.mollview(mask, nest=True)
        plt.show()
    
    nested = True
    mask = hp.read_map(maskFile5,nest=nested)
    print mask.shape
    print np.sum(mask)
    
    if doPlots:
        hp.mollview(mask, nest=True)
        plt.show()
    
    if newMatrix:
        # tests creating a larger matrix and the saving routine
        startTime = time.time()
        myCovMat2 = CovMatrix(maskFile=maskFile4, powerFile=ISWinFile, highpass=12, beamSmooth=True, pixWin=True, nested=True)
        print 'time elapsed: ',int((time.time()-startTime)/60),' minutes'
        print 'big matrix shape: ',myCovMat2.getCovMat().shape
        myCovMat2.covSave(saveFile = 'covar3000.npy')
        
    else:
        # tests the loading routine
        myCovMat2 = CovMatrix(loadFile='covar3000.npy',maskFile=maskFile4, powerFile=ISWinFile, 
                      highpass=12, beamSmooth=True, pixWin=True, nested=True)
        print 'big matrix shape: ',myCovMat2.getCovMat().shape
        
    # get submatrix
    subMat = myCovMat2.getSubMatrix(maskFile4) #use the same mask to check all pixels
    #subMat = myCovMat2.getSubMatrix(maskFile5) #submask does not ovelap and will crash program
    print 'sub matrix shape: ',subMat.shape
    


if __name__=='__main__':
    test()



    