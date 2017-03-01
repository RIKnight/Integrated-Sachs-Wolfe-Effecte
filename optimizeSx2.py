#! /usr/bin/env python
"""
Name:
  optimizeSx2  
Purpose:
  explore the presumed arbitrary cut off point for S_{1/2} by optimizing
    PTE(S_x) for random CMB realizations
Note:  
  Copi et al 2015 state: 
    "the statistical significance of the absence of large-angle correlations 
     is not particularly dependent either on the precise value of either limit" 
     (of the |C(theta)|^2 integral)
  This program originally copied from optimizeSx, and modified to call c programs 
    for nested looping
Uses:
  healpy
  legprodint (legendre product integral)
  ramdisk.sh (creates and deletes RAM disks)
  optimizeSx.so (c language shared object)
Inputs:

Outputs:

Modification History:
  Written by Z Knight, 2016.09.26
  Added S_x density and P histograms; ZK, 2016.09.30
  Added 2d histograms; ZK, 2016.10.03
  Extracted plotting to its own function; ZK, 2016.10.04
  Narrowed range of x for searching for optimal P-values; ZK, 2016.10.06
  Switched to useLensing=1 in loadCls; Added PvalPval function; ZK, 2016.10.07
  Added filterC2 and filtFactor functionality; ZK, 2016.10.10
  Added makeCornerPlotSmall function; ZK, 2016.10.20
  Modified for filtFactor range: filtFacLow and filtFacHigh; ZK, 2016.11.08
  Added C2 Ensemble for calculation of C_2^LCDM,cut-sky p-value; ZK, 2016.11.09
  Added doCovar option to test function; ZK, 2016.11.12
  Slight output adjustment; ZK, 2016.11.15
  Removed titles from plots; ZK, 2016.11.17
  Added saveAndExit for extracting S(x) curves; ZK, 2016.11.17
  Separated optSx from test function; removed SofXList; ZK, 2016.11.20
  Added parameter to optSx for sublist selection; ZK, 2016.12.07
  Added hardcoded kludge to PvalPval for error margin checking; ZK, 2016.12.13
  
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time               # for measuring duration
from scipy.interpolate import interp1d
import subprocess         # for calling RAM Disk scripts
import ctypes                         # for calling c .so file
from numpy.ctypeslib import ndpointer # for calling c .so file
from matplotlib import cm # color maps for 2d histograms
import corner             # for corner plot

import get_crosspower as gcp # for loadCls
from numpy.polynomial.legendre import legval # for C_l -> C(theta) conversion
from numpy.polynomial.legendre import legfit # for C(theta) -> C_l conversion
from ispice import ispice # for calculating C_l from map
import legprodint         # for integral of product of legendre polynomials
import sim_stats as sims  # for getSMICA and getCovar


def PvalPval(saveFile="optSxResult.npy"):
  """
    Name:
      PvalPval
    Purpose:
      calculate the p-value of the p-value
    Inputs:
      saveFile: name of a numpy file containing 3*(nSims+1) element array
        3 rows: PvalMinima,XvalMinima,SxEnsembleMin
        fisrt (0th) column: SMICA values
        the other columns: nSims
        Default: optSxResult.npy
    Returns:
      nothing, but prints result
  """
  
  # load results
  myPX = np.load(saveFile)
  PvalMinima = myPX[0]

  nUnder = 0 # will also include nEqual
  nOver = 0
  Psmica = PvalMinima[0]
  #Psmica = 0.00360996390036 #read from output
  #Psmica += 0.00019 #high end
  #Psmica -= 0.00019 #low end
  print 'P-value for ensemble ',saveFile,': ',Psmica
  for Pval in (PvalMinima[1:]):
    if Psmica >= Pval:
      nUnder +=1
    else:
      nOver  +=1
  return nUnder/float(nUnder+nOver)

################################################################################
# plotting

def makeC2Plot(saveFile='optSxC2.npy'):
  """
    Name:
      maceC2Plot
    Purpose:
      plot histogram of cut-sky C2 values and compare to SMICA C2 value
    Inputs:
      saveFile: the name of the file that C2 Ensemble is saved in
      Must have SMICA value as 0th member of numpy array
    Returns:
      p-value of SMICA C2 power
  """

  # create markers
  mCLASS_C2 = 1103.42
  m10 = mCLASS_C2*0.1
  m20 = mCLASS_C2*0.2

  # load results
  C2Ensemble = np.load(saveFile)
  nSims = C2Ensemble.size-1 #-1 due to SMICA value in 0 position

  # make plot
  print 'Plotting C2 distribution...'
  myBins = np.linspace(0,2500,100)
  plt.axvline(x=C2Ensemble[0],color='g',linewidth=3,label='SMICA masked')
  plt.axvline(x=mCLASS_C2,color='k',label='CLASS C_2')
  plt.axvline(x=m10,      color='k',label='CLASS C_2 * 0.1')
  plt.axvline(x=m20,      color='k',label='CLASS C_2 * 0.2')
  plt.hist(C2Ensemble[1:], bins=myBins,histtype='step',label='cut-sky sim.s')
                      # [1:] to omit SMICA value
  #plt.gca().set_xscale("log")
  plt.legend()
  plt.xlabel(r'$C_2$ power $(\mu K^2)$')
  plt.ylabel('Counts')
  #plt.title(r'$C_2$ of '+str(nSims)+' simulated CMBs') 
  plt.show()

  # find p-value
  nUnder = 0
  nOver = 0
  C2smica = C2Ensemble[0]
  for C2power in (C2Ensemble[1:]):
    if C2smica >= C2power:
      nUnder +=1
    else:
      nOver  +=1
  pVal = nUnder/float(nUnder+nOver)

  return pVal


def makeCornerPlotSmall(saveFile="optSxResult.npy",suppressC2=False):
  """
    Name:
      makeCornerPlotSmall
    Purpose:
      plotting results of optimizeSx calculations
      Makes a plot with fewer categories than makeCornerPlot
    Inputs:
      saveFile: name of a numpy file containing 3*(nSims+1) element array
        3 rows: PvalMinima,XvalMinima,SxEnsembleMin
        fisrt (0th) column: SMICA values
        the other columns: nSims
        Default: optSxResult.npy
      suppressC2: set to True if this was used in creating data
        Default: False
    Returns:
      nothing, but makes several plots
  """

  # load results
  myPX = np.load(saveFile)
  PvalMinima = myPX[0]
  XvalMinima = myPX[1]
  SxEnsembleMin = myPX[2]
  nSims = myPX.shape[1]  # actually nSims+1 since SMICA is in 0 position

  # S_x / delta_x
  #SxEnsembleMinDensity = SxEnsembleMin/(XvalMinima + 1)

  # reshape data for logarithmic P-value plot
  toPlot = np.vstack((np.log10(SxEnsembleMin[1:]),#np.log10(SxEnsembleMinDensity[1:]),
                      np.log10(PvalMinima[1:]),XvalMinima[1:]))
  toPlot = toPlot.T

  # make corner plot
  figure = corner.corner(toPlot,labels=[r"$\log_{10} S_x$", #r"$\log_{10} (S_x / \Delta x)$",
                                        r"$\log_{10}$ P-value",r"x value"],
                         show_titles=False,
                         truths=[np.log10(SxEnsembleMin[0]),#maknp.log10(SxEnsembleMinDensity[0]),
                                 np.log10(PvalMinima[0]),XvalMinima[0]]  )
  plt.show()

  print 'for SMICA: x = ',XvalMinima[0],', S_x = ',SxEnsembleMin[0],', P(x) = ',PvalMinima[0]


def makeCornerPlot(saveFile="optSxResult.npy",suppressC2=False):
  """
    Name:
      makeCornerPlot
    Purpose:
      plotting results of optimizeSx calculations
    Inputs:
      saveFile: name of a numpy file containing 3*(nSims+1) element array
        3 rows: PvalMinima,XvalMinima,SxEnsembleMin
        fisrt (0th) column: SMICA values
        the other columns: nSims
        Default: optSxResult.npy
      suppressC2: set to True if this was used in creating data
        Default: False
    Returns:
      nothing, but makes several plots
  """

  # load results
  myPX = np.load(saveFile)
  PvalMinima = myPX[0]
  XvalMinima = myPX[1]
  SxEnsembleMin = myPX[2]
  nSims = myPX.shape[1]  # actually nSims+1 since SMICA is in 0 position

  # S_x / delta_x
  SxEnsembleMinDensity = SxEnsembleMin/(XvalMinima + 1)

  # reshape data for logarithmic P-value plot
  toPlot = np.vstack((np.log10(SxEnsembleMin[1:]),np.log10(SxEnsembleMinDensity[1:]),
                      np.log10(PvalMinima[1:]),XvalMinima[1:]))
  toPlot = toPlot.T

  # make corner plot
  figure = corner.corner(toPlot,labels=[r"$\log_{10} S_x$", r"$\log_{10} (S_x / \Delta x)$",
                                        r"$\log_{10}$ P-value",r"x value"],
                         show_titles=False,
                         truths=[np.log10(SxEnsembleMin[0]),np.log10(SxEnsembleMinDensity[0]),
                                 np.log10(PvalMinima[0]),XvalMinima[0]]  )
  plt.show()

  """
  # reshape data for linear P-value plot
  toPlot = np.vstack((np.log10(SxEnsembleMin[1:]),np.log10(SxEnsembleMinDensity[1:]),
                      PvalMinima[1:],XvalMinima[1:]))
  toPlot = toPlot.T

  # make corner plot
  figure = corner.corner(toPlot,labels=[r"$\log_{10} S_x$", r"$\log_{10} (S_x / \Delta x)$",
                                        r"P-value",r"x value"],
                         show_titles=False,
                         truths=[np.log10(SxEnsembleMin[0]),np.log10(SxEnsembleMinDensity[0]),
                                 PvalMinima[0],XvalMinima[0]]  )
  plt.show()
  """


def makePlots(saveFile="optSxResult.npy",suppressC2=False):
  """
    Name:
      makePlots
    Purpose:
      plotting results of optimizeSx calculations
    Inputs:
      saveFile: name of a numpy file containing 3*(nSims+1) element array
        3 rows: PvalMinima,XvalMinima,SxEnsembleMin
        fisrt (0th) column: SMICA values
        the other columns: nSims
        Default: optSxResult.npy
      suppressC2: set to True if this was used in creating data
        Default: False
    Returns:
      nothing, but makes several plots
  """

  # load results
  myPX = np.load(saveFile)
  PvalMinima = myPX[0]
  XvalMinima = myPX[1]
  SxEnsembleMin = myPX[2]
  nSims = myPX.shape[1]  # actually nSims+1 since SMICA is in 0 position

  # S_x / delta_x
  SxEnsembleMinDensity = SxEnsembleMin/(XvalMinima + 1)


  print 'plotting S_x distribution... '
  myBinsS = np.logspace(0,6,100)
  plt.axvline(x=SxEnsembleMin[0],color='g',linewidth=3,label='SMICA masked')
  plt.hist(SxEnsembleMin[1:], bins=myBinsS,histtype='step',label='cut sky')
                      # [1:] to omit SMICA value
  plt.gca().set_xscale("log")
  #plt.legend()
  plt.xlabel('S_x (microK^4)')
  plt.ylabel('Counts')
  """
  if suppressC2:
    plt.title('S_x of '+str(nSims-1)+' simulated CMBs, C_2 suppressed') 
                                #-1 due to SMICA in zero position
  else:
    plt.title('S_x of '+str(nSims-1)+' simulated CMBs') 
                                #-1 due to SMICA in zero position
  """
  plt.show()

  print 'plotting S_x density distribution... '
  myBinsS = np.logspace(1,7,100)
  plt.axvline(x=SxEnsembleMinDensity[0],color='g',linewidth=3,label='SMICA masked')
  plt.hist(SxEnsembleMinDensity[1:], bins=myBinsS,histtype='step',label='cut sky')
  plt.gca().set_xscale("log")
  #plt.legend()
  plt.xlabel('S_x / delta_x (microK^4)')
  plt.ylabel('Counts')
  """
  if suppressC2:
    plt.title('S_x / delta_x of '+str(nSims-1)+' simulated CMBs, C_2 suppressed') 
  else:
    plt.title('S_x / delta_x of '+str(nSims-1)+' simulated CMBs') 
  """
  plt.show()

  print 'plotting P-value distribution (logarithmic)... '
  myBinsP = np.logspace(-3,0,100)
  myUniform = np.linspace(0,1,nSims) # for comparison to uniform distribution
  plt.axvline(x=PvalMinima[0],color='g',linewidth=3,label='SMICA masked')
  plt.hist(PvalMinima[1:], bins=myBinsP,histtype='step',label='cut sky')
  plt.hist(myUniform,bins=myBinsP,histtype='step',label='uniform dist.')
  plt.gca().set_xscale("log")
  #plt.legend()
  plt.xlabel('P-value')
  plt.ylabel('Counts')
  """
  if suppressC2:
    plt.title('P-value of S_x of '+str(nSims-1)+' simulated CMBs, C_2 suppressed') 
  else:
    plt.title('P-value of S_x of '+str(nSims-1)+' simulated CMBs') 
  """
  plt.show()

  """
  print 'plotting P-value distribution (linear)... '
  myBinsP2 = np.linspace(0,1,100)
  myUniform = np.linspace(0,1,nSims) # for comparison to uniform distribution
  plt.axvline(x=PvalMinima[0],color='g',linewidth=3,label='SMICA masked')
  plt.hist(PvalMinima[1:], bins=myBinsP2,histtype='step',label='cut sky')
  plt.hist(myUniform,bins=myBinsP2,histtype='step',label='uniform dist.')
  #plt.gca().set_xscale("log")
  #plt.legend()
  plt.xlabel('P-value')
  plt.ylabel('Counts')
  if suppressC2:
    plt.title('P-value of S_x of '+str(nSims-1)+' simulated CMBs, C_2 suppressed') 
  else:
    plt.title('P-value of S_x of '+str(nSims-1)+' simulated CMBs') 
  plt.show()
  """

  print 'plotting x distribution... '
  myBinsX = np.linspace(-1,1,100)
  plt.axvline(x=XvalMinima[0],color='g',linewidth=3,label='SMICA masked')
  plt.hist(XvalMinima[1:], bins=myBinsX,histtype='step',label='cut sky')
  plt.xlabel('x value')
  plt.ylabel('Counts')
  """
  if suppressC2:
    plt.title('x value of S_x of '+str(nSims-1)+' simulated CMBs, C_2 suppressed') 
  else:
    plt.title('x value of S_x of '+str(nSims-1)+' simulated CMBs') 
  """
  plt.show()


  print 'and now the 2d histograms... '
  log10SxEnsembleMin = np.log10(SxEnsembleMin)
  log10SxEnsembleMinDensity = np.log10(SxEnsembleMinDensity)
  log10PvalMinima = np.log10(PvalMinima)
  myBinsLog10S0 = np.linspace(0,6,100)
  myBinsLog10S1 = np.linspace(1,7,100)
  myBinsLog10P = np.linspace(-3,0,100)
  myBinsX = np.linspace(-1,1,100)
  cmap = cm.magma#Greens#Blues

  # SxEnsembleMin vs. XvalMinima
  plt.hist2d(log10SxEnsembleMin[1:],XvalMinima[1:],
      bins=[myBinsLog10S0,myBinsX],cmax=10,cmap=cmap)
  plt.plot(log10SxEnsembleMin[0],XvalMinima[0],'ro')
  plt.colorbar()
  plt.xlabel('log10(S_x)')
  plt.ylabel('X value')
  #plt.title('S_x vs. X value')
  plt.show()
  # SxEnsembleMinDensity vs. XvalMinima
  plt.hist2d(log10SxEnsembleMinDensity[1:],XvalMinima[1:],
      bins=[myBinsLog10S1,myBinsX],cmax=10,cmap=cmap)
  plt.plot(log10SxEnsembleMinDensity[0],XvalMinima[0],'ro')
  plt.colorbar()
  plt.xlabel('log10(S_x/delta_x)')
  plt.ylabel('X value')
  #plt.title('S_x/delta_x vs. X value')
  plt.show()
  # PvalMinima vs. XvalMinima
  plt.hist2d(log10PvalMinima[1:],XvalMinima[1:],
      bins=[myBinsLog10P,myBinsX],cmax=10,cmap=cmap)
  plt.plot(log10PvalMinima[0],XvalMinima[0],'ro')
  plt.colorbar()
  plt.xlabel('log10(P-value)')
  plt.ylabel('X value')
  #plt.title('P value vs. X value')
  plt.show()

  # SxEnsembleMin vs. PvalMinima
  plt.hist2d(log10SxEnsembleMin[1:],log10PvalMinima[1:],
      bins=[myBinsLog10S0,myBinsLog10P],cmax=50,cmap=cmap)
  plt.plot(log10SxEnsembleMin[0],log10PvalMinima[0],'ro')
  plt.colorbar()
  plt.xlabel('log10(S_x)')
  plt.ylabel('log10(P-value)')
  #plt.title('S_x vs. P-value')
  plt.show()
  # SxEnsembleMinDensity vs. PvalMinima
  plt.hist2d(log10SxEnsembleMinDensity[1:],log10PvalMinima[1:],
      bins=[myBinsLog10S1,myBinsLog10P],cmax=50,cmap=cmap)
  plt.plot(log10SxEnsembleMinDensity[0],log10PvalMinima[0],'ro')
  plt.colorbar()
  plt.xlabel('log10(S_x/delta_x)')
  plt.ylabel('log10(P-value)')
  #plt.title('S_x/delta_x vs. P-value')
  plt.show()

  # SxEnsembleMin vs. SxEnsembleMinDensity
  plt.hist2d(log10SxEnsembleMin[1:],log10SxEnsembleMinDensity[1:],
      bins=[myBinsLog10S0,myBinsLog10S1],cmax=50,cmap=cmap)
  plt.plot(log10SxEnsembleMin[0],log10SxEnsembleMinDensity[0],'ro')
  plt.colorbar()
  plt.xlabel('log10(S_x)')
  plt.ylabel('log10(S_x/delta_x)')
  #plt.title('S_x vs. S_x/delta_x')
  plt.show()


################################################################################
# the main functions: cOptSx

# define cOptSx wrapper
def optSx(xVals,nXvals,SxValsArray,nSims,xStart,xEnd,nSearch,
          PvalMinima,XvalMinima,mySxSubset=None):
  """
    Name:
      optSx
    Purpose:
      create Pval(x) for each S(x), using ensemble
        Pval: probability of result equal to or more extreme
    Prodecure:
      find global minimum for each Pval(x)
      if there are equal p-values along the range, the one with the lowest xVal
        will be reported
    Inputs:
      most are used identically to those in optimizeSx.c
      mySxSubset: (not the same as in optimizeSx.c)
        Set to a list of indices to include in the optSx calculation
        Default: None (indicates using all available SxVals; all indices included)
    Outputs:
      output is through parameters PvalMinima, XvalMinima

  """
  # check for mySxSubset
  if mySxSubset is None:
    mySxSubset = np.arange(nSims,dtype=np.uint64) #for all indices in range
    nSubset = nSims
  else:
    nSubset = mySxSubset.size

  lib = ctypes.cdll.LoadLibrary("../../C/optimizeSx.so")
  cOptSx = lib.optSx
  cOptSx.restype = None
  cOptSx.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                     ctypes.c_size_t,
                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS",ndim=2,shape=(nSims,nXvals)),
                     ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                     ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"), ctypes.c_size_t]
  
  cOptSx(xVals,nXvals,SxValsArray,nSims,xStart,xEnd,nSearch,
         PvalMinima,XvalMinima,mySxSubset,nSubset)


################################################################################
# testing code

def test(useCLASS=1,useLensing=1,classCamb=1,nSims=1000,lmax=100,lmin=2,
         newSMICA=False,newDeg=False,suppressC2=False,suppFactor=0.23,
         filterC2=False,filtFacLow=0.1,filtFacHigh=0.2,doCovar=False):
  """
    code for testing the other functions in this module
    Inputs:
      useCLASS: set to 1 to use CLASS, 0 to use CAMB
        CLASS Cl has early/late split at z=50
        CAMB Cl has ISWin/out split: ISWin: 0.4<z<0.75, ISWout: the rest
        Note: CAMB results include primary in ISWin and ISWout (not as intended)
        default: 1
      useLensing: set to 1 to use lensed Cl, 0 for non-lensed
        default: 1
      classCamb: if 1: use the CAMB format of CLASS output, if 0: use CLASS format
        Note: parameter not used if useCLASS = 0
        default: 1
      nSims: the number of simulations to do for ensemble
        default: 1000
      lmax: the highest l to include in Legendre transforms
        default: 100
      lmin: the lowest l to include in S_{1/2} = CIC calculations
        default: 2
      newSMICA: set to True to recalculate SMICA results
        default: False
      newDeg: set to True to recalculate map and mask degredations
        (only if newSMICA is also True)
        default: False
      suppressC2: set to True to suppress theoretical C_2 (quadrupole) by 
        suppFactor before creating a_lm.s
        Default: False
      suppFactor: multiplies C_2 if suppressC2 is True
        Default: 0.23 # from Tegmark et. al. 2003, figure 13 (WMAP)
      filterC2 : set to true to filter simulated CMBs after spice calculates
        cut sky C_l.  Sims will pass filter if C_2 * filtFacLow < C_2^sim <
        C_2 * filtFacHigh.
        Default: False
      filtFacLow,filtFacHigh: defines C_2 range for passing simulated CMBs
        Default: 0.1,0.2
      doCovar: set to True to calculate C(theta) and S_{1/2} distritutions for ensemble
        Note: meant to capture functionality from sim_stats.py; ZK 2016.11.13
        Default: False
  """

  ##############################################################################
  # load theoretical power spectra

  # load data
  ell,fullCl,primCl,lateCl,crossCl = gcp.loadCls(useCLASS=useCLASS,useLensing=useLensing,
                                                  classCamb=classCamb)

  # fill beginning with zeros
  startEll = int(ell[0])
  ell      = np.append(np.arange(startEll),ell)
  fullCl   = np.append(np.zeros(startEll),fullCl)
  primCl   = np.append(np.zeros(startEll),primCl)
  lateCl   = np.append(np.zeros(startEll),lateCl)
  crossCl  = np.append(np.zeros(startEll),crossCl)

  # suppress C_2 to see what happens in enesmble
  if suppressC2:
    fullCl[2] *= suppFactor
    primCl[2] *= suppFactor
    lateCl[2] *= suppFactor
    crossCl[2] *= suppFactor

  conv = ell*(ell+1)/(2*np.pi)
  #print ell,conv #ell[0]=2.0

  # apply beam and pixel window functions to power spectra
  #   note: to ignore the non-constant pixel shape, W(l) must be > B(l)
  #     however, this is not true for NSIDE=128 and gauss_beam(5')
  #   Here I ignore this anyway and proceed
  myNSIDE = 128 # must be same NSIDE as in sims.getSMICA function
  Wpix = hp.pixwin(myNSIDE)
  Bsmica = hp.gauss_beam(5./60*np.pi/180) # 5 arcmin
  WlMax = Wpix.size
  if WlMax < lmax:
    print 'die screaming!!!'
    return 0
  fullCl  =  fullCl[:WlMax]*(Wpix*Bsmica)**2
  primCl  =  primCl[:WlMax]*(Wpix*Bsmica)**2
  lateCl  =  lateCl[:WlMax]*(Wpix*Bsmica)**2
  crossCl = crossCl[:WlMax]*(Wpix*Bsmica)**2
  # note: i tried sims without this scaling, and results seemed the same at a glance


  ##############################################################################
  # load SMICA data, converted to C_l, via SpICE

  if newSMICA or doCovar:
    theta_i = 0.0 #degrees
    theta_f = 180.0 #degrees
    nSteps = 1800
    thetaArray2sp, C_SMICAsp, C_SMICAmaskedsp, S_SMICAnomasksp, S_SMICAmaskedsp = \
      sims.getSMICA(theta_i=theta_i,theta_f=theta_f,nSteps=nSteps,lmax=lmax,lmin=lmin,
               newSMICA=newSMICA,newDeg=newDeg,useSPICE=True)

  # filenames for SpICE to use
  # super lame that spice needs to read/write from disk, but here goes...
  RAMdisk     = '/Volumes/ramdisk/'
  ClTempFile  = RAMdisk+'tempCl.fits'
  mapTempFile = RAMdisk+'tempMap.fits'
  mapDegFile  = RAMdisk+'smicaMapDeg.fits' # this should have been created by sims.getSMICA
  maskDegFile = RAMdisk+'maskMapDeg.fits'  # this should have been created by sims.getSMICA
  
  # create RAM Disk for SpICE and copy these files there using bash
  RAMsize = 4 #Mb
  ramDiskOutput = subprocess.check_output('./ramdisk.sh create '+str(RAMsize), shell=True)
  print ramDiskOutput
  diskID = ramDiskOutput[31:41] # this might not grab the right part; works for '/dev/disk1'
  subprocess.call('cp smicaMapDeg.fits '+RAMdisk, shell=True)
  subprocess.call('cp maskMapDeg.fits ' +RAMdisk, shell=True)


  ispice(mapDegFile,ClTempFile,maskfile1=maskDegFile,subav="YES",subdipole="YES")
  ClsmicaCut = hp.read_cl(ClTempFile)

  # find S_{1/2} for SMICA.  Should actually optimize but see what happens here first.
  if doCovar:
    myJmn = legprodint.getJmn(endX=0.5,lmax=lmax,doSave=False)
    #Ssmica = np.dot(ClsmicaCut[lmin:lmax+1],np.dot(myJmn[lmin:,lmin:],
    #                ClsmicaCut[lmin:lmax+1]))*1e24 #K^4 to microK^4


  ##############################################################################
  # create ensemble of realizations and gather statistics

  spiceMax = myNSIDE*3 # should be lmax+1 for SpICE
  ClEnsembleCut  = np.zeros([nSims,spiceMax])
  if doCovar:
    ClEnsembleFull = np.zeros([nSims,lmax+1])
  simEll = np.arange(spiceMax)

  # option for creating C(\theta) and S_{1/2} ensembles
  if doCovar:
    cEnsembleCut  = np.zeros([nSims,nSteps+1])
    cEnsembleFull = np.zeros([nSims,nSteps+1])
    sEnsembleCut  = np.zeros(nSims)
    sEnsembleFull = np.zeros(nSims)

  doTime = True # to time the run and print output
  startTime = time.time()
  #for nSim in range(nSims):
  nSim = 0
  while nSim < nSims:
    print 'starting masked Cl sim ',nSim+1, ' of ',nSims
    alm_prim,alm_late = hp.synalm((primCl,lateCl,crossCl),lmax=lmax,new=True)
    mapSim = hp.alm2map(alm_prim+alm_late,myNSIDE,lmax=lmax)
    hp.write_map(mapTempFile,mapSim)
    if doCovar:
      ClEnsembleFull[nSim] = hp.alm2cl(alm_prim+alm_late)

    ispice(mapTempFile,ClTempFile,maskfile1=maskDegFile,subav="YES",subdipole="YES")
    ClEnsembleCut[nSim] = hp.read_cl(ClTempFile)

    # Check for low power of cut sky C_2
    if (filterC2 == True and fullCl[2]*filtFacHigh > ClEnsembleCut[nSim,2]
                         and ClEnsembleCut[nSim,2] > fullCl[2]*filtFacLow) or filterC2 == False:

      doPlot = False#True
      if doPlot:
        gcp.showCl(simEll[:lmax+1],ClEnsembleCut[nSim,:lmax+1],
                    title='power spectrum of simulation '+str(nSim+1))

      if doCovar:
        #   note: getCovar uses linspace in x for thetaArray
        thetaArray,cArray = sims.getCovar(simEll[:lmax+1],ClEnsembleCut[nSim,:lmax+1],
                                  theta_i=theta_i,theta_f=theta_f,nSteps=nSteps,lmin=lmin)
        cEnsembleCut[nSim] = cArray
        thetaArray,cArray = sims.getCovar(simEll[:lmax+1],ClEnsembleFull[nSim,:lmax+1],
                                  theta_i=theta_i,theta_f=theta_f,nSteps=nSteps,lmin=lmin)
        cEnsembleFull[nSim] = cArray
      
        # S_{1/2}
        sEnsembleCut[nSim]  = np.dot(ClEnsembleCut[nSim,lmin:lmax+1],
                                     np.dot(myJmn[lmin:,lmin:],ClEnsembleCut[nSim,lmin:lmax+1]))
        sEnsembleFull[nSim] = np.dot(ClEnsembleFull[nSim,lmin:lmax+1],
                                     np.dot(myJmn[lmin:,lmin:],ClEnsembleFull[nSim,lmin:lmax+1]))

      nSim +=1


  timeInterval1 = time.time()-startTime
  if doTime: print 'time elapsed: ',int(timeInterval1/60.),' minutes'

  # free the RAM used by SpICE's RAM disk
  ramDiskOutput = subprocess.check_output('./ramdisk.sh delete '+diskID, shell=True)
  print ramDiskOutput

  # put SMICA in as 0th member of the ensemble; 1e12 to convert K^2 to microK^2
  ClEnsembleCut   = np.vstack((ClsmicaCut*1e12,ClEnsembleCut))
  nSims += 1

  ##############################################################################
  # create S(x) for each C_l, using interpolation
  
  nXvals = 181
  thetaVals = np.linspace(0,180,nXvals) # one degree intervals
  xVals = np.cos(thetaVals*np.pi/180)
  
  Jmnx = np.empty([nXvals,lmax+1,lmax+1])
  for index, xVal in enumerate(xVals):
    Jmnx[index] = legprodint.getJmn(endX=xVal,lmax=lmax,doSave=False)
  SxToInterpolate = np.empty(nXvals)

  # create list of functions
  #dummy = lambda x: x**2
  #SofXList = [dummy for i in range(nSims)]


  # here is where this program starts to diverge from the purely python version
  # create array to hold S_x values
  SxValsArray = np.empty([nSims,nXvals])

  
  for nSim in range(nSims):
    print 'starting S(x) sim ',nSim+1,' of ',nSims
    for index,xVal in enumerate(xVals):  #not using xVal?
      SxToInterpolate[index] = np.dot(ClEnsembleCut[nSim,lmin:lmax+1],
          np.dot(Jmnx[index,lmin:,lmin:],ClEnsembleCut[nSim,lmin:lmax+1]))
    #SofX = interp1d(xVals,SxToInterpolate)
    #SofXList[nSim] = SofX

    SxValsArray[nSim] = SxToInterpolate

  """
    #print SofXList#[nSim]
    doPlot=False#True
    if doPlot:
      nplotx = (nXvals-1)*10+1
      plotTheta = np.linspace(0,180,nplotx)
      plotx = np.cos(plotTheta*np.pi/180)
      plotS = SofXList[nSim](plotx)
      plt.plot(plotx,plotS)
      plt.title('S(x) for simulation '+str(nSim+1))
      plt.show()

  doPlot = False#True
  if doPlot:
    for nSim in range(nSims):
      nplotx = (nXvals-1)*10+1
      plotTheta = np.linspace(0,180,nplotx)
      plotx = np.cos(plotTheta*np.pi/180)
      plotS = SofXList[nSim](plotx)
      plt.plot(plotx,plotS,label='sim '+str(nSim+1))
      #plt.plot(xVals,SxValsArray[nSim],label='sim '+str(nSim+1))
    #plt.legend()
    plt.title('S(x) for '+str(nSims)+ 'simulations')
    plt.xlabel('x')
    plt.ylabel('S_x')
    plt.show()
  """

  # Kludge for extracting the S(x) ensemble to disk for Jackknife testing later
  saveAndExit = False #True
  saveAndExitFile = 'SofXEnsemble.npy'
  if saveAndExit:
    np.save(saveAndExitFile,np.vstack((xVals,SxValsArray)))
    print 'saving file ',saveAndExitFile,' and exiting.'
    return 0


  ##############################################################################
  # send data to c library function in optimizeSx.so

  xStart = -1.0
  xEnd = 1.0
  nSearch = 181 # same num as nXvals for now, but spaced equally in x, not theta
  PvalMinima = np.empty(nSims) # for return values
  XvalMinima = np.empty(nSims) # for return values

  doTime = True # to time the run and print output
  startTime = time.time()
  optSx(xVals,nXvals,SxValsArray,nSims,xStart,xEnd,nSearch,PvalMinima,XvalMinima)
  timeInterval2 = time.time()-startTime
  if doTime: print 'time elapsed: ',int(timeInterval2/60.),' minutes'


  ##############################################################################
  # create distribution of S(XvalMinima)

  SxEnsembleMin = np.empty(nSims)
  for nSim in range(nSims):
    # need to interpolate since optSx uses interpolation
    SofX = interp1d(xVals,SxValsArray[nSim])
    SxEnsembleMin[nSim] = SofX(XvalMinima[nSim])


  ##############################################################################
  # save S_x, P(x), x results
  saveFile  = "optSxResult.npy"
  np.save(saveFile,np.vstack((PvalMinima,XvalMinima,SxEnsembleMin)))
  saveFileC2 = "optSxC2.npy"
  np.save(saveFileC2,ClEnsembleCut[:,2]) #for C_2

  # save C(theta) and S{1/2} results
  if doCovar:
    avgEnsembleFull = np.average(cEnsembleFull, axis = 0)
    stdEnsembleFull = np.std(cEnsembleFull, axis = 0)
    # do I need a better way to describe confidence interval?
    avgEnsembleCut = np.average(cEnsembleCut, axis = 0)
    stdEnsembleCut = np.std(cEnsembleCut, axis = 0)

    saveFile1 = "simStatResultC.npy"
    np.save(saveFile1,np.vstack((thetaArray,avgEnsembleFull,stdEnsembleFull,
                                 avgEnsembleCut,stdEnsembleCut)) )
    saveFile2 = "simStatC_SMICA.npy"
    np.save(saveFile2,np.vstack((thetaArray2sp,C_SMICAsp,C_SMICAmaskedsp)) )
    
    saveFile3 = "simStatResultS.npy"
    np.save(saveFile3,np.vstack(( np.hstack((np.array(S_SMICAnomasksp),sEnsembleFull)),
                                  np.hstack((np.array(S_SMICAmaskedsp),sEnsembleCut)) )) )

  ##############################################################################
  # plot/print results
  makePlots(saveFile=saveFile,suppressC2=suppressC2)
  #makeCornerPlot(saveFile=saveFile,suppressC2=suppressC2)
  makeCornerPlotSmall(saveFile=saveFile,suppressC2=suppressC2)
  c2pval = makeC2Plot(saveFile=saveFileC2)
  if doCovar:
    sims.makePlots(saveFile1=saveFile1,saveFile2=saveFile2,saveFile3=saveFile3)

  pv = PvalPval(saveFile=saveFile)
  print ' '
  print 'nSims = ',nSims-1
  print 'time interval 1: ',timeInterval1,'s, time interval 2: ',timeInterval2,'s'
  print '  => ',timeInterval1/(nSims-1),' s/sim, ',timeInterval2/(nSims-1),' s/sim'
  print 'SMICA optimized S_x: S = ',SxEnsembleMin[0],', for x = ',XvalMinima[0], \
        ', with p-value ',PvalMinima[0]
  print 'P-value of P-value for SMICA: ',pv
  print ' '
  print 'p-value of C_2^SMICA in distribution: ',c2pval
  print ' '


  print 'step 3: profit'
  print ''

if __name__=='__main__':
  test(nSims=500,suppressC2=False)


