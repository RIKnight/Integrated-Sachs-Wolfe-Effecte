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
  This program copied from optimizeSx, and modified to call c programs for
    nested looping
Uses:
  healpy
  legprodint (legendre product integral)
  ramdisk.sh (creates and deletes RAM disks)
  optimizeSx.so (c language shared object)
Inputs:

Outputs:

Modification History:
  Written by Z Knight, 2016.09.26
  
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time # for measuring duration
from scipy.interpolate import interp1d
import subprocess # for calling RAM Disk scripts
import ctypes
from numpy.ctypeslib import ndpointer

import get_crosspower as gcp # for loadCls
from numpy.polynomial.legendre import legval # for C_l -> C(theta) conversion
from numpy.polynomial.legendre import legfit # for C(theta) -> C_l conversion
from ispice import ispice
import legprodint
import sim_stats as sims # for getSMICA and getCovar


class Jmn():
  """
  Purpose:
    hold Jmn(x) for various x values and interpolate between them

  Note:
    I may drop this and do the interpolation with the S_x values instead


  Procedure:
    
  Inputs:

  Returns:
    
  """

  def __init__(self,lmax=100,new=True):
    if new:
      print 'new Jmn init running'
    else: 
      print 'old Jmn init running'
    
    nVals = 11#101
    self.Jmnx = np.empty([nVals,lmax+1,lmax+1])
    self.xvalues = np.linspace(0.25,0.75,nVals)
    for index, val in enumerate(self.xvalues):
      self.Jmnx[index] = legprodint.getJmn(endX=val,lmax=lmax,doSave=False)

  def getJmn(self,x):
    """
    Purpose:
      gets (interpolates?) Jmn(x)
    """
    print 'getJmn does not yet interpolate'

    return self.Jmnx



################################################################################
# testing code

def test(useCLASS=1,useLensing=0,classCamb=1,nSims=1000,lmax=100,lmin=2,
         newSMICA=False,newDeg=False,suppressC2=False,suppFactor=0.23):
  """
    code for testing the other functions in this module
    Inputs:
      useCLASS: set to 1 to use CLASS, 0 to use CAMB
        CLASS Cl has early/late split at z=50
        CAMB Cl has ISWin/out split: ISWin: 0.4<z<0.75, ISWout: the rest
        Note: CAMB results include primary in ISWin and ISWout (not as intended)
        default: 1
      useLensing: set to 1 to use lensed Cl, 0 for non-lensed
        default: 0
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
      suppressC2: set to True to suppress theoretical C_2 by suppFactor
        before creating a_lm.s
        Default: False
      suppFactor: multiplies C_2 if suppressC2 is True
        Default: 0.23 # from Tegmark et. al. 2003, figure 13 (WMAP)
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
  #suppressC2 = False
  #suppFactor = 0.23 # from Tegmark et. al. 2003, figure 13 (WMAP)
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
  # load SMICA data, converted to C(theta), via SpICE

  if newSMICA:
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
  #myJmn = legprodint.getJmn(endX=0.5,lmax=lmax,doSave=False)
  #Ssmica = np.dot(ClsmicaCut[lmin:lmax+1],np.dot(myJmn[lmin:,lmin:],
  #                ClsmicaCut[lmin:lmax+1]))*1e24 #K^4 to microK^4


  ##############################################################################
  # create ensemble of realizations and gather statistics

  spiceMax = myNSIDE*3 # should be lmax+1 for SpICE
  ClEnsembleCut = np.zeros([nSims,spiceMax])
  simEll = np.arange(spiceMax)

  doTime = True # to time the run and print output
  startTime = time.time()
  for nSim in range(nSims):
    print 'starting masked Cl sim ',nSim+1, ' of ',nSims
    alm_prim,alm_late = hp.synalm((primCl,lateCl,crossCl),lmax=lmax,new=True)
    mapSim = hp.alm2map(alm_prim+alm_late,myNSIDE,lmax=lmax)
    hp.write_map(mapTempFile,mapSim)

    ispice(mapTempFile,ClTempFile,maskfile1=maskDegFile,subav="YES",subdipole="YES")
    ClEnsembleCut[nSim] = hp.read_cl(ClTempFile)

    doPlot = False#True
    if doPlot:
      gcp.showCl(simEll[:lmax+1],ClEnsembleCut[nSim,:lmax+1],
                  title='power spectrum of simulation '+str(nSim+1))

  timeInterval1 = time.time()-startTime
  if doTime: print 'time elapsed: ',int(timeInterval1/60.),' minutes'

  # free the RAM used by SpICE's RAM disk
  ramDiskOutput = subprocess.check_output('./ramdisk.sh delete '+diskID, shell=True)
  print ramDiskOutput

  # put SMICA in as 0th member of the ensemble; 1e12 to convert K^2 to microK^2
  ClEnsembleCut = np.vstack((ClsmicaCut*1e12,ClEnsembleCut))
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
  dummy = lambda x: x**2
  SofXList = [dummy for i in range(nSims)]


  # here is where this program starts to diverge from the purely python version
  # create array to hold S_x values
  SxValsArray = np.empty([nSims,nXvals])

  for nSim in range(nSims):
    print 'starting S(x) sim ',nSim+1,' of ',nSims
    for index,xVal in enumerate(xVals):  #not using xVal?
      SxToInterpolate[index] = np.dot(ClEnsembleCut[nSim,lmin:lmax+1],
          np.dot(Jmnx[index,lmin:,lmin:],ClEnsembleCut[nSim,lmin:lmax+1]))
    SofX = interp1d(xVals,SxToInterpolate)
    SofXList[nSim] = SofX

    SxValsArray[nSim] = SxToInterpolate

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

  doPlot = True
  if doPlot:
    for nSim in range(nSims):
      nplotx = (nXvals-1)*10+1
      plotTheta = np.linspace(0,180,nplotx)
      plotx = np.cos(plotTheta*np.pi/180)
      plotS = SofXList[nSim](plotx)
      plt.plot(plotx,plotS,label='sim '+str(nSim+1))
      #plt.plot(xVals,SxValsArray[nSim],label='sim '+str(nSim+1))
    #plt.legend()
    plt.title('S(x) for simulations')
    plt.show()


  ##############################################################################
  # send data to c library function in optimizeSx.so
  #
  # create Pval(x) for each S(x), using ensemble
  #   Pval: probability of result equal to or more extreme
  # find global minimum for each Pval(x)
  # if there are equal p-values along the range, the one with the lowest xVal
  #   will be reported

  # write to file to send to c program
  #np.savetxt('optSx.tmp',np.vstack((xVals,SxValsArray)))
  #print 'file optSx.tmp created.'

  # try calling c as shared library
  #import ctypes
  #from numpy.ctypeslib import ndpointer
  lib = ctypes.cdll.LoadLibrary("../../C/optimizeSx.so")
  cOptSx = lib.optSx
  cOptSx.restype = None
  cOptSx.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                     ctypes.c_size_t,
                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS",ndim=2,shape=(nSims,nXvals)),
                     ctypes.c_size_t, ctypes.c_int,
                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
  
  nSearch = 181 # same num as nXvals for now, but spaced equally in x, not theta
  PvalMinima = np.empty(nSims) # for return values
  XvalMinima = np.empty(nSims) # for return values

  doTime = True # to time the run and print output
  startTime = time.time()
  #print  SxValsArray[5]
  #raw_input("that was python's SxValsArray[5].  Press enter.")
  cOptSx(xVals,nXvals,SxValsArray,nSims,nSearch,PvalMinima,XvalMinima)
  timeInterval2 = time.time()-startTime
  if doTime: print 'time elapsed: ',int(timeInterval2/60.),' minutes'


  ##############################################################################
  # create distribution of S(XvalMinima)

  SxEnsembleMin = np.empty(nSims)
  for nSim in range(nSims):
    SxEnsembleMin[nSim] = SofXList[nSim](XvalMinima[nSim])
  #print XvalMinima
  #print SxEnsembleMin

  # extract SMICA result
  Ssmica = SxEnsembleMin[0]


  ##############################################################################
  # plot/print results

  
  print 'plotting S_x distribution... '
  myBins = np.logspace(1,7,100)
  plt.axvline(x=Ssmica,color='g',linewidth=3,label='SMICA masked')
  plt.hist(SxEnsembleMin[1:], bins=myBins,histtype='step',label='cut sky')
                      # [1:] to omit SMICA value

  plt.gca().set_xscale("log")
  plt.legend()
  plt.xlabel('S_x (microK^4)')
  plt.ylabel('Counts')
  if suppressC2:
    plt.title('S_x of '+str(nSims-1)+' simulated CMBs, C_2 suppressed') 
                                #-1 due to SMICA in zero position
  else:
    plt.title('S_x of '+str(nSims-1)+' simulated CMBs') 
                                #-1 due to SMICA in zero position
  plt.show()

  print ' '
  print 'nSims = ',nSims-1
  print 'time interval 1: ',timeInterval1,'s, time interval 2: ',timeInterval2,'s'
  print '  => ',timeInterval1/(nSims-1),' s/sim, ',timeInterval2/(nSims-1),' s/sim'
  print 'SMICA optimized S_x: S = ',Ssmica,', for x = ',XvalMinima[0], \
        ', with p-value ',PvalMinima[0]
  print ' '


  print 'step 3: profit'
  print ''

if __name__=='__main__':
  test(nSims=500,suppressC2=False)


