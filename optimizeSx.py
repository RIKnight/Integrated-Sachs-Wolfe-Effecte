#! /usr/bin/env python
"""
Name:
  optimizeSx  
Purpose:
  explore the presumed arbitrary cut off point for S_{1/2} by optimizing
    PTE(S_x) for random CMB realizations
Note:  
  Copi et al 2015 state: 
    "the statistical significance of the absence of large-angle correlations 
     is not particularly dependent either on the precise value of either limit" 
     (of the |C(theta)|^2 integral)
Uses:
  healpy
  legprodint (legendre product integral)
Inputs:

Outputs:

Modification History:
  Written by Z Knight, 2016.09.12

  Still need to optimize smica S_x just like others.
  
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time # for measuring duration
from scipy.interpolate import interp1d

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
         newSMICA=False,newDeg=False):
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
  """
#  # first I want something that can create an array of Jmn(x) 
#  # this will be used to find S_x as well as help define histogram bins to store results
#  myJmn = Jmn(new=True)
#  Jmnx=myJmn.getJmn(0.5)
#  print 'Jmnx result: ',Jmnx.shape,Jmnx



  ##############################################################################
  # load theoretical power spectra

  # load data
  ell,fullCl,primCl,lateCl,crossCl = gcp.loadCls(useCLASS=useCLASS,useLensing=useLensing,
                                                  classCamb=classCamb)

  # fill beginning with zeros
  startEll = ell[0]
  ell      = np.append(np.arange(startEll),ell)
  fullCl   = np.append(np.zeros(startEll),fullCl)
  primCl   = np.append(np.zeros(startEll),primCl)
  lateCl   = np.append(np.zeros(startEll),lateCl)
  crossCl  = np.append(np.zeros(startEll),crossCl)

  # suppress C_2 to see what happens in enesmble
  suppressC2 = False
  suppFactor = 0.23 # from Tegmark et. al. 2003, figure 13 (WMAP)
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
  ClTempFile  = 'tempCl.fits'
  mapTempFile = 'tempMap.fits'
  mapDegFile  = 'smicaMapDeg.fits' # this should have been created by sims.getSMICA
  maskDegFile = 'maskMapDeg.fits'  # this should have been created by sims.getSMICA
  
  ispice(mapDegFile,ClTempFile,maskfile1=maskDegFile,subav="YES",subdipole="YES")
  ClsmicaCut = hp.read_cl(ClTempFile)

  # find S_{1/2} for SMICA.  Should actually optimize but see what happens here first.
  myJmn = legprodint.getJmn(endX=0.5,lmax=lmax,doSave=False)
  Ssmica = np.dot(ClsmicaCut[lmin:lmax+1],np.dot(myJmn[lmin:,lmin:],
                  ClsmicaCut[lmin:lmax+1]))*1e24 #K^4 to microK^4


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

  if doTime: print 'time elapsed: ',int((time.time()-startTime)/60.),' minutes'


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

  for nSim in range(nSims):
    print 'starting S(x) sim ',nSim+1,' of ',nSims
    for index,xVal in enumerate(xVals):
      SxToInterpolate[index] = np.dot(ClEnsembleCut[nSim,lmin:lmax+1],
          np.dot(Jmnx[index,lmin:,lmin:],ClEnsembleCut[nSim,lmin:lmax+1]))
    SofX = interp1d(xVals,SxToInterpolate)
    #SofXList = SofXList.append(SofX)
    # Apparently appending a function to an empty list is not allowed. Instead:
    SofXList[nSim] = SofX

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
    #plt.legend()
    plt.title('S(x) for simulations')
    plt.show()



  ##############################################################################
  # create Pval(x) for each S(x), using ensemble
  # Pval: probability of result equal to or more extreme

  # create list of functions
  PvalOfXList = [dummy for i in range(nSims)]
  
  for nSim in range(nSims):
    print 'starting Pval(x) sim ',nSim+1,' of ',nSims

    def PvalOfX(x):
      nUnder = 0 # will also include nEqual
      nOver  = 0
      threshold = SofXList[nSim](x)
      for nSim2 in range(nSims):
         Sx = SofXList[nSim2](x)
         if Sx > threshold: 
           nOver  += 1
         else: 
           nUnder += 1
      return nUnder/float(nUnder+nOver)
    PvalOfXList[nSim] = PvalOfX

  
  ##############################################################################
  # find global minimum for each Pval(x)
  # simply use same xVals as above, at one degree intervals
  # if there are equal p-values along the range, the one with the highest xVal
  #   will be reported

  PvalMinima = np.empty(nSims)
  xValMinima = np.empty(nSims)
  for nSim in range(nSims):
    PvalOfX = PvalOfXList[nSim]
    #print 'function: ',PvalOfX
    PvalMinima[nSim] = PvalOfX(1.0)
    xValMinima[nSim] = 1.0

    Pvals = np.empty(nXvals)
    for index,xVal in enumerate(xVals): # will start from 1 and go down to -1
      myPval = PvalOfX(xVal)
      Pvals[index] = myPval
      if myPval < PvalMinima[nSim]:
        PvalMinima[nSim] = myPval
        xValMinima[nSim] = xVal
        #print 'nSim: ',nSim+1,', new x for minimum Pval: ',xVal

    doPlot = True
    if doPlot and np.random.uniform() < 0.1: #randomly choose about 1/10 of them
      plt.plot(xVals,Pvals)
      plt.vlines(xValMinima[nSim],0,1)
      plt.xlabel('x = cos(theta), min at '+str(xValMinima[nSim]))
      plt.ylabel('P-value')
      plt.title('P-values for simulation '+str(nSim+1)+' of '+str(nSims)+
          ', p_min = '+str(PvalMinima[nSim]))
      plt.xlim(-1.05,1.05)
      plt.ylim(-0.05,1.05)
      plt.show()


  # A MYSTERY!  Something about the following code causes Pvals to always take 
  #   the values of PvalOfXList[nSims](xVals)  WTF?  Omit for now. 
  #   Testing seems to indicate that PvalOfXList functions still have different
  #   locations in memory, but they all seem to be evaluating the same.
  #   However, when the previous block of code is copied to come again after
  #   this one, it behaves properly again.
  # see how well it did
  doPlot = False#True
  if doPlot:
    nPlots = 10
    for nPlot in range(nPlots):
      print 'plot ',nPlot+1,' of ',nPlots
      toPlot = nPlot#np.random.randint(0,high=nSims)
      #for nSim in range(nSims):
      Pvals = np.empty(nXvals)
      PvalOfX = PvalOfXList[nPlot]
      print 'function: ',PvalOfX
      for index, xVal in enumerate(xVals):
        Pvals[index] = PvalOfX(xVal)
        #print index,Pvals[index]
      #print Pvals
      plt.plot(xVals,Pvals)
      plt.vlines(xValMinima[toPlot],0,1)
      plt.xlabel('x = cos(theta), min at '+str(xValMinima[toPlot]))
      plt.ylabel('P-value')
      plt.title('P-values for simulation '+str(toPlot+1)+' of '+str(nSims))
      plt.show()


  ##############################################################################
  # create distribution of S(xValMinima)

  SxEnsembleMin = np.empty(nSims)
  for nSim in range(nSims):
    SxEnsembleMin[nSim] = SofXList[nSim](xValMinima[nSim])


  ##############################################################################
  # plot/print results

  
  print 'plotting S_x distribution... '
  myBins = np.logspace(2,7,100)
  plt.axvline(x=Ssmica,color='g',linewidth=3,label='SMICA masked')
  plt.hist(SxEnsembleMin, bins=myBins,histtype='step',label='cut sky')

  plt.gca().set_xscale("log")
  plt.legend()
  plt.xlabel('S_{1/2} (microK^4)')
  plt.ylabel('Counts')
  plt.title('S_{1/2} of '+str(nSims)+' simulated CMBs')
  plt.show()




  print 'step 3: profit'

if __name__=='__main__':
  test()


