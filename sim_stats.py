#! /usr/bin/env python
"""
Name:
  sim_stats
Purpose:
  create simulated early and late CMB maps and analyze them for CMB anomalies
Uses:
  healpy
  get_crosspower.py
  spice, ispice.py
  legprodint.py
Inputs:
  Data files as specified in get_crosspower.loadCls function
Outputs:

Modification History:
  Written by Z Knight, 2016.06.28
  Fixed x,theta conversion error; ZK, 2016.07.01
  Added spice functionality; ZK, 2016.07.14
  Added legprodint, S_{1/2}; ZK, 2016.07.22

"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time # for measuring duration
#import make_Cmatrix as mcm # for getCl
#from numpy.linalg import cholesky
#from numpy.random import randn
import get_crosspower as gcp
from numpy.polynomial.legendre import legval # for C_l -> C(theta) conversion
from numpy.polynomial.legendre import legfit # for C(theta) -> C_l conversion
#from scipy import stats # stats.norm.interval for finding confidence intervals
from ispice import ispice
from legprodint import getJmn
from scipy.interpolate import interp1d


def getCovar(ell,Cl,theta_i=0.0,theta_f=180.0,nSteps = 1800,doTime=False):
  """
  Purpose:
    create real space covariance array from harmonic space covariance
  Inputs:
    ell: the l values for C_l
      must start with ell value for first nonzero Cl (eg, ell[0]=2)
      must be the same length as Cl
    Cl: the power spectrum C_l
    theta_i: the starting point for angles calculated
      Default: 0.0 degrees
    theta_f: the ending point for angles calculated
      Default: 180.0 degrees
    nSteps: the number of intervals between theta_i and theta_f
      Default: 1800 (1801 values returned)
    doTime: set to True to output time elapsed
      Default: False
  Outputs:
    theta, the theta values for C(theta) array
    C(theta), the covariance array

  """
  startTime = time.time()

  # fill beginning with zeros
  startEll = ell[0]
  ell = np.append(np.arange(startEll),ell)
  Cl  = np.append(np.zeros(startEll),Cl)

  # create legendre coefficients
  legCoef = (2*ell+1)/(4*np.pi) * Cl
  
  # create x values for P(x)
  x_i = np.cos(theta_i*np.pi/180.)
  x_f = np.cos(theta_f*np.pi/180.)
  xArray = np.linspace(x_i,x_f,num=nSteps+1)

  # evalueate legendre polynomials at x values with coefficients
  covar = legval(xArray,legCoef)

  if doTime: print 'time elapsed: ',int((time.time()-startTime)/60),' minutes'

  return np.arccos(xArray)*180./np.pi,covar


def getSMICA(theta_i=0.0,theta_f=180.0,nSteps=1800,lmax=100,newSMICA=False,useSPICE=False):
  """
  Purpose:
    load CMB and mask maps from files, return correlation function for
    unmasked and masked CMB
    Mostly follows Copi et. al. 2013 for cut sky C(theta)
  Uses:
    get_crosspower.py (for plotting)
    C(theta) save file getSMICAfile.npy
  Inputs:
    theta_i,theta_f: starting and ending points for C(theta) in degrees
    nSteps: number of intervals between i,f points
    lmax: the maximum l value to include in legendre series for C(theta)
    newSMICA: set to True to reload data from files and recompute
      if False, will load C(theta) curves from file
    useSPICE: if True, will use SPICE to find power spectra
      if False, will use anafast, following Copi et. al. 2013
      Default: False
  Outupts:
    theta: nSteps+1 angles that C(theta) arrays are for (degrees)
    unmasked: C(theta) unmasked (microK^2)
    masked: C(theta) masked     (microK^2)

  """
  saveFile  = 'getSMICAfile.npy'  #for anafast
  saveFile2 = 'getSMICAfile2.npy' #for spice
  if newSMICA:

    # load maps; default files have 2048,NESTED,GALACTIC
    dataDir   = '/Data/'
    smicaFile = 'COM_CMB_IQU-smica-field-int_2048_R2.01_full.fits'
    maskFile  = 'COM_CMB_IQU-common-field-MaskInt_2048_R2.01.fits'
    print 'opening file ',smicaFile,'... '
    smicaMap,smicaHead = hp.read_map(dataDir+smicaFile,nest=True,h=True)
    print 'opening file ',maskFile,'... '
    maskMap, maskHead  = hp.read_map(dataDir+maskFile, nest=True,h=True)

    # degrade map and mask resolutions from 2048 to 128; convert NESTED to RING
    NSIDE_deg = 128
    while 4*NSIDE_deg < lmax:
      NSIDE_deg *=2
    print 'resampling maps at NSIDE = ',NSIDE_deg,'... '
    order_out = 'RING'
    smicaMapDeg = hp.ud_grade(smicaMap,nside_out=NSIDE_deg,order_in='NESTED',order_out=order_out)
    maskMapDeg  = hp.ud_grade(maskMap, nside_out=NSIDE_deg,order_in='NESTED',order_out=order_out)
      # note: degraded resolution mask will no longer be only 0s and 1s.
      #   Should it be?

    # find power spectra
    print 'find power spectra... '
    if useSPICE:
      # create necessary fits files
      mapDegFile = 'smicaMapDeg.fits'
      maskDegFile = 'maskMapDeg.fits'
      hp.write_map(mapDegFile,smicaMapDeg,nest=False) # use False if order_out='RING' above
      hp.write_map(maskDegFile,maskMapDeg,nest=False)

      # use spice
      ClFile1 = 'spiceCl_unmasked.fits'
      ClFile2 = 'spiceCl_masked.fits'
      ClFile3 = 'spiceCl_mask.fits'
      ispice(mapDegFile,ClFile1)
      Cl_unmasked = hp.read_cl(ClFile1)
      #ispice(mapDegFile,ClFile2,maskfile1=maskDegFile)
      ispice(mapDegFile,ClFile2,weightfile1=maskDegFile)
      Cl_masked = hp.read_cl(ClFile2)
      ispice(maskDegFile,ClFile2)
      Cl_mask = hp.read_cl(ClFile2)
      ell = np.arange(Cl_unmasked.shape[0])

    else: # use anafast
      Cl_unmasked = hp.anafast(smicaMapDeg,lmax=lmax)
      Cl_masked   = hp.anafast(smicaMapDeg*maskMapDeg,lmax=lmax)
      Cl_mask     = hp.anafast(maskMapDeg,lmax=lmax)
      ell = np.arange(lmax+1) #anafast output seems to start at l=0
  
    # plot them
    doPlot = False#True
    if doPlot:
      gcp.showCl(ell,np.array([Cl_masked,Cl_unmasked]),
                 title='power spectra of unmasked, masked SMICA map')

    # create legendre coefficients
    legCoef_unmasked = (2*ell+1)/(4*np.pi) * Cl_unmasked
    legCoef_masked   = (2*ell+1) * Cl_masked
    legCoef_mask     = (2*ell+1) * Cl_mask
  
    # Legendre transform to real space
    print 'Legendre transform to real space... '
    thetaDomain     = np.linspace(theta_i,theta_f,nSteps+1)
    xArray          = np.cos(thetaDomain*np.pi/180.)
    CofTheta        = legval(xArray,legCoef_unmasked)*1e12 #K^2 to microK^2
    CCutofTheta     = legval(xArray,legCoef_masked  )*1e12 #K^2 to microK^2
    AofThetaInverse = legval(xArray,legCoef_mask    )
    CCutofTheta = CCutofTheta/AofThetaInverse

    # back to frequency space for S_{1/2} = CIC calculation
    legCoefs = legfit(xArray,CCutofTheta,lmax)
    CCutofL = legCoefs*(4*np.pi)/(2*ell[:lmax+1]+1)

    # S_{1/2}
    myJmn = getJmn(lmax=lmax)[2:,2:] # do not include monopole, dipole
    SMasked = np.dot(CCutofL[2:],np.dot(myJmn,CCutofL[2:]))
    SNoMask = np.dot(Cl_unmasked[2:lmax+1],
                     np.dot(myJmn,Cl_unmasked[2:lmax+1]))*1e24 #two factors of K^2 to muK^2

    # save results
    if useSPICE:
      np.save(saveFile2,np.array([thetaDomain,CofTheta,CCutofTheta,SNoMask,SMasked]))
    else:
      np.save(saveFile, np.array([thetaDomain,CofTheta,CCutofTheta,SNoMask,SMasked]))

  else: # load from file
    if useSPICE:
      fileData = np.load(saveFile2)
    else:
      fileData = np.load(saveFile)
    thetaDomain = fileData[0]
    CofTheta    = fileData[1]
    CCutofTheta = fileData[2]
    SNoMask     = fileData[3]
    SMasked     = fileData[4]

  return thetaDomain, CofTheta, CCutofTheta, SNoMask, SMasked

def SOneHalf(thetaArray, CArray, nTerms=250):
  """
  Purpose:
    calculate S_{1/2} in real space
  Procedure:
    approximates integral with a finite sum
  Inputs:
    thetaArray: array of angles that CArray is evaluated at
    CArray: the covariance function
    nPoints: the number of terms in the partial sum
      Default: 250
  Returns:
    S_{1/2}

  """
  # Create C(cos(theta)) interpolation
  C = interp1d(np.cos(thetaArray*np.pi/180),CArray)
  nTerms = 250
  xArray = np.linspace(-1.0,0.5,nTerms+1) #use all but the first one
  dx = 1.5 / nTerms
  mySum = 0.0
  for x in xArray[1:]:
    mySum += C(x)**2*dx
  return mySum



################################################################################
# testing code

def test(useCLASS=1,useLensing=0,classCamb=1,nSims=1000,lmax=100,newSMICA=False):
  """
    code for testing the other functions in this module
    Inputs:
      useCLASS: set to 1 to use CLASS, 0 to use CAMB
        CLASS Cl has early/late split at z=50
        CAMB Cl has ISWin/out split: ISWin: 0.4<z<0.75, ISWout: the rest
        Note: CAMB results include primary in ISWin and ISWout (not as intended)
        default: 1
      useLensing: set to 1 to use lensed maps, 0 for non-lensed
        default: 0
      classCamb: if 1: use the CAMB format of CLASS output, if 0: use CLASS format
        Note: parameter not used if useCLASS = 0
        default: 1
      nSims: the number of simulations to do for ensemble
        default: 1000
      lmax: the highest l to include in Legendre transforms
        default: 100
      newSMICA: set to True to recalculate SMICA results
        default: False

  """

  # load data
  ell,fullCl,primCl,lateCl,crossCl = gcp.loadCls(useCLASS=useCLASS,useLensing=useLensing,classCamb=classCamb)

  conv = ell*(ell+1)/(2*np.pi)
  #print ell,conv #ell[0]=2.0

  """
  # verify statistical properties of alm realizations
  nSims = 1000
  lmax = 100
  Clprim_sum = np.zeros(lmax+1)
  Cllate_sum = np.zeros(lmax+1)
  Clcros_sum = np.zeros(lmax+1) # prim x late
  for nSim in range(nSims):
    print 'starting sim ',nSim+1, ' of ',nSims
    #alm_prim,alm_late = getAlms(A_lij,lmax=lmax) #AKW method defunct
    # see if synalm can do it
    alm_prim,alm_late = hp.synalm((primCl,lateCl,crossCl),lmax=lmax,new=True)
    Clprim_sum = Clprim_sum + hp.alm2cl(alm_prim)
    Cllate_sum = Cllate_sum + hp.alm2cl(alm_late)
    Clcros_sum = Clcros_sum + hp.alm2cl(alm_prim,alm_late)
  Cl_prim_avg = Clprim_sum/nSims
  Cl_late_avg = Cllate_sum/nSims
  Cl_cros_avg = Clcros_sum/nSims

  doPlot = True
  if doPlot:
    plt.plot(ell[:lmax+1],Cl_prim_avg*conv[:lmax+1])
    plt.plot(ell[:lmax+1],primCl[:lmax+1]*conv[:lmax+1])
    plt.title('primary')
    plt.ylabel('D_l')
    plt.show()

    plt.plot(ell[:lmax+1],Cl_late_avg*conv[:lmax+1])
    plt.plot(ell[:lmax+1],lateCl[:lmax+1]*conv[:lmax+1])
    plt.title('late')
    plt.ylabel('D_l')
    plt.show()

    plt.plot(ell[:lmax+1],Cl_cros_avg*conv[:lmax+1])
    plt.plot(ell[:lmax+1],crossCl[:lmax+1]*conv[:lmax+1])
    plt.title('cross')
    plt.ylabel('D_l')
    plt.show()
  """


  # get covariances from SMICA map and mask
  theta_i = 0.0 #degrees
  theta_f = 180.0 #degrees
  nSteps = 1800
  #lmax = 100

  # get unmasked and masked SMICA covariances
  #   note: getSMICA uses linspace in theta for thetaArray
  #newSMICA = False#True
  thetaArray2, C_SMICA, C_SMICAmasked, S_SMICAnomask, S_SMICAmasked = \
    getSMICA(theta_i=theta_i,theta_f=theta_f,nSteps=nSteps,lmax=lmax,newSMICA=newSMICA,
             useSPICE=False)
  print ''
  print 'S_{1/2}(anafast): SMICA, no mask: ',S_SMICAnomask,', masked: ',S_SMICAmasked
  print ''
  # get C_l from SPICE to compare to above method
  #   note: getSMICA uses linspace in theta for thetaArray
  #newSMICA = False#True
  thetaArray2sp, C_SMICAsp, C_SMICAmaskedsp, S_SMICAnomasksp, S_SMICAmaskedsp = \
    getSMICA(theta_i=theta_i,theta_f=theta_f,nSteps=nSteps,lmax=lmax,newSMICA=newSMICA,
             useSPICE=True)
  print ''
  print 'S_{1/2}(spice): SMICA, no mask: ',S_SMICAnomasksp,', masked: ',S_SMICAmaskedsp
  print ''

  # Find S_{1/2} in real space to compare methods
  SSnm2   = SOneHalf(thetaArray2,  C_SMICA)
  SSmd2   = SOneHalf(thetaArray2,  C_SMICAmasked)
  SSnm2sp = SOneHalf(thetaArray2sp,C_SMICAsp)
  SSmd2sp = SOneHalf(thetaArray2sp,C_SMICAmaskedsp)


  # create ensemble of realizations and gather statistics
  covEnsemble  = np.zeros([nSims,nSteps+1]) # for maskless
  covEnsemble2 = np.zeros([nSims,nSteps+1]) # for masked
  covTheta = np.array([])
  #nSims = 1000

  # apply beam and pixel window functions to power spectra
  #   note: to ignore the non-constant pixel shape, W(l) must be > B(l)
  #     however, this is not true for NSIDE=128 and gauss_beam(5'')
  #   Here I ignore this anyway and proceed
  myNSIDE = 128 # must be same NSIDE as in getSMICA function
  Wpix = hp.pixwin(myNSIDE)
  Bsmica = hp.gauss_beam(5./60*np.pi/180) # 5 arcmin
  WlMax = Wpix.size
  if WlMax < lmax:
    print 'die screaming!!!'
    return 0
  primCl  =  primCl[:WlMax]*(Wpix*Bsmica)**2
  lateCl  =  lateCl[:WlMax]*(Wpix*Bsmica)**2
  crossCl = crossCl[:WlMax]*(Wpix*Bsmica)**2

  doTime = True # to time the run and print output
  startTime = time.time()
  for nSim in range(nSims):
    print 'starting sim ',nSim+1, ' of ',nSims
    alm_prim,alm_late = hp.synalm((primCl,lateCl,crossCl),lmax=lmax,new=True)
    #print 'alm_prim: ',alm_prim
    #print 'alm l,m indices: ',hp.Alm.getlm(lmax+1)

    # calculate C(theta) of simulation
    Clsim_prim = hp.alm2cl(alm_prim)
    Clsim_late = hp.alm2cl(alm_late)
    Clsim_cros = hp.alm2cl(alm_prim,alm_late)
    Clsim_full = Clsim_prim + 2*Clsim_cros + Clsim_late
    # use Cl_sim_full to omit prim/late distinction for now

    # first without mask  
    #   note: getCovar uses linspace in x for thetaArray
    thetaArray,cArray = getCovar(ell[:lmax+1],Clsim_full,theta_i=theta_i,theta_f=theta_f,
                                 nSteps=nSteps)
    covEnsemble[nSim] = cArray
    covTheta = thetaArray

    # now with a mask
    mapSim = hp.synfast(Clsim_full,myNSIDE,lmax=lmax) # should have default RING ordering
    # super lame that spice needs to read/write from disk, but here goes...
    mapTempFile = 'tempMap.fits'
    ClTempFile  = 'tempCl.fits'
    maskDegFile = 'maskMapDeg.fits' # this should have been created by getSMICA
    hp.write_map(mapTempFile,mapSim)
    ispice(mapTempFile,ClTempFile,weightfile1=maskDegFile)
    Cl_masked = hp.read_cl(ClTempFile)
    ell2 = np.arange(Cl_masked.shape[0])
    #   note: getCovar uses linspace in x for thetaArray
    thetaArray,cArray2 = getCovar(ell2[:lmax+1],Cl_masked[:lmax+1],theta_i=theta_i,
                                   theta_f=theta_f,nSteps=nSteps)
    covEnsemble2[nSim] = cArray2
    
    doPlot = False#True
    if doPlot:
      plt.plot(thetaArray,cArray)
      plt.xlabel('theta (degrees)')
      plt.ylabel('C(theta)')
      plt.title('covariance of simulated CMB')
      plt.show()


  if doTime: print 'time elapsed: ',int((time.time()-startTime)/60.),' minutes'
  avgEnsemble = np.average(covEnsemble, axis = 0)
  stdEnsemble = np.std(covEnsemble, axis = 0)
  # do I need a better way to describe confidence interval?
  avgEnsemble2 = np.average(covEnsemble2, axis = 0)
  stdEnsemble2 = np.std(covEnsemble2, axis = 0)



  doPlot = True
  if doPlot:
    print 'plotting correlation functions... '
    # first the whole sky statistics
    plt.plot(thetaArray,avgEnsemble,label='sim. ensemble average (no mask)')
    plt.fill_between(thetaArray,avgEnsemble+stdEnsemble,avgEnsemble-stdEnsemble,alpha=0.25,
                     label='simulation 1sigma envelope')
    plt.plot(thetaArray2,C_SMICA,label='SMICA R2 (inpainted,anafast)')
    plt.plot(thetaArray2sp,C_SMICAsp,label='SMICA R2 (inpainted,spice)')
    

    plt.xlabel('theta (degrees)')
    plt.ylabel('C(theta)')
    plt.title('whole sky covariance of '+str(nSims)+' simulated CMBs, lmax='+str(lmax))
    plt.ylim([-500,1000])
    plt.plot([0,180],[0,0]) #horizontal line
    plt.legend()
    plt.show()

    # now the cut sky
    plt.plot(thetaArray,avgEnsemble2,label='sim. ensemble average (masked)')
    plt.fill_between(thetaArray,avgEnsemble2+stdEnsemble2,avgEnsemble2-stdEnsemble2,alpha=0.25,
                     label='simulation 1sigma envelope')
    plt.plot(thetaArray2,C_SMICAmasked,label='SMICA R2 (masked ,anafast)')
    plt.plot(thetaArray2sp,C_SMICAmaskedsp,label='SMICA R2 (masked ,spice)')
    
    plt.xlabel('theta (degrees)')
    plt.ylabel('C(theta)')
    plt.title('cut sky covariance of '+str(nSims)+' simulated CMBs, lmax='+str(lmax))
    plt.ylim([-500,1000])
    plt.plot([0,180],[0,0]) #horizontal line
    plt.legend()
    plt.show()

  # S_{1/2} output
  print ''
  print 'using CIC method: '
  print 'S_{1/2}(anafast): SMICA, no mask: ',S_SMICAnomask,', masked: ',S_SMICAmasked
  print 'S_{1/2}(spice): SMICA, no mask: ',S_SMICAnomasksp,', masked: ',S_SMICAmaskedsp
  print ''
  print 'using CCdx method: '
  print 'S_{1/2}(anafast): SMICA, no mask: ',SSnm2,', masked: ',SSmd2
  print 'S_{1/2}(spice): SMICA, no mask: ',SSnm2sp,', masked: ',SSmd2sp
  print ''

  



if __name__=='__main__':
  test()


