#! /usr/bin/env python
"""
Name:
  quadoctcorr (quadrupole-octopole and correlation)  
Purpose:
  extract l=2,3 a_lm.s from SMICA to compare C(theta) against simulations
Uses:
  healpy
  legprodint
  get_crosspower
  sim_stats
  ispice
Inputs:

Outputs:

Modification History:
  Largely copied from sim_stats; Z Knight, 2016.09.13
  Added [:lmax+1] to getCovar calls; ZK, 2016.09.14
  Switched to useLensing=1 in loadCls; ZK, 2016.10.07

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
from legprodint import getJmn
import sim_stats as sims # for getSMICA and getCovar


def mf():
  """
  Purpose:
    
  Procedure:
    
  Inputs:

  Returns:
    
  """

  pass




################################################################################
# testing code

def test(useCLASS=1,useLensing=1,classCamb=1,nSims=1000,lmax=3,lmin=2,
         newSMICA=True,newDeg=False):
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
        default: 3
      lmin: the lowest l to include in S_{1/2} = CIC calculations
        default: 2
      newSMICA: set to True to recalculate SMICA results
        default: True
      newDeg: set to True to recalculate map and mask degredations
        default: False
  """

  ##############################################################################
  # load theoretical power spectra

  # load data
  ell,fullCl,primCl,lateCl,crossCl = gcp.loadCls(useCLASS=useCLASS,useLensing=useLensing,classCamb=classCamb)

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

  # extract the part I want
  myL  = ell[:lmax]
  myCl = fullCl[:lmax]

  ##############################################################################
  # load SMICA data and filter out all but low-l a_lm.s

  theta_i = 0.0 #degrees
  theta_f = 180.0 #degrees
  nSteps = 1800
  # default: lmax=3,lmin=2
  #newSMICA = True # so I don't use lmax=100 from previous calc.
  thetaArray2sp, C_SMICAsp, C_SMICAmaskedsp, S_SMICAnomasksp, S_SMICAmaskedsp = \
    sims.getSMICA(theta_i=theta_i,theta_f=theta_f,nSteps=nSteps,lmax=lmax,lmin=lmin,
             newSMICA=newSMICA,newDeg=newDeg,useSPICE=True)

  ##############################################################################
  # create ensemble of realizations and gather statistics

  covEnsembleFull  = np.zeros([nSims,nSteps+1]) # for maskless
  covEnsembleCut   = np.zeros([nSims,nSteps+1]) # for masked
  sEnsembleFull    = np.zeros(nSims)
  sEnsembleCut     = np.zeros(nSims)
  covTheta = np.array([])

  # get Jmn matrix for harmonic space S_{1/2} calc.
  myJmn = getJmn(lmax=lmax)


  doTime = True # to time the run and print output
  startTime = time.time()
  for nSim in range(nSims):
    print 'starting sim ',nSim+1, ' of ',nSims
    alm_prim,alm_late = hp.synalm((primCl,lateCl,crossCl),lmax=lmax,new=True)

    # calculate C(theta) of simulation
    Clsim_prim = hp.alm2cl(alm_prim)
    Clsim_late = hp.alm2cl(alm_late)
    Clsim_cros = hp.alm2cl(alm_prim,alm_late)
    Clsim_full = Clsim_prim + 2*Clsim_cros + Clsim_late
    # use Cl_sim_full to omit prim/late distinction for now
    #Clsim_full_sum += Clsim_full

    # first without mask  
    #   note: getCovar uses linspace in x for thetaArray
    thetaArray,cArray = sims.getCovar(ell[:lmax+1],Clsim_full[:lmax+1],theta_i=theta_i,
                                  theta_f=theta_f,nSteps=nSteps,lmin=lmin)
    covEnsembleFull[nSim] = cArray
    covTheta = thetaArray

    # S_{1/2}
    sEnsembleFull[nSim] = np.dot(Clsim_full[lmin:],np.dot(myJmn[lmin:,lmin:],Clsim_full[lmin:]))

    # now with a mask
    # should have default RING ordering
    # pixel window and beam already accounted for in true Cls
    #mapSim = hp.alm2map(alm_prim+alm_late,myNSIDE,lmax=lmax,pixwin=True,sigma=5./60*np.pi/180)
    mapSim = hp.alm2map(alm_prim+alm_late,myNSIDE,lmax=lmax)

    # super lame that spice needs to read/write from disk, but here goes...
    mapTempFile = 'tempMap.fits'
    ClTempFile  = 'tempCl.fits'
    maskDegFile = 'maskMapDeg.fits' # this should have been created by sims.getSMICA
    hp.write_map(mapTempFile,mapSim)
    ispice(mapTempFile,ClTempFile,maskfile1=maskDegFile,subav="YES",subdipole="YES")
    Cl_masked = hp.read_cl(ClTempFile)
    ell2 = np.arange(Cl_masked.shape[0])
    #   note: getCovar uses linspace in x for thetaArray
    thetaArray,cArray2 = sims.getCovar(ell2[:lmax+1],Cl_masked[:lmax+1],theta_i=theta_i,
                                   theta_f=theta_f,nSteps=nSteps,lmin=lmin)
    covEnsembleCut[nSim] = cArray2
    
    # S_{1/2}
    sEnsembleCut[nSim] = np.dot(Cl_masked[lmin:lmax+1],np.dot(myJmn[lmin:,lmin:],
                                Cl_masked[lmin:lmax+1]))

    doPlot = False#True
    if doPlot:
      plt.plot(thetaArray,cArray)
      plt.xlabel('theta (degrees)')
      plt.ylabel('C(theta)')
      plt.title('covariance of simulated CMB')
      plt.show()


  if doTime: print 'time elapsed: ',int((time.time()-startTime)/60.),' minutes'
  avgEnsembleFull = np.average(covEnsembleFull, axis = 0)
  stdEnsembleFull = np.std(covEnsembleFull, axis = 0)
  # do I need a better way to describe confidence interval?
  avgEnsembleCut = np.average(covEnsembleCut, axis = 0)
  stdEnsembleCut = np.std(covEnsembleCut, axis = 0)

  #Clsim_full_avg = Clsim_full_sum / nSims

  ##############################################################################
  # plot/print results

  doPlot = True
  if doPlot:

    print 'plotting correlation functions... '
    # first the whole sky statistics
    plt.plot(thetaArray,avgEnsembleFull,label='sim. ensemble average (no mask)')
    plt.fill_between(thetaArray,avgEnsembleFull+stdEnsembleFull,
                     avgEnsembleFull-stdEnsembleFull,alpha=0.25,
                     label='simulation 1sigma envelope')
    #plt.plot(thetaArray2,C_SMICA,label='SMICA R2 (inpainted,anafast)')
    plt.plot(thetaArray2sp,C_SMICAsp,label='SMICA R2 (inpainted,spice)')
    
    plt.xlabel('theta (degrees)')
    plt.ylabel('C(theta)')
    plt.title('whole sky covariance of '+str(nSims)+' simulated CMBs, lmax='+str(lmax))
    plt.ylim([-500,1000])
    plt.plot([0,180],[0,0]) #horizontal line
    plt.legend()
    plt.show()

    # now the cut sky
    plt.plot(thetaArray,avgEnsembleCut,label='sim. ensemble average (masked)')
    plt.fill_between(thetaArray,avgEnsembleCut+stdEnsembleCut,
                     avgEnsembleCut-stdEnsembleCut,alpha=0.25,
                     label='simulation 1sigma envelope')
    #plt.plot(thetaArray2,C_SMICAmasked,label='SMICA R2 (masked ,anafast)')
    plt.plot(thetaArray2sp,C_SMICAmaskedsp,label='SMICA R2 (masked ,spice)')
    
    plt.xlabel('theta (degrees)')
    plt.ylabel('C(theta)')
    plt.title('cut sky covariance of '+str(nSims)+' simulated CMBs, lmax='+str(lmax))
    plt.ylim([-500,1000])
    plt.plot([0,180],[0,0]) #horizontal line
    plt.legend()
    plt.show()

    print 'plotting S_{1/2} distributions... '
    myBins = np.logspace(2,7,100)
    plt.axvline(x=S_SMICAnomasksp,color='b',linewidth=3,label='SMICA inpainted')
    plt.axvline(x=S_SMICAmaskedsp,color='g',linewidth=3,label='SMICA masked')
    plt.hist(sEnsembleFull,bins=myBins,histtype='step',label='full sky')
    plt.hist(sEnsembleCut, bins=myBins,histtype='step',label='cut sky')

    plt.gca().set_xscale("log")
    plt.legend()
    plt.xlabel('S_{1/2} (microK^4)')
    plt.ylabel('Counts')
    plt.title('S_{1/2} of '+str(nSims)+' simulated CMBs')
    plt.show()


  # S_{1/2} output
  print ''
  print 'using CIC method: '
  #print 'S_{1/2}(anafast): SMICA, no mask: ',S_SMICAnomask,', masked: ',S_SMICAmasked
  print 'S_{1/2}(spice): SMICA, no mask: ',S_SMICAnomasksp,', masked: ',S_SMICAmaskedsp
  print ''
  #print 'using CCdx method: '
  #print 'S_{1/2}(anafast): SMICA, no mask: ',SSnm2,', masked: ',SSmd2
  #print 'S_{1/2}(spice): SMICA, no mask: ',SSnm2sp,', masked: ',SSmd2sp
  #print ''





  ##############################################################################
  # step 3
  print 'step 3: profit'

if __name__=='__main__':
  test()


