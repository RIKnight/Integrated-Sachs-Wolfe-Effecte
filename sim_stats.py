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
  legprodint.py (legendre product integral)
  ramdisk.sh (creates and deletes RAM disks)
Inputs:
  Data files as specified in get_crosspower.loadCls function
Outputs:

Modification History:
  Written by Z Knight, 2016.06.28
  Fixed x,theta conversion error; ZK, 2016.07.01
  Added spice functionality; ZK, 2016.07.14
  Added legprodint, S_{1/2}; ZK, 2016.07.22
  Upgraded the getSMICA map degradation procedure to use harmonic space
    pixel window scaling; ZK, 2016.08.16
  Added C_l ensemble average; ZK, 2016.08.26
  Modified treatment of SPICE output to be C_l, not psuedo-C_l, which brought
    SPICE C_l results in line with anafast psuedo-C_l results; ZK, 2016.08.31 
  Added S_{1/2} histogram; ZK, 2016.08.31
  Added subav="YES",subdipole="YES" to ispice calls; ZK, 2016.09.01 
  Fixed indexing problem for Cl in hp.synalm; ZK, 2016.09.07
  Added lmin to CIC calculations; ZK, 2016.09.07
  Added lmin to C(theta) plotting; ZK, 2016.09.08
  Added option to suppress C2 in sims; ZK, 2016.09.13
  Switched useSPICE default to True in getSMICA; ZK, 2016.09.13
  Added [:lmax+1] to getCovar calls; ZK, 2016.09.14
  Fixed int warning on index in getCovar; 
    Switched to useLensing=1 in loadCls; ZK, 2016.10.07
  Added suppressC2,suppFactor,filterC2,filtFactor to test function
    parameter list; implemented filterC2; 
    added ramdisk functionality for SpICE speed; ZK, 2016.10.21
  Modified plotting programs with LATEX; ZK, 2016.11.07
  Modified for filtFactor range: filtFacLow and filtfacHigh; 
    commented out anafast section of test function; ZK, 2016.11.10
  Added R1 option to getSMICA; ZK, 2016.11.16
  Removed titles from plots; ZK, 2016.11.17
  Made bigger axis labels on plots for publication; ZK, 2017.04.30

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
import subprocess # for calling RAM Disk scripts


################################################################################
# plotting

def combPlots(saveFile1="simStatResultC_10000.npy",saveFile2="simStatC_SMICA.npy",
              saveFile3="simStatResultS_10000.npy",
              saveFile4="simStatResultC_10000_C2filtered.npy",
              saveFile5="simStatResultS_10000_C2filtered.npy",lmax=100):
  """
  name:
    combPlots
  purpose:
    combine nonfiltered and filtered, masked and unmasked results
    plot them
  inputs:
    ...describe these please
    lmax: same as in makePlots
  """

  # load results
  mySimStatResultC = np.load(saveFile1)
  myC_SMICA        = np.load(saveFile2)
  mySimStatResultS = np.load(saveFile3)
  mySimStatResultC_filt = np.load(saveFile4)
  mySimStatResultS_filt = np.load(saveFile5)

  thetaArray2sp   = myC_SMICA[0]
  C_SMICAsp       = myC_SMICA[1]
  C_SMICAmaskedsp = myC_SMICA[2]

  # nonfiltered
  thetaArray      = mySimStatResultC[0]
  avgEnsembleFull = mySimStatResultC[1]
  stdEnsembleFull = mySimStatResultC[2]
  avgEnsembleCut  = mySimStatResultC[3]
  stdEnsembleCut  = mySimStatResultC[4]

  sEnsembleFull   = mySimStatResultS[0]
  sEnsembleCut    = mySimStatResultS[1]

  # filtered
  thetaArray_filt      = mySimStatResultC_filt[0]
  avgEnsembleFull_filt = mySimStatResultC_filt[1]
  stdEnsembleFull_filt = mySimStatResultC_filt[2]
  avgEnsembleCut_filt  = mySimStatResultC_filt[3]
  stdEnsembleCut_filt  = mySimStatResultC_filt[4]

  sEnsembleFull_filt   = mySimStatResultS_filt[0]
  sEnsembleCut_filt    = mySimStatResultS_filt[1]

  nSims = sEnsembleCut.size -1

  # do the plotting
  print 'plotting correlation functions... '

  # first the unfiltered results
  # first the whole sky statistics
  plt.plot(thetaArray,avgEnsembleFull,label='sim. ensemble average (no mask)',color='b')
  plt.fill_between(thetaArray,avgEnsembleFull+stdEnsembleFull,
                   avgEnsembleFull-stdEnsembleFull,alpha=0.25,
                   label='simulation 1sigma envelope',color='b')
  #plt.plot(thetaArray2,C_SMICA,label='SMICA R2 (inpainted,anafast)')
  #plt.plot(thetaArray2sp,C_SMICAsp,label='SMICA R2 (inpainted,spice)')
  plt.plot(thetaArray2sp,C_SMICAsp,label='SMICA R2 (inpainted)',linestyle='-.',linewidth=2)

  #plt.xlabel('theta (degrees)')
  #plt.ylabel('C(theta)')
  #plt.title('whole sky covariance of '+str(nSims)+' simulated CMBs, lmax='+str(lmax))
  #plt.ylim([-500,1000])
  #plt.plot([0,180],[0,0]) #horizontal line
  #plt.legend()
  #plt.show()

  # now the cut sky
  plt.plot(thetaArray,avgEnsembleCut,label='sim. ensemble average (masked)',color='g')
  plt.fill_between(thetaArray,avgEnsembleCut+stdEnsembleCut,
                   avgEnsembleCut-stdEnsembleCut,alpha=0.25,
                   label='simulation 1sigma envelope',color='g')
  #plt.plot(thetaArray2,C_SMICAmasked,label='SMICA R2 (masked ,anafast)')
  #plt.plot(thetaArray2sp,C_SMICAmaskedsp,label='SMICA R2 (masked ,spice)')
  plt.plot(thetaArray2sp,C_SMICAmaskedsp,label='SMICA R2 (masked)',linestyle='--',linewidth=2)
  
  plt.xlabel('theta (degrees)')
  plt.ylabel('C(theta)')
  #plt.title('cut-sky covariance of '+str(nSims)+' simulated CMBs, lmax='+str(lmax))
  plt.ylim([-500,1000])
  plt.plot([0,180],[0,0],color='k') #horizontal line
  plt.legend()
  plt.show()


  # now the filtered results
  # first the whole sky statistics
  plt.plot(thetaArray,avgEnsembleFull_filt,label='sim. ensemble average (no mask)',color='b')
  plt.fill_between(thetaArray,avgEnsembleFull_filt+stdEnsembleFull_filt,
                   avgEnsembleFull_filt-stdEnsembleFull_filt,alpha=0.25,
                   label='simulation 1sigma envelope',color='b')
  #plt.plot(thetaArray2,C_SMICA,label='SMICA R2 (inpainted,anafast)')
  #plt.plot(thetaArray2sp,C_SMICAsp,label='SMICA R2 (inpainted,spice)')
  plt.plot(thetaArray2sp,C_SMICAsp,label='SMICA R2 (inpainted)',linestyle='-.',linewidth=2)

  #plt.xlabel('theta (degrees)')
  #plt.ylabel('C(theta)')
  #plt.title('whole-sky covariance of '+str(nSims)+' simulated CMBs, lmax='+str(lmax))
  #plt.ylim([-500,1000])
  #plt.plot([0,180],[0,0]) #horizontal line
  #plt.legend()
  #plt.show()

  # now the cut sky
  plt.plot(thetaArray,avgEnsembleCut_filt,label='sim. ensemble average (masked)',color='g')
  plt.fill_between(thetaArray,avgEnsembleCut_filt+stdEnsembleCut_filt,
                   avgEnsembleCut_filt-stdEnsembleCut_filt,alpha=0.25,
                   label='simulation 1sigma envelope',color='g')
  #plt.plot(thetaArray2,C_SMICAmasked,label='SMICA R2 (masked ,anafast)')
  #plt.plot(thetaArray2sp,C_SMICAmaskedsp,label='SMICA R2 (masked ,spice)')
  plt.plot(thetaArray2sp,C_SMICAmaskedsp,label='SMICA R2 (masked)',linestyle='--',linewidth=2)
  
  plt.xlabel('theta (degrees)')
  plt.ylabel('C(theta)')
  #plt.title('cut-sky covariance of '+str(nSims)+' simulated CMBs, lmax='+str(lmax))
  plt.ylim([-500,1000])
  plt.plot([0,180],[0,0],color='k') #horizontal line
  plt.legend()
  plt.show()



def makePlots(saveFile1="simStatResultC.npy",saveFile2="simStatC_SMICA.npy",
              saveFile3="simStatResultS.npy",lmax=100):
  """
  name:
    makePlots
  purpose:
    plotting results of sim_stats and printing p-values
  inputs:
    saveFile1,saveFile2,saveFile3... describe these please

    lmax: should be what was used to create results
      (would be better if it were in the save file but it's not)
      Default: 100
  outputs:
    returns p-values for full-sky, cut-sky
    prints p-values for S_1/2
  """
  # load results
  mySimStatResultC = np.load(saveFile1)
  myC_SMICA        = np.load(saveFile2)
  mySimStatResultS = np.load(saveFile3)

  thetaArray      = mySimStatResultC[0]
  avgEnsembleFull = mySimStatResultC[1]
  stdEnsembleFull = mySimStatResultC[2]
  avgEnsembleCut  = mySimStatResultC[3]
  stdEnsembleCut  = mySimStatResultC[4]

  thetaArray2sp   = myC_SMICA[0]
  C_SMICAsp       = myC_SMICA[1]
  C_SMICAmaskedsp = myC_SMICA[2]

  sEnsembleFull   = mySimStatResultS[0]
  sEnsembleCut    = mySimStatResultS[1]

  nSims = sEnsembleCut.size -1


  # do the plotting
  print 'plotting correlation functions... '
  # first the whole sky statistics
  plt.plot(thetaArray,avgEnsembleFull,label='sim. ensemble average (no mask)',linestyle='--',linewidth=2)
  plt.fill_between(thetaArray,avgEnsembleFull+stdEnsembleFull,
                   avgEnsembleFull-stdEnsembleFull,alpha=0.25,
                   label="simulation $1\sigma$ envelope")
  #plt.plot(thetaArray2,C_SMICA,label='SMICA PR2 (inpainted,anafast)')
  #plt.plot(thetaArray2sp,C_SMICAsp,label='SMICA PR2 (inpainted,spice)')
  plt.plot(thetaArray2sp,C_SMICAsp,label='SMICA PR2 (inpainted)',linewidth=2)

  myfs = 16 # font size for labels
  plt.xlabel(r'Angular Separation $\theta$ (degrees)',fontsize=myfs)
  plt.ylabel(r'$C(\theta) (\mu K^2)$',fontsize=myfs)
  #plt.title('whole-sky covariance of '+str(nSims)+' simulated CMBs, lmax='+str(lmax))
  plt.ylim([-500,1000])
  plt.plot([0,180],[0,0],color='k') #horizontal line
  plt.legend()
  plt.show()

  # now the cut sky
  plt.plot(thetaArray,avgEnsembleCut,label='sim. ensemble average (masked)',linestyle='--',linewidth=2)
  plt.fill_between(thetaArray,avgEnsembleCut+stdEnsembleCut,
                   avgEnsembleCut-stdEnsembleCut,alpha=0.25,
                   label='simulation $1\sigma$ envelope')
  #plt.plot(thetaArray2,C_SMICAmasked,label='SMICA PR2 (masked ,anafast)')
  #plt.plot(thetaArray2sp,C_SMICAmaskedsp,label='SMICA PR2 (masked ,spice)')
  plt.plot(thetaArray2sp,C_SMICAmaskedsp,label='SMICA PR2 (masked)',linewidth=2)
  
  #myfs = 16 # font size for labels
  plt.xlabel(r'Angular Separation $\theta$ (degrees)',fontsize=myfs)
  plt.ylabel(r'$C(\theta) (\mu K^2)$',fontsize=myfs)
  #plt.title('cut-sky covariance of '+str(nSims)+' simulated CMBs, lmax='+str(lmax))
  plt.ylim([-500,1000])
  plt.plot([0,180],[0,0],color='k') #horizontal line
  plt.legend()
  plt.show()

  print 'plotting S_{1/2} distributions... '
  #myBins = np.logspace(2,7,100)
  myBins = np.logspace(2,6,100)
  """
  # whole sky
  plt.axvline(x=sEnsembleFull[0],color='b',linewidth=3,label='SMICA inpainted')
  #plt.axvline(x=sEnsembleCut[0] ,color='g',linewidth=3,label='SMICA masked')
  plt.hist(sEnsembleFull[1:],bins=myBins,color='b',histtype='step',label='full-sky sims')
  #plt.hist(sEnsembleCut[1:], bins=myBins,color='g',histtype='step',label='cut-sky sims')

  plt.gca().set_xscale("log")
  plt.legend()
  plt.xlabel(r'$S_{1/2} (\mu K^4)$')
  plt.ylabel('Counts')
  #plt.title(r'$S_{1/2}$ of '+str(nSims)+' simulated CMBs')
  plt.xlim((500,10**6))
  plt.show()
  """

  # cut sky
  #plt.axvline(x=sEnsembleFull[0],color='b',linewidth=3,label='SMICA inpainted')
  plt.axvline(x=sEnsembleCut[0] ,color='g',linewidth=3,label='SMICA masked')
  #plt.hist(sEnsembleFull[1:],bins=myBins,color='b',histtype='step',label='full-sky sims')
  plt.hist(sEnsembleCut[1:], bins=myBins,color='g',histtype='step',label='cut-sky sims')

  plt.gca().set_xscale("log")
  plt.legend()
  #myfs = 16 # font size for labels
  plt.xlabel(r'$S_{1/2} (\mu K^4)$',fontsize=myfs)
  plt.ylabel('Counts',fontsize=myfs)
  #plt.title(r'$S_{1/2}$ of '+str(nSims)+' simulated CMBs')
  plt.xlim((500,10**6))
  plt.show()

  # whole and cut together
  plt.axvline(x=sEnsembleFull[0],color='b',linewidth=3,label='SMICA inpainted')
  plt.axvline(x=sEnsembleCut[0] ,color='g',linewidth=3,label='SMICA masked')
  plt.hist(sEnsembleFull[1:],bins=myBins,color='b',histtype='step',label='full-sky sims')
  plt.hist(sEnsembleCut[1:], bins=myBins,color='g',histtype='step',label='cut-sky sims')

  plt.gca().set_xscale("log")
  plt.legend()
  #myfs = 16 # font size for labels
  plt.xlabel(r'$S_{1/2} (\mu K^4)$',fontsize=myfs)
  plt.ylabel('Counts',fontsize=myfs)
  #plt.title(r'$S_{1/2}$ of '+str(nSims)+' simulated CMBs')
  plt.xlim((500,10**6))
  plt.show()


  # calculate p-values
  nUnderFull = 0 # also will include nEqual
  nUnderCut  = 0 # also will include nEqual
  nOverFull  = 0
  nOverCut   = 0

  thresholdFull = sEnsembleFull[0]
  thresholdCut  = sEnsembleCut[0]

  for nSim in range(nSims-1): # -1 due to SMICA in 0 position
    if sEnsembleFull[nSim+1] > thresholdFull:
      nOverFull  +=1
    else:
      nUnderFull +=1
    if sEnsembleCut[nSim+1] > thresholdCut:
      nOverCut   +=1
    else:
      nUnderCut  +=1
  pValFull = nUnderFull/float(nUnderFull+nOverFull)
  pValCut  = nUnderCut /float(nUnderCut +nOverCut )

  print 'S_{1/2} full-sky: ',sEnsembleFull[0]
  print 'S_{1/2}  cut-sky: ',sEnsembleCut[0]
  print 'P-value full-sky: ',pValFull
  print 'P-value  cut-sky: ',pValCut

  return pValFull,pValCut

################################################################################
# the main functions: getCovar, getSMICA, and SOneHalf


def getCovar(ell,Cl,theta_i=0.0,theta_f=180.0,nSteps = 1800,doTime=False,lmin=0):
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
    lmin: the minimum l value to include in conversion
      Default: 0
  Outputs:
    theta, the theta values for C(theta) array
    C(theta), the covariance array

  """

  #plt.plot(ell,Cl)
  #plt.show()

  startTime = time.time()

  # fill beginning with zeros
  startEll = int(ell[0])
  ell = np.append(np.arange(startEll),ell)
  Cl  = np.append(np.zeros(startEll),Cl)

  # limit low l powers if needed
  #for zilch in range(lmin):
  # kludge for removal of low power dip
  for zilch in range(lmin):#+range(20,28):
    Cl[zilch] = 0

  # create legendre coefficients
  legCoef = (2*ell+1)/(4*np.pi) * Cl
  
  # create x values for P(x)
  x_i = np.cos(theta_i*np.pi/180.)
  x_f = np.cos(theta_f*np.pi/180.)
  xArray = np.linspace(x_i,x_f,num=nSteps+1)

  # evalueate legendre polynomials at x values with coefficients
  covar = legval(xArray,legCoef)

  if doTime: print 'time elapsed: ',int((time.time()-startTime)/60),' minutes'

  #plt.plot(np.arccos(xArray)*180./np.pi,covar)
  #plt.show()

  return np.arccos(xArray)*180./np.pi,covar


def getSMICA(theta_i=0.0,theta_f=180.0,nSteps=1800,lmax=100,lmin=2,
             newSMICA=False,useSPICE=True,newDeg=False,R1=False):
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
    lmin: the lowest l to use in C(theta,Cl) and S_{1/2} = CIC calculation
    newSMICA: set to True to reload data from files and recompute
      if False, will load C(theta) curves from file
    useSPICE: if True, will use SPICE to find power spectra
      if False, will use anafast, following Copi et. al. 2013
      Default: True
    newDeg: set to True to recalculate map and mask degredations
      Note: the saved files are dependent on the value of lmax that was used
      Default: False
    R1: set to True to use R1 versions of SMICA and mask.  Otherwise, R2 is used
      Only affects which Planck files are used; irrelevant if newDeg=False.
      Default: False
  Outupts:
    theta: nSteps+1 angles that C(theta) arrays are for (degrees)
    unmasked: C(theta) unmasked (microK^2)
    masked: C(theta) masked     (microK^2)

  """
  saveFile  = 'getSMICAfile.npy'  #for anafast
  saveFile2 = 'getSMICAfile2.npy' #for spice
  if newSMICA:
    # start with map degredations
    mapDegFile = 'smicaMapDeg.fits'
    maskDegFile = 'maskMapDeg.fits'
    if newDeg:
      # load maps; default files have 2048,NESTED,GALACTIC
      dataDir   = '/Data/'
      if R1:
        smicaFile = 'COM_CompMap_CMB-smica-field-I_2048_R1.20.fits'
        maskFile  = 'COM_Mask_CMB-union_2048_R1.10.fits'
      else:
        smicaFile = 'COM_CMB_IQU-smica-field-int_2048_R2.01_full.fits'
        maskFile  = 'COM_CMB_IQU-common-field-MaskInt_2048_R2.01.fits'
      print 'opening file ',smicaFile,'... '
      smicaMap,smicaHead = hp.read_map(dataDir+smicaFile,nest=True,h=True)
      print 'opening file ',maskFile,'... '
      maskMap, maskHead  = hp.read_map(dataDir+maskFile, nest=True,h=True)
      if R1:
        smicaMap *= 1e-6 #microK to K

      # degrade map and mask resolutions from 2048 to 128; convert NESTED to RING
      useAlm = True # set to True to do harmonic space scaling, False for ud_grade
      NSIDE_big = 2048
      NSIDE_deg = 128
      while 4*NSIDE_deg < lmax:
        NSIDE_deg *=2
      print 'resampling maps at NSIDE = ',NSIDE_deg,'... '
      order_out = 'RING'
      if useAlm:
        # transform to harmonic space
        smicaMapRing = hp.reorder(smicaMap,n2r=True)
        maskMapRing  = hp.reorder(maskMap,n2r=True)
        smicaCl,smicaAlm = hp.anafast(smicaMapRing,alm=True,lmax=lmax)
        maskCl, maskAlm  = hp.anafast(maskMapRing, alm=True,lmax=lmax)
          # this gives 101 Cl values and 5151 Alm values.  Why not all 10201 Alm.s?

        # scale by pixel window functions
        bigWin = hp.pixwin(NSIDE_big)
        degWin = hp.pixwin(NSIDE_deg)
        winRatio = degWin/bigWin[:degWin.size]
        degSmicaAlm = hp.almxfl(smicaAlm,winRatio)
        degMaskAlm  = hp.almxfl(maskAlm, winRatio)
        
        # re-transform back to real space
        smicaMapDeg = hp.alm2map(degSmicaAlm,NSIDE_deg)
        maskMapDeg  = hp.alm2map(degMaskAlm, NSIDE_deg)

      else:
        smicaMapDeg = hp.ud_grade(smicaMap,nside_out=NSIDE_deg,order_in='NESTED',order_out=order_out)
        maskMapDeg  = hp.ud_grade(maskMap, nside_out=NSIDE_deg,order_in='NESTED',order_out=order_out)
        # note: degraded resolution mask will no longer be only 0s and 1s.
        #   Should it be?  Yes.

      # turn smoothed mask back to 0s,1s mask
      threshold = 0.9
      maskMapDeg[np.where(maskMapDeg >  threshold)] = 1
      maskMapDeg[np.where(maskMapDeg <= threshold)] = 0

      #testing
      #hp.mollview(smicaMapDeg)
      #plt.show()
      #hp.mollview(maskMapDeg)
      #plt.show()
      #return 0

      hp.write_map(mapDegFile,smicaMapDeg,nest=False) # use False if order_out='RING' above
      hp.write_map(maskDegFile,maskMapDeg,nest=False)

    else: # just load previous degradations (dependent on previous lmax)
      print 'loading previously degraded map and mask...'
      smicaMapDeg = hp.read_map( mapDegFile,nest=False)
      maskMapDeg  = hp.read_map(maskDegFile,nest=False)


    # find power spectra
    print 'find power spectra... '
    if useSPICE:
      ClFile1 = 'spiceCl_unmasked.fits'
      ClFile2 = 'spiceCl_masked.fits'

      # note: lmax for spice is 3*NSIDE-1 or less
      ispice(mapDegFile,ClFile1,subav="YES",subdipole="YES")
      Cl_unmasked = hp.read_cl(ClFile1)
      ispice(mapDegFile,ClFile2,maskfile1=maskDegFile,subav="YES",subdipole="YES")
      Cl_masked = hp.read_cl(ClFile2)
      Cl_mask = np.zeros(Cl_unmasked.shape[0]) # just a placeholder
      ell    = np.arange(Cl_unmasked.shape[0])

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

    # Legendre transform to real space
    print 'Legendre transform to real space... '
    # note: getCovar uses linspace in x for thetaArray
    thetaDomain,CofTheta      = getCovar(ell[:lmax+1],Cl_unmasked[:lmax+1],theta_i=theta_i,
                                    theta_f=theta_f,nSteps=nSteps,lmin=lmin)
    thetaDomain,CCutofThetaTA = getCovar(ell[:lmax+1],Cl_masked[:lmax+1],  theta_i=theta_i,
                                    theta_f=theta_f,nSteps=nSteps,lmin=lmin)
    CofTheta      *= 1e12 # K^2 to microK^2
    CCutofThetaTA *= 1e12 # K^2 to microK^2
    
    if useSPICE:
      CCutofTheta     = CCutofThetaTA#/(4*np.pi)
    else:
      thetaDomain,AofThetaInverse = getCovar(ell[:lmax+1],Cl_mask[:lmax+1],theta_i=theta_i,
                                    theta_f=theta_f,nSteps=nSteps,lmin=0) # don't zilch the mask
      # note: zilching the mask's low power drastically changed C(theta) for masked anafast  
      #   Not sure why.
      CCutofTheta     = CCutofThetaTA/AofThetaInverse

    xArray            = np.cos(thetaDomain*np.pi/180.)

    # back to frequency space for S_{1/2} = CIC calculation
    if useSPICE:
      CCutofL = Cl_masked[:lmax+1]*1e12 #K^2 to microK^2
    else:
      legCoefs = legfit(xArray,CCutofTheta,lmax)
      CCutofL = legCoefs*(4*np.pi)/(2*ell[:lmax+1]+1)

    # S_{1/2}
    myJmn = getJmn(lmax=lmax)
    SMasked = np.dot(CCutofL[lmin:],np.dot(myJmn[lmin:,lmin:],CCutofL[lmin:]))
    SNoMask = np.dot(Cl_unmasked[lmin:lmax+1],np.dot(myJmn[lmin:,lmin:],
          Cl_unmasked[lmin:lmax+1]))*1e24 #two factors of K^2 to muK^2

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
  #nTerms = 250
  xArray = np.linspace(-1.0,0.5,nTerms+1) #use all but the first one
  dx = 1.5 / nTerms
  mySum = 0.0
  for x in xArray[1:]:
    mySum += C(x)**2*dx
  return mySum



################################################################################
# testing code

def test(useCLASS=1,useLensing=1,classCamb=1,nSims=1000,lmax=100,lmin=2,
         newSMICA=False,newDeg=False,suppressC2=False,suppFactor=0.23,
         filterC2=False,filtFacLow=0.1,filtFacHigh=0.2,R1=False):
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
      R1: set to True to use SMICA and Mask R1.  Otherwise, R2 used.
        Only affects calculation of newly degraded map.
        Default: False
  """

  # load data
  ell,fullCl,primCl,lateCl,crossCl = gcp.loadCls(useCLASS=useCLASS,useLensing=useLensing,classCamb=classCamb)

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

  """ # don't want anafast after all

  # get unmasked and masked SMICA covariances
  #   note: getSMICA uses linspace in theta for thetaArray
  #newSMICA = False#True
  thetaArray2, C_SMICA, C_SMICAmasked, S_SMICAnomask, S_SMICAmasked = \
    getSMICA(theta_i=theta_i,theta_f=theta_f,nSteps=nSteps,lmax=lmax,lmin=lmin,
             newSMICA=newSMICA,newDeg=newDeg,useSPICE=False,R1=R1)
  print ''
  print 'S_{1/2}(anafast): SMICA, no mask: ',S_SMICAnomask,', masked: ',S_SMICAmasked
  print ''
  """


  # get C_l from SPICE to compare to above method
  #   note: getSMICA uses linspace in theta for thetaArray
  #newSMICA = False#True
  thetaArray2sp, C_SMICAsp, C_SMICAmaskedsp, S_SMICAnomasksp, S_SMICAmaskedsp = \
    getSMICA(theta_i=theta_i,theta_f=theta_f,nSteps=nSteps,lmax=lmax,lmin=lmin,
             newSMICA=newSMICA,newDeg=newDeg,useSPICE=True,R1=R1)
  print ''
  print 'S_{1/2}(spice): SMICA, no mask: ',S_SMICAnomasksp,', masked: ',S_SMICAmaskedsp
  print ''

  # Find S_{1/2} in real space to compare methods
  nTerms = 10000
  #SSnm2   = SOneHalf(thetaArray2,  C_SMICA,         nTerms=nTerms)
  #SSmd2   = SOneHalf(thetaArray2,  C_SMICAmasked,   nTerms=nTerms)
  SSnm2sp = SOneHalf(thetaArray2sp,C_SMICAsp,       nTerms=nTerms)
  SSmd2sp = SOneHalf(thetaArray2sp,C_SMICAmaskedsp, nTerms=nTerms)


  # create ensemble of realizations and gather statistics
  covEnsembleFull  = np.zeros([nSims,nSteps+1]) # for maskless
  covEnsembleCut   = np.zeros([nSims,nSteps+1]) # for masked
  sEnsembleFull    = np.zeros(nSims)
  sEnsembleCut     = np.zeros(nSims)
  covTheta = np.array([])
  #nSims = 1000

  # apply beam and pixel window functions to power spectra
  #   note: to ignore the non-constant pixel shape, W(l) must be > B(l)
  #     however, this is not true for NSIDE=128 and gauss_beam(5')
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
  # note: i tried sims without this scaling, and results seemed the same at a glance

  # collect simulated Cl for comparison to model
  Clsim_full_sum = np.zeros(lmax+1)

  # get Jmn matrix for harmonic space S_{1/2} calc.
  myJmn = getJmn(lmax=lmax)

  # set up ramdisk for SpICE
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


  doTime = True # to time the run and print output
  startTime = time.time()
  #for nSim in range(nSims):
  nSim = 0
  while nSim < nSims:
    print 'starting sim ',nSim+1, ' of ',nSims
    alm_prim,alm_late = hp.synalm((primCl,lateCl,crossCl),lmax=lmax,new=True)

    # calculate C(theta) of simulation
    Clsim_prim = hp.alm2cl(alm_prim)
    Clsim_late = hp.alm2cl(alm_late)
    Clsim_cros = hp.alm2cl(alm_prim,alm_late)
    Clsim_full = Clsim_prim + 2*Clsim_cros + Clsim_late
    # use Cl_sim_full to omit prim/late distinction for now


    # start with a mask 
    #   -> for optional C2 filtering based on cut sky map
    #   alm2map should create map with default RING ordering
    #   pixel window and beam already accounted for in true Cls
    #mapSim = hp.alm2map(alm_prim+alm_late,myNSIDE,lmax=lmax,pixwin=True,sigma=5./60*np.pi/180)
    mapSim = hp.alm2map(alm_prim+alm_late,myNSIDE,lmax=lmax)

    hp.write_map(mapTempFile,mapSim)
    ispice(mapTempFile,ClTempFile,maskfile1=maskDegFile,subav="YES",subdipole="YES")
    Cl_masked = hp.read_cl(ClTempFile)
    ell2 = np.arange(Cl_masked.shape[0])
    
    # Check for low power of cut sky C_2
    if (filterC2 == True and fullCl[2]*filtFacHigh > Cl_masked[2]
                         and Cl_masked[2] > fullCl[2]*filtFacLow) or filterC2 == False:

      #   note: getCovar uses linspace in x for thetaArray
      thetaArray,cArray2 = getCovar(ell2[:lmax+1],Cl_masked[:lmax+1],theta_i=theta_i,
                                     theta_f=theta_f,nSteps=nSteps,lmin=lmin)
      covEnsembleCut[nSim] = cArray2
    
      # S_{1/2}
      sEnsembleCut[nSim] = np.dot(Cl_masked[lmin:lmax+1],np.dot(myJmn[lmin:,lmin:],Cl_masked[lmin:lmax+1]))

      doPlot = False#True
      if doPlot:
        plt.plot(thetaArray,cArray)
        plt.xlabel('theta (degrees)')
        plt.ylabel('C(theta)')
        plt.title('covariance of CMB simulation '+str(nSim+1))
        plt.show()


      # now without the mask
      # uses the same sims that passed the C2 filter
      Clsim_full_sum += Clsim_full
      
      #   note: getCovar uses linspace in x for thetaArray
      thetaArray,cArray = getCovar(ell[:lmax+1],Clsim_full[:lmax+1],theta_i=theta_i,
                                    theta_f=theta_f,nSteps=nSteps,lmin=lmin)
      covEnsembleFull[nSim] = cArray
      covTheta = thetaArray

      # S_{1/2}
      sEnsembleFull[nSim] = np.dot(Clsim_full[lmin:],np.dot(myJmn[lmin:,lmin:],Clsim_full[lmin:]))




      nSim +=1

  if doTime: print 'time elapsed: ',int((time.time()-startTime)/60.),' minutes'
  
  # free the RAM used by SpICE's RAM disk
  ramDiskOutput = subprocess.check_output('./ramdisk.sh delete '+diskID, shell=True)
  print ramDiskOutput


  avgEnsembleFull = np.average(covEnsembleFull, axis = 0)
  stdEnsembleFull = np.std(covEnsembleFull, axis = 0)
  # do I need a better way to describe confidence interval?
  avgEnsembleCut = np.average(covEnsembleCut, axis = 0)
  stdEnsembleCut = np.std(covEnsembleCut, axis = 0)

  Clsim_full_avg = Clsim_full_sum / nSims


  # save results
  saveFile1 = "simStatResultC.npy"
  np.save(saveFile1,np.vstack((thetaArray,avgEnsembleFull,stdEnsembleFull,
                               avgEnsembleCut,stdEnsembleCut)) )
  saveFile2 = "simStatC_SMICA.npy"
  np.save(saveFile2,np.vstack((thetaArray2sp,C_SMICAsp,C_SMICAmaskedsp)) )
  
  saveFile3 = "simStatResultS.npy"
  np.save(saveFile3,np.vstack(( np.hstack((np.array(S_SMICAnomasksp),sEnsembleFull)),
                                np.hstack((np.array(S_SMICAmaskedsp),sEnsembleCut)) )) )

  doPlot = True
  if doPlot:
    print 'plotting C_l... '
    #print ell.size,conv.size,primCl.size,crossCl.size,lateCl.size
    plt.plot(ell[:lmax+1],conv[:lmax+1]*(primCl+2*crossCl+lateCl)[:lmax+1],label='model D_l')
    plt.plot(ell[:lmax+1],conv[:lmax+1]*Clsim_full_avg,label='ensemble average D_l')
    plt.legend()
    plt.show()

    makePlots(saveFile1=saveFile1,saveFile2=saveFile2,saveFile3=saveFile3)

  # S_{1/2} output
  print ''
  print 'using CIC method: '
  #print 'S_{1/2}(anafast): SMICA, no mask: ',S_SMICAnomask,', masked: ',S_SMICAmasked
  print 'S_{1/2}(spice): SMICA, no mask: ',S_SMICAnomasksp,', masked: ',S_SMICAmaskedsp
  print ''
  print 'using CCdx method: '
  #print 'S_{1/2}(anafast): SMICA, no mask: ',SSnm2,', masked: ',SSmd2
  print 'S_{1/2}(spice): SMICA, no mask: ',SSnm2sp,', masked: ',SSmd2sp
  print ''

  #print 'S ensemble full: ',sEnsembleFull  
  #print 'S ensemble cut:  ',sEnsembleCut  



if __name__=='__main__':
  test()


