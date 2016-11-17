#! /usr/bin/env python
"""
Name:
  simple_Sonehalf
Purpose:
  create simplistic CMB simulations and analyze S_{1/2} properties
  shows trend when increasing l_min
Uses:
  healpy
  get_crosspower.py
  spice, ispice.py
  legprodint.py (legendre product integral)
  ramdisk.sh    (creates and deletes ramdisks)
Inputs:
  Data files as specified in get_crosspower.loadCls function
Outputs:
  creates plot of S_{1/2} distributions with varying values of l_min
Modification History:
  Written by Z Knight, 2016.09.07
  Added cutSky option; ZK, 2016.11.16
  Removed titles from plots; ZK, 2016.11.17
  
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time # for measuring duration
import subprocess # for calling RAM Disk scripts
import get_crosspower as gcp
from numpy.polynomial.legendre import legval # for C_l -> C(theta) conversion
from numpy.polynomial.legendre import legfit # for C(theta) -> C_l conversion
from ispice import ispice
from legprodint import getJmn
from scipy.interpolate import interp1d


def getSsim(ell,Cl,lmax=100,cutSky=False):
  """
  Purpose:
    create simulated S_{1/2} from input power spectrum
  Note:
    this calculates Jmn every time it is run so should not be used for ensembles
  Procedure:
    simulates full sky CMB, measures S_{1/2}
  Inputs:
    ell: the l values for the power spectrum
    Cl: the power spectrum
    lmax: the maximum ell value to use in calculation
      Default: 100
    cutSky: set to True to convert to real space, apply mask, etc.
      Default: False
      Note: true option not yet implemented
  Returns:
    simulated S_{1/2}
  """
  # get Jmn matrix for harmonic space S_{1/2} calc.
  myJmn = getJmn(lmax=lmax)[2:,2:] # do not include monopole, dipole

  #alm_prim,alm_late = hp.synalm((primCl,lateCl,crossCl),lmax=lmax,new=True)
  almSim = hp.synalm(Cl,lmax=lmax) # question: does this need to start at ell[0]=1?
  ClSim = hp.alm2cl(almSim)

  return np.dot(ClSim[2:],np.dot(myJmn,ClSim[2:]))




################################################################################
# testing code

def test(nSims=100,lmax=100,lmin=2,useCLASS=1,useLensing=1,cutSky=True,myNSIDE=128):
  """
    Purpose:
      function for testing S_{1/2} calculations
    Inputs:
      nSims: the number of simulations to do
      lmax: the highest l to use in the calculation
      lmin: the lowest l to use in the calculation
      useCLASS: set to 1 to use CLASS Cl, 0 for CAMB
      useLensing: set to 1 to use lensed Cls
      cutSky: set to True to do cut-sky sims
        Default: True
      myNSIDE: HEALPix parameter for simulated maps if cutSky=True
        Default: 128
  """
  # get power spectrum
  # starts with ell[0]=2
  ell,fullCl,primCl,lateCl,crossCl = gcp.loadCls(useCLASS=useCLASS,useLensing=useLensing)

  # fill beginning with zeros
  startEll = int(ell[0])
  ell = np.append(np.arange(startEll),ell)
  Cl  = np.append(np.zeros(startEll),fullCl)
  #conv = ell*(ell+1)/(2*np.pi)

  # get Jmn matrix for harmonic space S_{1/2} calc.
  myJmn = getJmn(lmax=lmax) # do not include monopole, dipole

  if cutSky:
    # yeah.. disk access is annoying so...
    RAMdisk     = '/Volumes/ramdisk/'
    ClTempFile  = RAMdisk+'tempCl.fits'
    mapTempFile = RAMdisk+'tempMap.fits'
    mapDegFile  = RAMdisk+'smicaMapDeg.fits' #created by sim_stats.getSMICA
    maskDegFile = RAMdisk+'maskMapDeg.fits'  #created by sim_stats.getSMICA

    # create RAM Disk for SpICE and copy these files there using bash
    RAMsize = 4 #Mb
    ramDiskOutput = subprocess.check_output('./ramdisk.sh create '+str(RAMsize), shell=True)
    print ramDiskOutput
    diskID = ramDiskOutput[31:41] # this might not grab the right part; works for '/dev/disk1'
    subprocess.call('cp smicaMapDeg.fits '+RAMdisk, shell=True)
    subprocess.call('cp maskMapDeg.fits ' +RAMdisk, shell=True)

    ispice(mapDegFile,ClTempFile,maskfile1=maskDegFile,subav="YES",subdipole="YES")
    Clsmica = hp.read_cl(ClTempFile)
  else:
    ClTempFile  = 'tempCl.fits'
    mapTempFile = 'tempMap.fits'
    mapDegFile  = 'smicaMapDeg.fits' #created by sim_stats.getSMICA 
    maskDegFile = 'maskMapDeg.fits'  #created by sim_stats.getSMICA
    ispice(mapDegFile,ClTempFile,subav="YES",subdipole="YES")
    Clsmica = hp.read_cl(ClTempFile)


  # collect results
  partialMax = 4#10 # must be more than lmin
  sEnsemblePartial = np.zeros([nSims,partialMax+1])
  for i in range(nSims):
    print "starting sim ",i+1," of ",nSims,"... "

    almSim = hp.synalm(Cl,lmax=lmax) # should start with ell[0] = 0
    if cutSky:
      mapSim = hp.alm2map(almSim,myNSIDE,lmax=lmax)
      hp.write_map(mapTempFile,mapSim)
      ispice(mapTempFile,ClTempFile,maskfile1=maskDegFile,subav="YES",subdipole="YES")
      ClSim = hp.read_cl(ClTempFile)
    else:  
      ClSim = hp.alm2cl(almSim)

    for myLmin in range(lmin,partialMax+1):
      sEnsemblePartial[i,myLmin] = np.dot(ClSim[myLmin:lmax+1],
                                np.dot(myJmn[myLmin:,myLmin:],ClSim[myLmin:lmax+1]))

  if cutSky:
    # free the RAM used by SpICE's RAM disk
    ramDiskOutput = subprocess.check_output('./ramdisk.sh delete '+diskID, shell=True)
    print ramDiskOutput


  # plot results
  print 'plotting S_{1/2} distributions... '
  #myBins = np.logspace(2,7,100)
  myBins = np.logspace(2,6,100)
  #plt.axvline(x=6763,color='b',linewidth=3,label='SMICA inpainted')
  #plt.axvline(x=2145,color='g',linewidth=3,label='SMICA masked')
  #plt.hist(sEnsembleFull,bins=myBins,color='b',histtype='step',label='full sky')
  #plt.hist(sEnsembleCut, bins=myBins,color='g',histtype='step',label='cut sky')

  myColors = ('g','b','r','c','m','k')#need more?  prob. not.
  myLines  = ('-','--','-.')#need more?
  for myEll in range(lmin,partialMax+1):
    plt.hist(sEnsemblePartial[:,myEll],bins=myBins,histtype='step',
        label=r'sims: $l_{\rm min}$ = '+str(myEll),
        color=myColors[myEll-lmin],linestyle=myLines[myEll-lmin])

    Sonehalf = np.dot(Clsmica[myEll:lmax+1],
                  np.dot(myJmn[myEll:,myEll:],Clsmica[myEll:lmax+1])) *1e24
    plt.axvline(x=Sonehalf,linewidth=3,label=r'SMICA: $l_{\rm min}$='+str(myEll),
        color=myColors[myEll-lmin],linestyle=myLines[myEll-lmin])

  plt.gca().set_xscale("log")
  plt.legend()
  plt.xlabel(r'$S_{1/2} (\mu K^4)$')
  plt.ylabel('Counts')
  plt.xlim((500,10**6))
  if cutSky:
    sName = ' cut-sky'
  else:
    sName = ' full-sky'
  #plt.title(r'$S_{1/2}$ of '+str(nSims)+sName+' simulated CMBs')
  plt.show()

if __name__=='__main__':
  test()


