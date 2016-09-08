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
  legprodint.py
Inputs:
  Data files as specified in get_crosspower.loadCls function
Outputs:
  creates plot of S_{1/2} distributions with varying values of l_min
Modification History:
  Written by Z Knight, 2016.09.07
  
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time # for measuring duration
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

def test(nSims=100,lmax=100,lmin=2,useCLASS=1,useLensing=1):
  """
    Purpose:
      function for testing S_{1/2} calculations
    Inputs:
      nSims: the number of simulations to do
      lmax: the highest l to use in the calculation
      lmin: the lowest l to use in the calculation
      useCLASS: set to 1 to use CLASS Cl, 0 for CAMB
      useLensing: set to 1 to use lensed Cls
  """
  # get power spectrum
  # starts with ell[0]=2
  ell,fullCl,primCl,lateCl,crossCl = gcp.loadCls(useCLASS=useCLASS,useLensing=useLensing)

  # fill beginning with zeros
  startEll = ell[0]
  ell = np.append(np.arange(startEll),ell)
  Cl  = np.append(np.zeros(startEll),fullCl)
  #conv = ell*(ell+1)/(2*np.pi)

  # get Jmn matrix for harmonic space S_{1/2} calc.
  myJmn = getJmn(lmax=lmax) # do not include monopole, dipole

  # collect results
  partialMax = 10 # must be more than lmin
  #sEnsembleFull    = np.zeros(nSims)
  sEnsemblePartial = np.zeros([nSims,partialMax+1])
  #sEnsembleCut     = np.zeros(nSims)
  for i in range(nSims):
    print "starting sim ",i+1," of ",nSims,"... "
    #sEnsembleFull[i] = getSsim(ell,fullCl,lmax=lmax)
    #sEnsembleCut[i]  = getSsim(ell,fullCl,lmax=lmax,cutSky=True)

    almSim = hp.synalm(Cl,lmax=lmax) # should start with ell[0] = 0
    ClSim = hp.alm2cl(almSim)

    #sEnsembleFull[i] = np.dot(ClSim[lmin:],np.dot(myJmn[lmin:,lmin:],ClSim[lmin:]))

    # collect partials for subtraction later
    #for myEll in range(lmin,partialMax+1):
    #  sEnsemblePartial[i,myEll] = np.dot(ClSim[myEll],
    #    np.dot(myJmn[myEll,myEll],ClSim[myEll]))
    for myLmin in range(lmin,partialMax+1):
      sEnsemblePartial[i,myLmin] = np.dot(ClSim[myLmin:],np.dot(myJmn[myLmin:,myLmin:],ClSim[myLmin:]))


  # plot results
  print 'plotting S_{1/2} distributions... '
  myBins = np.logspace(2,7,100)
  plt.axvline(x=6763,color='b',linewidth=3,label='SMICA inpainted')
  plt.axvline(x=2145,color='g',linewidth=3,label='SMICA masked')
  #plt.hist(sEnsembleFull,bins=myBins,histtype='step',label='full sky')
  #plt.hist(sEnsembleCut, bins=myBins,histtype='step',label='cut sky')

  #partial = sEnsembleFull
  #print partial.shape
  #print partial
  for myEll in range(lmin,partialMax+1):
    #partial -= sEnsemblePartial[:,myEll] 
    #print partial.shape
    #print partial
    plt.hist(sEnsemblePartial[:,myEll],bins=myBins,histtype='step',label='l_min = '+str(myEll))

  plt.gca().set_xscale("log")
  plt.legend()
  plt.xlabel('S_{1/2} (microK^4)')
  plt.ylabel('Counts')
  plt.title('S_{1/2} of '+str(nSims)+' simulated CMBs, l_min = '+str(lmin))
  plt.show()

if __name__=='__main__':
  test()


