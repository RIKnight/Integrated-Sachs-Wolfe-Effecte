#! /usr/bin/env python
"""
Name:
  get_crosspower
Purpose:
  load total, primordial, and late Cl power spectra and create cross power:
    C_l^tot = C_l^(prim,prim) + 2*C_l^(prim,late) + C_l^(late,late)
NOTE:
  the getAlms function does not appear to work correctly.  
    Use healpy.synalm instead.
Uses:
  healpy
Inputs:
  Data files as specified in loadCls function
Outputs:

Modification History:
  Written by Z Knight, 2016.06.08
  Added single spectrum testing; switched to synalm; ZK, 2016.06.27
  Added support for multiple Cl in showCl; ZK, 2016.07.01
  Switched default in loadCls to useLensing=1; ZK, 2016.10.07

"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
#import make_Cmatrix as mcm # for getCl
from numpy.linalg import cholesky
from numpy.random import randn



def showCl(ell,tempsSet,title='power spectrum'):
  """
    create a plot of power spectrum
    ell: the multipole number
    tempsSet: the temperature power in Kelvin**2, or an array of power spectra
    title : the title for the plot
    uses ell*(ell+1)/2pi scaling on vertical axis

    Modification History:
      copied from make_Cmatrix.showCl; 2016.06.08
      added support for multiple Cl in tempsSet; ZK, 2016.07.01
  """
  # check for multiple spectra
  if tempsSet.shape.__len__() == 1:
    tempsSet = np.array([tempsSet])
  for temps in tempsSet:
    plt.plot(ell,temps*ell*(ell+1)/(2*np.pi) *1e12) #1e12 to convert to microK**2
  plt.xlabel('multipole moment l')
  plt.ylabel('l(l+1)C_l/(2pi) [microK**2]')
  plt.title(title)
  plt.show()


def getAlms(A_lij,lmax=100):
  """
  Function to calculate a_lm^primary and a_lm^late from cholesky dec. of cov. mat.
  Follows the a_lm construction from AKW 15

  Inputs:
    A_lij: the Cholesky decomposition of the covariance matrix
    lmax: set this to the lmax of the realization
      default: 100
  Returns:
    alm_prim, alm_late: the primary and late alm coefficients
  NOTE:
    This method appears to produce alm.s which combine to have lower power than
      the power spectra that created A_lij.  Reason unknown; do not use this.

  """
  # use standard healpy indexing 
  lindex,mindex=hp.Alm.getlm(lmax=lmax)
  indmax = lindex.size #5151 with lmax=100

  alm_prim = np.zeros(indmax,dtype=complex)
  alm_late = np.zeros(indmax,dtype=complex)
  for index in range(indmax):
    zeta = np.array([np.complex(randn(1),randn(1)),np.complex(randn(1),randn(1))])/np.sqrt(2.)
    ell = lindex[index]
    aDotZeta = np.dot(A_lij[ell],zeta)
    alm_prim[index] = aDotZeta[0]
    alm_late[index] = aDotZeta[1]
  return alm_prim,alm_late

def loadCls(useCLASS=1,useLensing=1,classCamb=1,doPlot=False):
  """
  Purpose:
    for loading power spectra from CLASS output files
  
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
        Also: the CLASS format option is not yet fully implemented
        default: 1
    doPlot: set to 1 to make plots
        default: False

  Ouptuts:
    returns ell and power spectra

  """

  dataDir = '/Data/sims/'
  if useCLASS:
    if useLensing:
      if classCamb:
        fullCl_file = 'CAMB_LCDM_full_Cl_lensed.dat'
        primCl_file = 'CAMB_LCDM_prim_Cl_lensed.dat'
        lateCl_file = 'CAMB_LCDM_late_Cl_lensed.dat'
      else: #these need to be replaced; was typo in n_s
        fullCl_file = 'LCDM_full_Cl_lensed.dat'
        primCl_file = 'LCDM_prim_Cl_lensed.dat'
        lateCl_file = 'LCDM_late_Cl_lensed.dat'
    else:
      if classCamb:
        fullCl_file = 'CAMB_LCDM_full_Cl.dat'
        primCl_file = 'CAMB_LCDM_prim_Cl.dat'
        lateCl_file = 'CAMB_LCDM_late_Cl.dat'
      else: #these need to be replaced; was typo in n_s
        fullCl_file = 'LCDM_full_Cl.dat'
        primCl_file = 'LCDM_prim_Cl.dat'
        lateCl_file = 'LCDM_late_Cl.dat'
  else: # use CAMB
    if useLensing:
      fullCl_file = 'CAMB_full_lensedCls.dat'
      primCl_file = 'CAMB_ISWout_lensedCls.dat' # actually prim + ISWout
      lateCl_file = 'CAMB_ISWin_lensedCls.dat'  # actually prim + ISWin
    else: 
      fullCl_file = 'CAMB_full_scalCls.dat'
      primCl_file = 'CAMB_ISWout_scalCls.dat'   # actually prim + ISWout
      lateCl_file = 'CAMB_ISWin_scalCls.dat'    # actually prim + ISWin

  fullClArray = np.loadtxt(dataDir+fullCl_file) #shape: (2999,8)
  primClArray = np.loadtxt(dataDir+primCl_file)
  lateClArray = np.loadtxt(dataDir+lateCl_file)
  ell = fullClArray[:,0] #starts with ell[0]=2
  # call them Dl since they have Dl scaling
  fullDl = fullClArray[:,1] #TT
  primDl = primClArray[:,1] #TT
  lateDl = lateClArray[:,1] #TT

  #doPlot = False
  # Is this C_l or D_l?  Check for familiar plot
  #showCl(ell,fullDl)
  #plt.plot(ell,fullDl)
  #plt.title('no l scaling')
  #plt.show()

  # try a plot
  if doPlot:
    doLog = False
    if doLog:
      plt.loglog(ell,fullDl)
      plt.loglog(ell,primDl)
      plt.loglog(ell,lateDl)
      #plt.xlim([1,100])
    else:
      plt.plot(ell,fullDl)
      plt.plot(ell,primDl)
      plt.plot(ell,lateDl)
      plt.xlim([2,20])
      plt.ylim([0,1200])
    plt.title('full, primary, late Dl')
    plt.xlabel('l')
    plt.ylabel('D_l')
    plt.show()

  # find the difference
  crossDl = 0.5*(fullDl-primDl-lateDl)

  # convert to Cl
  conv = ell*(ell+1)/(2*np.pi)
  fullCl  = fullDl/conv
  primCl  = primDl/conv
  lateCl  = lateDl/conv
  crossCl = crossDl/conv
  #print ell, conv #ell[0] = 2.0

  #doPlot = False#True
  if doPlot:
    plt.loglog(ell,np.abs(crossCl)) #what does negative cross power mean?
    plt.xlim([1,100])
    plt.title('abs(cross power)') 
    plt.xlabel('l')
    plt.ylabel('C_l')
    plt.show()

  #print crossCl[:50]

  # AKW specify C_l^ii * C_l^jj >= (C_l^ij)^2.  Check this:
  if doPlot:
    plt.semilogx(ell,crossCl/np.sqrt(primCl*lateCl))
    plt.title('crossCl/sqrt(primCl*lateCl)')
    plt.xlabel('l')
    plt.ylabel('r_l')
    plt.show()

  return ell,fullCl,primCl,lateCl,crossCl



################################################################################
# testing code

def test(useCLASS=1,useLensing=0,classCamb=1):
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
  """

  # load data
  ell,fullCl,primCl,lateCl,crossCl = loadCls(useCLASS=useCLASS,useLensing=useLensing,classCamb=classCamb)


  """ # this section from AKW; didn't work: don't use it
  # form matrices
  C_lij = np.transpose(np.array([[primCl,crossCl],[crossCl,lateCl]]) )
  #print C_lij.shape #(2,2,3999) transposes to (3999,2,2)
  num_l = C_lij.shape[0]
  #print num_l #3999

  #Cholesky decomposition
  A_lij = cholesky(C_lij)
  #print A_lij.shape #(2,2,3999) transposes to (3999,2,2)

  # get a_lms
  lmax = 100
  alm_prim,alm_late = getAlms(A_lij,lmax=lmax)


  # make maps
  if doPlot:
    nside = 128
    myMap_prim = hp.alm2map(alm_prim,nside)
    myMap_late = hp.alm2map(alm_late,nside)
    hp.mollview(myMap_prim,title='primary CMB realization')
    plt.show()
    hp.mollview(myMap_late,title='ISW realization')
    plt.show()
  """

  
  """
  # try a simple case
  nSims = 1000
  lmax = 100
  Clfull_sum = np.zeros(lmax+1)
  for nSim in range(nSims):
    print 'starting sim ',nSim+1, ' of ',nSims
    alm_sim = hp.synalm(fullCl,lmax=lmax)
    Clfull_sum = Clfull_sum + hp.alm2cl(alm_sim)
  Cl_full_avg = Clfull_sum/nSims

  doPlot = True
  if doPlot:
    plt.plot(ell[:lmax+1],Cl_full_avg*conv[:lmax+1])
    plt.plot(ell[:lmax+1],fullCl[:lmax+1]*conv[:lmax+1])
    plt.title('primary')
    plt.ylabel('D_l')
    plt.show()

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
  



if __name__=='__main__':
  test()


