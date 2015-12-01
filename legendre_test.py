#! /usr/bin/env python
"""
  NAME:
    legendre_test.py
  PURPOSE:
    to test accuracy of legendre transforms
  USES:
    healpy
    make_Cmatrix.py    
    gacf.py
  MODIFICATION HISTORY:
    Written by Z Knight, 2015.11.05
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
#import time # for measuring duration
import make_Cmatrix as mcm
import gacf

from numpy.polynomial.legendre import legval
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import legendre


def invLegval(xArray,covar,nLpoints):
  """ 
    inverse legendre transformation 
    xArray: an array of x values, should span -1 to 1
    covar: the covariance(x) to be transformed to C_l
      should have same length as xArray
    nLpoints: the nubmer of l values to produce

    returns an array of C_l values the same length as xArray
      corresponding to ell = 0,1, ..., xArray.size-1
  """
  #nPoints = xArray.size

  # interpolation function for transformed power spectrum
  Cinterp = interp1d(xArray,covar)
  #plt.plot(xArray,Cinterp(xArray))
  #plt.title('interpolation function for transform of C_l')
  #plt.show()

  C_ell = np.zeros(nLpoints)
  for n in range(nLpoints):
    print 'Starting P_n with n = ',n
    Pn = legendre(n)
    result = quad(lambda x: Pn.__call__(x)*Cinterp(x), -1,1) #, limit=500)
    C_ell[n] = result[0]*2*np.pi

  return C_ell


def test():
  """ function for doing the testing """

  """
  # test using gacf: gaussian autocorrelation function
  print 'testing with gaussian autocorrelation function'
  C_0 = 100
  theta_C = .25 #radians; rather arbitrary
  nLpoints = 100
  nXpoints = 10*nLpoints
  thetaDomain = np.linspace(0,np.pi,nXpoints) #radians
  xDomain = np.cos(thetaDomain)
  ellDomain = np.arange(nLpoints)
  #print xDomain

  myC_GACF = gacf.C_GACF(C_0,thetaDomain,theta_C)
  myC_ell = gacf.C_ell(C_0,ellDomain,theta_C)

  # find trans of myCGACF and inv.trans of myCell
  lFac = (2*ellDomain+1)/(4*np.pi)
  nuC_ell  = invLegval(xDomain,myC_GACF,nLpoints)
  nuC_GACF = legval(xDomain,lFac*myC_ell) # ell values implicit

  # inverse transforms
  nnuC_GACF = legval(xDomain,lFac*nuC_ell)
  nnuC_ell  = invLegval(xDomain,nuC_GACF,nLpoints)

  # plot comparisons
  plt.plot(ellDomain,myC_ell)
  plt.plot(ellDomain,nuC_ell)
  plt.plot(ellDomain,nnuC_ell)
  #plt.plot(ellDomain,myC_ell/nuC_ell)
  plt.title('C_ell actual, invLegval(C_GACF), and invLegval(legval(actual))')
  plt.xlabel('ell')
  plt.show()

  plt.plot(xDomain,myC_GACF)
  plt.plot(xDomain,nuC_GACF)
  plt.plot(xDomain,nnuC_GACF)
  #plt.plot(xDomain,myC_GACF/nuC_GACF)
  plt.title('C_GACF actual, legval(C_ell), and legval(invLegval(actual))')
  plt.ylim([-1*C_0,C_0])
  plt.xlabel('x = cos(theta)')
  plt.show()

  # plot ratios
  plt.plot(ellDomain,myC_ell/nuC_ell)
  plt.title('myC_ell / nuC_ell')
  plt.show()
  plt.plot(xDomain,myC_GACF/nuC_GACF)
  plt.title('myC_GACF / nuC_GACF')
  plt.show()



  """


  # test using the actual power spectrum I'll be using
  print 'testing with CAMB power spectrum'
  ISWoutFile = 'ISWout_scalCls.fits'
  ell,temps = mcm.getCl(ISWoutFile)
  #mcm.showCl(ell,temps)

  lmax  = 250
  pbeam = hp.gauss_beam(5./60*np.pi/180,lmax=lmax)
  mbeam = hp.gauss_beam(120./60*np.pi/180,lmax=lmax)
  #mbeam = hp.gauss_beam(90./60*np.pi/180,lmax=lmax)
  Wpix  = hp.pixwin(64)
  B_l   = mbeam/pbeam
  W_l   = Wpix[:lmax+1]
  Cell2 = temps[:lmax+1]*B_l**2*W_l**2 #windowed and beamsmoothed C_l
  #Cell2 = temps[:lmax+1]  # no smoothing for testing
  
  
  # plot beam and window and squares, etc
  plt.plot(B_l)
  plt.plot(W_l)
  plt.plot(B_l**2*W_l**2)
  plt.title('beam, window, beam^2*window^2')
  plt.show()

  mcm.showCl(ell,temps)
  mcm.showCl(ell[:lmax+1],Cell2,title='CAMB C_l * B_l**2 * W_l**2')
  

  # create array to contain legendre transform
  nlpoints = lmax+1
  nxpoints = 10*nlpoints
  thetaDomain = np.linspace(0,np.pi,nxpoints) #radians
  xDomain = np.cos(thetaDomain)
  ellDomain = np.arange(nlpoints)
  #xArray = np.linspace(-1,1,lmax+1)

  # create prefactor for leg.trans. of power spectrum
  lFac = (2*ellDomain+1)/(4*np.pi)

  # make high pass filter
  #highpass = 12
  #Cell2 = np.concatenate((np.zeros(highpass),Cell2[highpass:]))

  # modify Cl
  #Cell2 = Cell2*ellDomain*(ellDomain-1)
  Cell2 = Cell2*1e12

  # find legendre expansion
  covar = legval(xDomain,lFac*Cell2) # ell values implicit
  plt.plot(xDomain,covar)
  plt.title('legendre transform of modified power spectrum')
  plt.xlabel('x = cos(theta)')
  plt.show()

  """
  # check interpolation
  Cinterp = interp1d(xDomain,covar)
  xDomain2 = np.linspace(-1,1,nlpoints*100)
  plt.plot(xDomain,covar)
  plt.plot(xDomain2,Cinterp(xDomain2))
  plt.title('comparison of expansion and its interpolation')
  plt.xlabel('x = cos(theta)')
  plt.show()
  """

  # find inverse legendre
  CellInv = invLegval(xDomain,covar,nlpoints)
  plt.plot(ellDomain,Cell2)
  plt.plot(ellDomain,CellInv)
  plt.xlabel('ell')
  plt.ylabel('C_l')
  plt.title('comparing modified C_l and L(Linv(C_l))')
  plt.show()
  
  #mcm.showCl(ellDomain,temps[:lmax+1])
  #mcm.showCl(ellDomain,W_l**2*B_l[:lmax+1]**2,title='W_l**2 * B_l**2')
  #mcm.showCl(ellDomain,Cell2,title='modified C_l')
  #mcm.showCl(ellDomain,CellInv,title='inverse transform of transform of modified C_l')
  
  """
  plotFac = ellDomain*(ellDomain-1)/(2*np.pi)
  plt.plot(ellDomain,plotFac*Cell2)
  plt.plot(ellDomain,plotFac*CellInv)
  plt.xlabel('ell')
  plt.ylabel('C_l * ell(ell-1)/(2pi)')
  plt.title('comparing modified C_l and L(Linv(C_l))')
  plt.show()
  """
  
  # retransform difference
  cDiff = CellInv-Cell2
  covarDiff = legval(xDomain,lFac*cDiff)
  plt.plot(xDomain,covarDiff)
  plt.title('Linv( L(Linv(Cl))-Cl )')
  plt.xlabel('x = cos(theta)')
  plt.show()


  #import readline
  #readline.write_history_file('my_history.txt')


if __name__=='__main__':
  test()


