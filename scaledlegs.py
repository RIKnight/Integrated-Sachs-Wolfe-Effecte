#! /usr/bin/env python
"""
Name:
  scaledlegs 
Purpose:
  show legendre polynomials * (2l+1) together
Uses:
  
Inputs:

Outputs:

Modification History:
  Written by Z Knight, 2016.09.07
  
"""

import numpy as np
import matplotlib.pyplot as plt
#import healpy as hp
#import time # for measuring duration
#import get_crosspower as gcp
from numpy.polynomial.legendre import legval # for C_l -> C(theta) conversion
from numpy.polynomial.legendre import legfit # for C(theta) -> C_l conversion
#from ispice import ispice
#from legprodint import getJmn
#from scipy.interpolate import interp1d


def legplot(nterms=6):
  """
  Purpose:
    calculate (2l+1)/4pi * Pl and plot
  Procedure:
    
  Inputs:
    nterms: the number of Pl to plot, starting from 0
  Returns:
    theta values and scaled legendre polynomials
  """
  coefs = np.zeros([nterms,nterms])
  for coef in range(nterms):
    coefs[coef,coef] = ((2*coef)+1)/(4*np.pi)
  npoints = 1801
  thetavals = np.linspace(0,180,npoints)
  xvals = np.cos(thetavals*np.pi/180.)
  print 'x vals: ',xvals
  Pl = np.empty([nterms,npoints])
  for ell in range(nterms):
    Pl[ell] = legval(xvals,coefs[ell])
    #print 'l = ',ell,', scaled Pl = ',Pl[ell]
    plt.plot(thetavals,Pl[ell],label='l = '+str(ell))
  plt.xlabel('angle (degrees)')
  plt.ylabel('P_l * (2*l+1)/4pi')
  plt.title('scaled legendre polynomials')
  #plt.legend()
  plt.show()
  
  return thetavals,Pl



################################################################################
# testing code

def test(nterms=6):
  """
    Purpose:
      function for testing legplot
    Inputs:

  """
  thetavals,Pl = legplot(nterms=nterms)

  print 'step 3: profit'

if __name__=='__main__':
  test()


