#! /usr/bin/env python
"""
Name:
  
Purpose:

Uses:
  healpy
Inputs:

Outputs:

Modification History:
  Written by Z Knight, 2016.09.07
  
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time # for measuring duration
from scipy.interpolate import interp1d

import get_crosspower as gcp # for loadCls
from numpy.polynomial.legendre import legval # for C_l -> C(theta) conversion
from numpy.polynomial.legendre import legfit # for C(theta) -> C_l conversion
#from ispice import ispice
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

def test(nSims=100,lmax=100,lmin=2,useCLASS=1,useLensing=1):
  """
    Purpose:
      function for testing 
    Inputs:

  """
  
  print 'step 3: profit'

if __name__=='__main__':
  test()


