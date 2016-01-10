#! /usr/bin/env python
"""
  NAME:
    make_sims.py
  PURPOSE:
    to load simulated ISW maps and simulated CMB maps and add them
  USES:
    make_Cmatrix.py
  MODIFICATION HISTORY:
    Written by Z Knight, 2015.10.02
    Removed saving of 100 huge FITS files; Added map smoothing; ZK 2015.10.19
    Added support for nested ordering; ZK, 2016.01.09
"""

import numpy as np
#import matplotlib.pyplot as plt
import healpy as hp
#import time # for measuring duration
#import make_Cmatrix as mcm

mapDirectory = '/shared/Data/sims/'
cmbFiles=['simCMB01.fits',
          'simCMB02.fits',
          'simCMB03.fits',
          'simCMB04.fits',
          'simCMB05.fits',
          'simCMB06.fits',
          'simCMB07.fits',
          'simCMB08.fits',
          'simCMB09.fits',
          'simCMB10.fits'] #all NSIDE=1024
#iswFile='/shared/Data/PSG/hundred_point/ISWmap1024_RING_din1_R010.fits' #NSIDE=1024, DeltaT/T
iswFile='/shared/Data/PSG/hundred_point/ISWmap_RING_R010.fits' #NSIDE=64, DeltaT

nAmps = 10
amplitudes = np.logspace(-1,2,nAmps)
ampTags = ['a','b','c','d','e','f','g','h','i','j']
nested = False

print 'reading map ',iswFile
isw = hp.read_map(iswFile,nest=nest)
for cmbFile in cmbFiles:
  print 'reading map ',cmbFile
  cmb = hp.read_map(mapDirectory+cmbFile,nest=nested)
  # rebin to NSIDE=64
  if nested:
    cmb = hp.ud_grade(cmb,64,order_in='NESTED',order_out='NESTED')
  else:
    cmb = hp.ud_grade(cmb,64,order_in='RING',order_out='RING')
  for ampNum in range(nAmps):
    myCMB = cmb+isw*amplitudes[ampNum]
    hp.write_map(mapDirectory+cmbFile[:-5]+ampTags[ampNum]+'.fits', myCMB, nest=nested, coord='G')

print 'done'


