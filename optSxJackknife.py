#! /usr/bin/env python
"""
Name:
  optSxJackknife 
Purpose:
  do Jackknife testing to check for stability of optimized x, S_x, P(x)
Uses:
  healpy
  legprodint    (legendre product integral)
  ramdisk.sh    (creates and deletes RAM disks)
  optimizeSx2   (wrapper for optimizeSx.so and plotting)
Inputs:
  Needs file SofXEnsemble_noFilt_100000.npy, created by optimizeSx2.py
Outputs:

Modification History:
  Branched from optimizeSx2.py; ZK, 2016.11.20
  Made random number generator test; ZK, 2016.11.24
  Fixed serious error in nSimsBig initialization; ZK, 2016.11.25
  
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time               # for measuring duration
from scipy.interpolate import interp1d
import subprocess         # for calling RAM Disk scripts
import ctypes                         # for calling c .so file
from numpy.ctypeslib import ndpointer # for calling c .so file
from matplotlib import cm # color maps for 2d histograms
import corner             # for corner plot
from pyfiglet import Figlet           # for big text

import get_crosspower as gcp # for loadCls
from numpy.polynomial.legendre import legval # for C_l -> C(theta) conversion
from numpy.polynomial.legendre import legfit # for C(theta) -> C_l conversion
from ispice import ispice # for calculating C_l from map
import legprodint         # for integral of product of legendre polynomials
import sim_stats as sims  # for getSMICA and getCovar

import optimizeSx2 as op2 # for optSx, PvalPval, makeC2Plot, makeCornerPlotSmall



################################################################################
# test the random number generator

def randintTest(nSimsBig=100000,nPerEnsemble=10000,nEnsembles=1):
  """
    Purpose:
      the jackknife code is behaving as if random numbers aren't so here I am testing
        the randint function to see whether or not it is responsible for this
    Inputs:
      nSimsBig: the number of sims in the larger ensemble, from which the 
        jackknife sampes will be drawn
        Default: 10^5 (to match the case below)
      nPerEnsemble: the number of draws from the larger ensemble per test
        Default: 10^4 (to match the case below)
      nEnsembles: the number of ensembles to do
        note: not used; default=1
    Outputs:


  """
  # create container for random numbers
  myRandoms = np.zeros(nSimsBig)

  # use the same command as in the code below
  #jackKnifeIndices = np.random.randint(2,nSimsBig+2,size=nPerEnsemble)
  # use slightly different command
  jackKnifeIndices = np.random.randint(nSimsBig,size=nPerEnsemble)

  myRandoms[jackKnifeIndices] += 1
  plt.plot(myRandoms)
  plt.show()


################################################################################
# testing code

def test(nEnsembles=2,nPerEnsemble=100,loadFile='SofXEnsemble_noFilt_100000.npy',
         saveFile='JackKnife5000.npy',doPlot=False):
  """
    Purpose:
      program to create jackknife subensembles and optimize x, P(x)
    Uses:
      input file SofXEnsemble_noFilt_100000.npy, created by optimizeSx2.py
    Inputs:
      nEnsembles: Number of ensembles to pull from sample
        Default: 2
      nPerEnsemble: number of sims per ensemble
        Default: 100
      loadFile: numpy file name to load 2d S_x data from
        First row of array contains x values, the rest has S_x
        Default: SofXEnsemble_noFilt_100000.npy
      saveFile: numpy file name to save P(x), S_x, x jackknifed array
        shape will be (nEnsembles+1,3); +1 for comparison point, 3 for p,x,s
      np.save(saveFile,pxs_results[nEnsemble])
        Default: JackKnife5000.npy
      doPlot: set to True to do plots for each subensemble
  """

  # prep for Figlet printing
  f=Figlet(font='slant')

  # load data
  SxValsArrayBig = np.load(loadFile)
  xVals    = SxValsArrayBig[0]
  sSMICA   = SxValsArrayBig[1]
  nXvals   = xVals.size
  nSimsBig = np.shape(SxValsArrayBig)[0]-2
  nSims    = nPerEnsemble+1 #+1 for SMICA in position 0


  #iStart=455
  #SxValsArray = np.vstack((sSMICA,SxValsArrayBig[iStart:iStart+nSims-1]))
  #print SxValsArray.shape

  xStart = -1.0
  xEnd = 1.0
  nSearch = nXvals # same as nXvals for now, but spaced equally in x, not theta
  PvalMinima = np.empty(nSims) # for return values
  XvalMinima = np.empty(nSims) # for return values
  pxs_results = np.empty((nEnsembles,3)) #3 for p,x,s
  saveFile2  = "optSxResult.npy"
  
  # loop over subsamples
  doTime = True # to time the optimization run and print output
  startTimeOuter=time.time()
  for nEnsemble in range(nEnsembles):
    # sample for testing; 0 for x vals, 1 for SMICA
    jackKnifeIndices = np.random.randint(2,nSimsBig+2,size=nPerEnsemble)
    SxValsArray = np.vstack((sSMICA,SxValsArrayBig[jackKnifeIndices]))
    #print SxValsArray.shape
    #return

    startTime = time.time()
    op2.optSx(xVals,nXvals,SxValsArray,nSims,xStart,xEnd,nSearch,PvalMinima,XvalMinima)
    timeInterval = time.time()-startTime
    if doTime: print 'time elapsed: ',int(timeInterval/60.),' minutes'


    SxEnsembleMin = np.empty(nSims)
    for nSim in range(nSims):
      # need to interpolate since optSx uses interpolation
      SofX = interp1d(xVals,SxValsArray[nSim])
      SxEnsembleMin[nSim] = SofX(XvalMinima[nSim])

    # save S_x, P(x), x results
    np.save(saveFile2,np.vstack((PvalMinima,XvalMinima,SxEnsembleMin)))

    # add result[0] into pxs_results array
    pxs_results[nEnsemble] = np.array((PvalMinima[0],XvalMinima[0],SxEnsembleMin[0]))

    if doPlot:
      #op2.makePlots(saveFile=saveFile2)
      op2.makeCornerPlotSmall(saveFile=saveFile2)

    # print results
    pv = op2.PvalPval(saveFile=saveFile2)
    #print ' '
    #print 'nSims = ',nSims-1
    #print 'time interval 1: ',timeInterval1,'s, time interval 2: ',timeInterval2,'s'
    #print '  => ',timeInterval1/(nSims-1),' s/sim, ',timeInterval2/(nSims-1),' s/sim'
    
    print f.renderText(str(nEnsemble+1)+' of '+str(nEnsembles))
    print 'Ensemble ',nEnsemble+1,' of ',nEnsembles,': '
    print '  SMICA optimized S_x: S = ',SxEnsembleMin[0],', for x = ',XvalMinima[0], \
          ', with p-value ',PvalMinima[0]
    print '  P-value of P-value for SMICA: ',pv
    print ' '

  timeInterval = time.time()-startTimeOuter
  if doTime: print 'total time elapsed: ',int(timeInterval/60.),' minutes'

  # save Jackknife results
  comparisonPoint = (0.0037,0.34,1276) # from 10000f sims
    #log10(0.0037)=-2.43, log10(1276)=3.105
  np.save(saveFile,np.vstack((comparisonPoint,pxs_results)).T)

  # plot and print results
  op2.makePlots(saveFile=saveFile)
  op2.makeCornerPlotSmall(saveFile=saveFile)
  print 'log_10(0.0037)=-2.34, log_10(1276)=3.105'
  print ''

  print 'step 3: profit'
  print ''

if __name__=='__main__':
  test(nSims=500,suppressC2=False)


