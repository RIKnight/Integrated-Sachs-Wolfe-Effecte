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
  Fixed serious error in nSimsBig initialization; 
    added saveText to test function; ZK, 2016.11.25
  Upgraded to new version of op2.optSx that handles m*n nested loops
    and sublists; ZK, 2016.12.07
  
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
# plotting

def makePlotPX(saveFile="optSxResult.npy"):
  """
    Name:
      makePlotPX
    Purpose:
      plotting results of x vs p
    Inputs:
      saveFile: name of a numpy file containing 3*(nSims+1) element array
        3 rows: PvalMinima,XvalMinima,SxEnsembleMin
        fisrt (0th) column: SMICA values
        the other columns: nSims
        Default: optSxResult.npy
    Returns:
      nothing, but makes a plot
  """

  # load results
  myPX = np.load(saveFile)
  PvalMinima = myPX[0]
  XvalMinima = myPX[1]
  SxEnsembleMin = myPX[2]
  nSims = myPX.shape[1]  # actually nSims+1 since SMICA is in 0 position

  # plot
  plt.plot(XvalMinima[0],PvalMinima[0]*100,marker='o',color='r')
  for nResult in range(1,nSims):
    plt.plot(XvalMinima[nResult],PvalMinima[nResult]*100,marker='^',color='b')
  plt.xlim((-.2,.5))
  plt.xlabel('x')
  plt.ylabel('p-value (%)')
  plt.show()


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
# convert text file to numpy format for plotting
def txt2npy(loadFile='optSxJKtext.txt',saveFile='optSxResult.npy'):
  """
  Purpose:
    convert saved text file to numpy format for plotting with op2.makePlots,etc.
  Inputs:
    loadFile: (.txt)
    saveFile: (.npy)
  Outputs:
    saves file saveFile
  """
  PvalMinima,XvalMinima,SxEnsembleMin = np.loadtxt(loadFile,unpack=True)
  np.save(saveFile,np.vstack((PvalMinima,XvalMinima,SxEnsembleMin)))


################################################################################
# testing code

def test(nEnsembles=2,nPerEnsemble=100,loadFile='SofXEnsemble_noFilt_100000.npy',
         saveFile='JackKnife10000.npy',doPlot=False,saveText=True):
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
        Default: JackKnife10000.npy
      doPlot: set to True to do plots for each subensemble
      saveText: set to True to save (append) jackknife results to text file
        filename: optSxJKtext.txt
        Default: True
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
  saveFile2  = 'optSxResult.npy'
  textFile   = 'optSxJKtext.txt'
  
  # peel off x values for new optSx version
  SxValsArray = SxValsArrayBig[1:]

  # loop over subsamples
  doTime = True # to time the optimization run and print output
  startTimeOuter=time.time()
  for nEnsemble in range(nEnsembles):
    # sample for testing; 0 for x vals, 1 for SMICA
    #jackKnifeIndices = np.random.randint(2,nSimsBig+2,size=nPerEnsemble)
    #SxValsArray = np.vstack((sSMICA,SxValsArrayBig[jackKnifeIndices]))
    #print SxValsArray.shape
    #return

    # create random indices but keep SMICA at start
    jackKnifeIndices = np.hstack((np.array([0],dtype=np.uint64),
                          np.random.randint(1,nSimsBig+1,size=nPerEnsemble,dtype=np.uint64)))
    print jackKnifeIndices
    print SxValsArray.shape

    startTime = time.time()
    op2.optSx(xVals,nXvals,SxValsArray,nSimsBig+1,xStart,xEnd,nSearch,PvalMinima,XvalMinima,
              mySxSubset=jackKnifeIndices)
    timeInterval = time.time()-startTime
    if doTime: print 'time elapsed: ',int(timeInterval/60.),' minutes'


    SxEnsembleMin = np.empty(nSims)
    for nSim in range(nSims):
      # need to interpolate since optSx uses interpolation
      SofX = interp1d(xVals,SxValsArray[nSim])
      SxEnsembleMin[nSim] = SofX(XvalMinima[nSim])

    # add result[0] into pxs_results array
    pxs_results[nEnsemble] = np.array((PvalMinima[0],XvalMinima[0],SxEnsembleMin[0]))

    # save S_x, P(x), x results
    np.save(saveFile2,np.vstack((PvalMinima,XvalMinima,SxEnsembleMin)))
    if saveText:
      f_handle=file(textFile,'a')
      np.savetxt(f_handle,np.array([pxs_results[nEnsemble]]))
      f_handle.close()

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


