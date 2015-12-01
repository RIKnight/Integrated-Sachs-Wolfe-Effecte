#! /usr/bin/env python
"""
  NAME:
    make_overmass.py

  PURPOSE:
    This program creates sets of ISW profiles following the formulation of
      Papai and Szapudi, 2010. (PS)

  USES:
    ISWprofile.py             : code for creating overmass profiles
    test_matterpower_2015.dat : contains matter power spectrum

  MODIFICATION HISTORY:
    Written by Z Knight, 2015.08.28
    Added plotting by listdir; ZK, 2015.09.01
    Expanded rmax to 600 due to corrected overmass calc; ZK, 2015.10.12

"""

import ISWprofile as ISW
import numpy as np
import time #for measuring duration of processes
import matplotlib.pyplot as plt
from os import listdir

def plotAll(directory):
  """
    plot all M(r) files in a given directory
    directory is the directory to plot; ends in /
  """
  files = listdir(directory)
  files = [file for file in files if 'overmass' in file]
  for filename in files:
    rcent,Mcent = np.loadtxt(directory+filename,unpack=True)
    plt.plot(rcent,Mcent)
    plt.xlabel('r [Mpc/h]')
    plt.ylabel('overmass M(r)')
    plt.title('file: '+filename)
    plt.show()

# maximum radius to calculate overmass
rmax = 600 #Mpc/h
npoints = 200

doPSGrange = False#True
# create a range of M(R) plots for a range of R values, all with delta_in(R) = +1
# this section is for recreating results in PSG10
if doPSGrange:
  dataDir = '/shared/Data/PSG/'
  Rvalues = np.linspace(10,160,16) # copies PSG values
  basename = 'overmass_R'
  startTime = time.time()
  for R in Rvalues:
    filename = basename+"%03d"%R+'.txt'
    print 'starting R = ',R 
    ISW.save_overmass(dataDir+filename,R=R,rmax=rmax,delta_in=1.0,npoints=npoints)
    print 'saved file ',dataDir+filename
    print 'time elapsed: ',int((time.time()-startTime)/60),' minutes'

doPlot = True#False
# plot the previously calculated overmass profiles
if doPlot:
  plotAll('/shared/Data/PSG/')


doNaive = False#True
doScaled = False#True

if doNaive or doScaled:
  #open data files
  clusterFile = '/shared/Data/Gr08_clustercat.txt'
  voidFile    = '/shared/Data/Gr08_voidcat.txt'
  cz,cra,cdec,cmean_r_sky,cmax_r_sky,cvol,cd_all,cd_pos,cd_max,cdenscontrast,cprob = np.loadtxt(clusterFile,skiprows=1,unpack=True)
  vz,vra,vdec,vmean_r_sky,vmax_r_sky,vvol,vd_all,vd_neg,vd_min,vdenscontrast,vprob = np.loadtxt(voidFile,skiprows=1,unpack=True)
  #print 'cz:',cz
  #print 'vz:',vz
  # use mean_r_sky for R; cd_pos, vd_neg for delta_in(R)

# create profiles using catalog components
# this section applies catalog parameters blindly to create profiles
if doNaive:
  dataDir = '/shared/Data/GNS01/'
  startTime = time.time()
  for clusterVoid in range(25):
    namemid = "%02d"%(clusterVoid+1)+'_meanRsky_'
    cfilename = 'GNS01overmass_cluster'+namemid+'Dpos.txt'
    vfilename = 'GNS01overmass_void'+namemid+'Dneg.txt'
    #print cfilename,vfilename
    print 'starting cluster ',clusterVoid+1
    ISW.save_overmass(dataDir+cfilename,R=cmean_r_sky[clusterVoid],rmax=rmax,delta_in=cd_pos[clusterVoid],npoints=npoints)
    print 'saved file ',dataDir+cfilename
    print 'time elapsed: ',int((time.time()-startTime)/60),' minutes'
    print 'starting void ',clusterVoid+1
    ISW.save_overmass(dataDir+vfilename,R=vmean_r_sky[clusterVoid],rmax=rmax,delta_in=vd_neg[clusterVoid],npoints=npoints)
    print 'saved file ',dataDir+vfilename
    print 'time elapsed: ',int((time.time()-startTime)/60),' minutes'

doPlot = False
# plot the previously calculated overmass profiles
if doPlot:
  plotAll('/shared/Data/GNS01/')


# crate profiles as above but scale to match average R, delta_in from PSG10
if doScaled:
  RAvgTarget = 55*0.6704 #Mpc; h=H_0/100 from ISWprofile.ClusterVoid.H_0 / 100
  dinAvgTarget = 2.0 #read from plot in PSG figure 6
  RAvg = (cmean_r_sky.mean()+vmean_r_sky.mean())/2.0
  dinAvg = (cd_pos.mean()-vd_neg.mean())/2.0
  cRScaled = cmean_r_sky  *RAvgTarget/RAvg
  vRScaled = vmean_r_sky  *RAvgTarget/RAvg
  cDScaled = cd_pos       *dinAvgTarget/dinAvg
  vDScaled = vd_neg       *dinAvgTarget/dinAvg

  dataDir = '/shared/Data/GNS02/'
  if not doNaive:
    startTime = time.time()
  for clusterVoid in range(25):
    namemid = "%02d"%(clusterVoid+1)+'_meanRskyScaled55_'
    cfilename = 'GNS02overmass_cluster'+namemid+'DposScaled2.0.txt'
    vfilename = 'GNS02overmass_void'+namemid+'DnegScaled2.0.txt'
    #print cfilename,vfilename
    print 'starting cluster ',clusterVoid+1
    ISW.save_overmass(dataDir+cfilename,R=cRScaled[clusterVoid],rmax=rmax,delta_in=cDScaled[clusterVoid],npoints=npoints)
    print 'saved file ',dataDir+cfilename
    print 'time elapsed: ',int((time.time()-startTime)/60),' minutes'
    print 'starting void ',clusterVoid+1
    ISW.save_overmass(dataDir+vfilename,R=vRScaled[clusterVoid],rmax=rmax,delta_in=vDScaled[clusterVoid],npoints=npoints)
    print 'saved file ',dataDir+vfilename
    print 'time elapsed: ',int((time.time()-startTime)/60),' minutes'

doPlot = False
# plot the previously calculated overmass profiles
if doPlot:
  plotAll('/shared/Data/GNS02/')


