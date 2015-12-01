#! /usr/bin/env python
"""
  NAME:
    make_ISW_map

  PURPOSE:
    create a HEALpix spherical map containing an ISW map from one or more
      superclusters or supervoids
    Pixel values will be DeltaT/T * 2.726K
  
  USES:
    ISWprofile.py
    cosmography.py

  MODIFICATION HISTORY:
    Written by Z Knight, 2015.08.14
    FIxed comoving distance bug; ZK, 2015.08.20
    Switched to nside=64; expanded to 100 clusters+voids; ZK, 2015.09.02
    Modified ISW profile to have only 10 points and log spacing; 
      added listdir for filenames; ZK, 2015.09.04
    Added zero ISW point at 2x max radius; 
      switched from 10 pt logspace to 100 pt linspace; ZK, 2015.09.23
    Added newProfile switch; ZK, 2015.09.29
    Added nested switch; ZK, 2015.10.02
    Added newMap switch; ZK, 2015.10.12
    Added rmax,npoints to makeISWProfile; ZK, 2015.10.16

"""

import numpy as np
import ISWprofile as ISW
import cosmography
import matplotlib.pyplot as plt
import healpy as hp
from scipy.interpolate import interp1d
from os import listdir

def glgb2vec(gl,gb):
  """ 
    convert galactic latitude and longitude to position vector on sphere
    gl: galactic longitude in degrees
    gb: galactic latitude in degrees from equator
    returns normalized cartesian vector on sphere
  """
  return hp.ang2vec((90-gb)*np.pi/180.,gl*np.pi/180.)

def makeISWProfile(MFile,zCent,profileFile,noSave=False,rmax=400,npoints=101):
  """
    use an overmass file to create an ISW profile
    innermost point is at r=0
    parameters:
      MFile: name of file that contains r [Mpc/h], M(r) [(Mpc/h)**3] columns
      zCent: the redshift of the center of the cluster/void
      profileFile: name of file to create with r [Mpc], DeltaT/T columns
      noSave: set this to True to not save a file
        default: False
      rmax: the maximum radius for the profile [Mpc]
        default: 400
      npoints: number of points in profile
        default: 101
    returns:
      impact: an array of impact parameters [Mpc]
      ISW: a corresponding array of DeltaT/T values (no units)
  """
  myCluster = ISW.ClusterVoid(MFile,zCent)
  delta_z = 0.2
  #impactDomain = np.logspace(-1,1,npoints)*rmax/10 # rmax/100 to rmax Mpc
  impactDomain = np.linspace(0,rmax,npoints)
  ISWRange = np.zeros(npoints)
  for pointnum in range(npoints):
    print 'calculating point number ',pointnum+1,' of ',npoints
    ISWRange[pointnum] = myCluster.overtemp(impactDomain[pointnum],delta_z)
  if not noSave:
    np.savetxt(profileFile,np.vstack((impactDomain,ISWRange)).T)
  return impactDomain,ISWRange



doPlot = True
CMBtemp = 2.7260 # +-0.0013 K (WMAP) Fixen, 2009 


# get parameters from ISW.ClusterVoid
H_0     = ISW.ClusterVoid.H_0      #km/sec/Mpc
Omega_M = ISW.ClusterVoid.Omega_M
Omega_L = ISW.ClusterVoid.Omega_L
Omega_k = 0.0 # not included in ClusterVoid
c_light = ISW.ClusterVoid.c_light  #km/sec
DH = c_light/H_0 # hubble distance in Mpc
print 'H_0: ',H_0

# find comoving distance to redshift in Mpc
# currently has omega_k = 0 hardcoded
zMax = 1 # further than all clusters in survey
nSteps = 5000 # for discrete sum appx. to integration
zVals, comDists = cosmography.ComovingDistance(zMax,Omega_M,Omega_L,nSteps,H_0)
comovInterp = interp1d(zVals,comDists)


nside = 64 #1024
nested = False
print 'NSIDE=',nside,' NESTED=',nested
# load HEALpix coordinates file
datadir = '/shared/Data/'
if nside == 64:
  if nested:
    coordfile = 'pixel_coords_map_nested_galactic_res6.fits'
  else: # 'RING'
    coordfile = 'pixel_coords_map_ring_galactic_res6.fits'
if nside == 1024:
  if nested:
    coordfile = 'pixel_coords_map_nested_galactic_res10.fits'
  else: # 'RING'
    coordfile = 'pixel_coords_map_ring_galactic_res10.fits'
# from header: field 1 is longitude, field 2 is latitude; both in degrees
print 'loading coordinates from file ',coordfile
longitudes,latitudes = hp.read_map(datadir+coordfile,(0,1),nest=nested)

# load GNS catalog coordinates
clusterFile = '/shared/Data/Gr08_clustercat.txt'
voidFile    = '/shared/Data/Gr08_voidcat.txt'
#clusterFile = '/shared/Data/Gr08_clcat_trunc1.txt' #truncated catalog for 3 cl, 3 vo close together
#voidFile    = '/shared/Data/Gr08_vocat_trunc1.txt'
#clusterFile = '/shared/Data/Gr08_clcat_trunc2.txt' #truncated catalog for 2 cl, 2 vo far apart
#voidFile    = '/shared/Data/Gr08_vocat_trunc2.txt'
cz,cra,cdec,cmean_r_sky,cmax_r_sky,cvol,cd_all,cd_pos,cd_max,cdenscontrast,cprob = np.loadtxt(clusterFile,skiprows=1,unpack=True)
vz,vra,vdec,vmean_r_sky,vmax_r_sky,vvol,vd_all,vd_neg,vd_min,vdenscontrast,vprob = np.loadtxt(voidFile,skiprows=1,unpack=True)

# transform coordinates
r = hp.rotator.Rotator(coord=['C','G']) # from equitorial to galactic
cgl,cgb = r(cra,cdec,lonlat=True)
vgl,vgb = r(vra,vdec,lonlat=True)

# collect filenames
overmassDirectory = '/shared/Data/PSG/'
ISWDirectory = '/shared/Data/PSG/hundred_point/'
directoryFiles = listdir(overmassDirectory)
overmassFiles = [file for file in directoryFiles if 'overmass' in file]

# just do one file for now
overmassFiles = [file for file in overmassFiles if 'R120' in file]
newProfile = True
newMap = True#False


# create healpix maps

zCent = 0.52 # used by PSG for median of GNS catalog
D_comov = comovInterp(zCent)
print 'redshift: ',zCent,', comoving dist: ',D_comov,' Mpc'

rmax = 1200 #Mpc, twice the rmax of overmass profiles
npoints = 101

for omFile in overmassFiles:
  # get ISW profile
  ISWProfileFile = 'ISWprofile'+omFile[8:] # 'overmass' is at start of omFile and has 8 characters
  if newProfile:
    print 'reading file ',omFile
    impactDomain,ISWRange = makeISWProfile(overmassDirectory+omFile,zCent,ISWDirectory+ISWProfileFile,rmax=rmax,npoints=npoints)
  else:
    print 'loading file ',ISWProfileFile
    impactDomain,ISWRange = np.loadtxt(ISWDirectory+ISWProfileFile,unpack=True)
  print 'impactDomain: ',impactDomain,' Mpc'
  print 'ISWRange: ',ISWRange,' DeltaT/T'

  if newMap:
    # find cutoff radius at cutoff*100% of maximum amplitude
    cutoff = 0.02
    maxAmp = ISWRange[0]
    impactLookup = interp1d(ISWRange,impactDomain,kind='cubic')
    print 'maxAmp, cutoff, product: ',maxAmp,cutoff,maxAmp*cutoff
    maxRadius = impactLookup(maxAmp*cutoff)
    print 'max radius: ',maxRadius, ' Mpc'
    #maxRadius = impactDomain[-1] #Mpc

    # for easier calculations, use maxRadius as arc length instead of tangential length
    #maxAngle = np.arctan(maxRadius/D_comov) #radians
    maxAngle = maxRadius/D_comov #radians
    print 'radius for disc: ',maxAngle*180/np.pi, ' degrees'

    # create ISW signal interpolation function
    
    ## what about central 10 Mpc? Use negative of smallest 2 r values and interp
    ## also add zero ISW point at 2* max radius
    #impactDomain = np.append([-1*impactDomain[1],-1*impactDomain[0]],np.append(impactDomain,2*impactDomain[-1]))
    #ISWRange = np.append([ISWRange[1],ISWRange[0]],np.append(ISWRange,[0]))
    #ISWinterp = interp1d(impactDomain,ISWRange,kind='cubic')
    ISWinterp = interp1d(impactDomain,ISWRange)

    impactTest = np.linspace(0,maxRadius,100)
    ISWTest = ISWinterp(impactTest)

    if doPlot:
      plt.plot(impactDomain,ISWRange) # data points used to make interpolation
      plt.plot(impactTest,ISWTest) # from interpolation function
      plt.xlabel('r [Mpc]')
      plt.ylabel('ISW: DeltaT / T')
      plt.show()

    numCV = 50
    #numCV = 2
    mapArray = np.zeros(hp.nside2npix(nside))
    mask     = np.zeros(hp.nside2npix(nside))
    cCentralVec = glgb2vec(cgl,cgb) #returns array of unit vectors
    vCentralVec = glgb2vec(vgl,vgb) #returns array of unit vectors
    for cvNum in range(numCV):
      print 'starting cv number ',cvNum+1

      # cluster
      myPixels = hp.query_disc(nside,cCentralVec[cvNum],maxAngle,nest=nested)
      numPix = myPixels.size
      for pixNum in range(numPix):
        myGb = latitudes[myPixels[pixNum]]
        myGl = longitudes[myPixels[pixNum]]
        myVec = glgb2vec(myGl,myGb) #returns unit vector
        angSep = np.arccos(np.dot(cCentralVec[cvNum],myVec))
        mapArray[myPixels[pixNum]] -= ISWinterp(angSep*D_comov) # -for sign error in PSG
        mask[    myPixels[pixNum]] = 1
      
      # void
      myPixels = hp.query_disc(nside,vCentralVec[cvNum],maxAngle,nest=nested)
      numPix = myPixels.size
      for pixNum in range(numPix):
        myGb = latitudes[myPixels[pixNum]]
        myGl = longitudes[myPixels[pixNum]]
        myVec = glgb2vec(myGl,myGb) #returns unit vector
        angSep = np.arccos(np.dot(vCentralVec[cvNum],myVec))
        mapArray[myPixels[pixNum]] += ISWinterp(angSep*D_comov) # +for sign error in PSG
        mask[    myPixels[pixNum]] = 1

    # 'overmass' is at start of omFile and has 8 characters; 'txt' is at end
    ISWMap = 'ISWmap_RING'+omFile[8:-3]+'fits' 
    hp.write_map(ISWDirectory+ISWMap,mapArray*CMBtemp,nest=nested,coord='GALACTIC')
    ISWMask = 'ISWmask_RING'+omFile[8:-3]+'fits' 
    hp.write_map(ISWDirectory+ISWMask,mask,nest=nested,coord='GALACTIC')



