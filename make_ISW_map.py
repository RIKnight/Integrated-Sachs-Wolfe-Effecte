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
    Broke main function apart into separate functions potentially usable by
      other programs; Added makeMasks and showMap; ZK, 2016.01.19
    Added showProfile; added doTangent flag and procedures;
      added outer point kludge to ISW profile; nside1024 filenaming; ZK, 2016.01.26
    Removed map creation code from test function and dumped it into new
      function getCVmap; Created zCent loop;
      changed delta_z defalut to 0.3; ZK, 2016.01.29

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

def makeISWProfile(MFile,zCent,profileFile,noSave=False,rmax=400,npoints=101,delta_z=0.3):
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
      delta_z: the redshift difference from zCent for starting and stopping integration
        default: 0.3
    returns:
      impact: an array of impact parameters [Mpc]
      ISW: a corresponding array of DeltaT/T values (no units)
  """
  myCluster = ISW.ClusterVoid(MFile,zCent)
  #delta_z = 0.3 #0.2
  #impactDomain = np.logspace(-1,1,npoints)*rmax/10 # rmax/100 to rmax Mpc
  impactDomain = np.linspace(0,rmax,npoints)
  ISWRange = np.zeros(npoints)
  for pointnum in range(npoints):
    print 'calculating point number ',pointnum+1,' of ',npoints
    ISWRange[pointnum] = myCluster.overtemp(impactDomain[pointnum],delta_z)
  if not noSave:
    np.savetxt(profileFile,np.vstack((impactDomain,ISWRange)).T)
  return impactDomain,ISWRange

def getComDist(zMax=1.0,nSteps=5000):
  """
  Purpose:
      calculate an array of comoving distances for given z values
  Args:
      zMax: The maximum redshift to calculate out to
        Default: 1 (further than all clusters in survey)
      nSteps: The number of steps to use in the discrete sum
        which approximates integration
        Default: 5000
  Uses:
      ISWprofile.py as ISW for ClusterVoid parameters
      cosmography.py for comoving distance calculation
  Returns:
      zVals: an array of redshift values
      comDists: a corresponding array of comoving distances
  """
  # get parameters from ISW.ClusterVoid
  H_0     = ISW.ClusterVoid.H_0      #km/sec/Mpc
  Omega_M = ISW.ClusterVoid.Omega_M
  Omega_L = ISW.ClusterVoid.Omega_L
  #Omega_k = 0.0 # not included in ClusterVoid
  c_light = ISW.ClusterVoid.c_light  #km/sec
  #DH = c_light/H_0 # hubble distance in Mpc
  print 'H_0: ',H_0

  # find comoving distance to redshift in Mpc
  # currently has omega_k = 0 hardcoded
  zVals, comDists = cosmography.ComovingDistance(zMax,Omega_M,Omega_L,nSteps,H_0)
  return zVals,comDists


def getMapCoords(nside,nested):
  """
  Purpose:
      Load map coordinates from fits file
  Args:
      nside: must be 64 or 1024
      nested: True or False.
        the NESTED vs RING parameter for healpy functions
  Returns:
      the longitude,latitude of pixels as two healix maps
  """
  datadir = '/Data/'
  if nside == 64:
    if nested:
      coordfile = 'pixel_coords_map_nested_galactic_res6.fits'
    else: # 'RING'
      coordfile = 'pixel_coords_map_ring_galactic_res6.fits'
  elif nside == 1024:
    if nested:
      coordfile = 'pixel_coords_map_nested_galactic_res10.fits'
    else: # 'RING'
      coordfile = 'pixel_coords_map_ring_galactic_res10.fits'
  else:
    print 'error; unknown NSIDE: ',nside
    return 0
  # from header: field 1 is longitude, field 2 is latitude; both in degrees
  print 'loading coordinates from file ',coordfile
  longitudes,latitudes = hp.read_map(datadir+coordfile,(0,1),nest=nested)
  return longitudes,latitudes

def getGNScoords():
  """
  Purpose:
    extract void and cluster coordinates from GNS catalog and transform into
      galactic coordinate system
  Returns:
    cgl: cluster galactic longitude
    cgb: cluster galactic latitude
    vgl: void galactic longitude
    vgb: void galactic latitude
  """
  clusterFile = '/Data/Gr08_clustercat.txt'
  voidFile    = '/Data/Gr08_voidcat.txt'
  #clusterFile = '/Data/Gr08_clcat_trunc1.txt' #truncated catalog for 3 cl, 3 vo close together
  #voidFile    = '/Data/Gr08_vocat_trunc1.txt'
  #clusterFile = '/Data/Gr08_clcat_trunc2.txt' #truncated catalog for 2 cl, 2 vo far apart
  #voidFile    = '/Data/Gr08_vocat_trunc2.txt'
  cz,cra,cdec,cmean_r_sky,cmax_r_sky,cvol,cd_all,cd_pos,cd_max,cdenscontrast,cprob = np.loadtxt(clusterFile,skiprows=1,unpack=True)
  vz,vra,vdec,vmean_r_sky,vmax_r_sky,vvol,vd_all,vd_neg,vd_min,vdenscontrast,vprob = np.loadtxt(voidFile,skiprows=1,unpack=True)

  # transform catalog coordinates
  r = hp.rotator.Rotator(coord=['C','G']) # from equitorial to galactic
  cgl,cgb = r(cra,cdec,lonlat=True)
  vgl,vgb = r(vra,vdec,lonlat=True)

  return cgl,cgb,vgl,vgb


def makeMasks(nside=64,nested=False,ISWDir='/Data/PSG/hundred_point/'):
  """
  Purpose:
      Makes a set of masks around GNS coordinates of various apertures
  Args:
      nside:
      nested:
      ISWDir:

  Returns:
      writes healpix files to ISWDir containing masks
  """

  # load HEALpix coordinates file
  print 'NSIDE=',nside,' NESTED=',nested
  longitudes,latitudes = getMapCoords(nside,nested)

  # load GNS catalog coordinates
  cgl,cgb,vgl,vgb = getGNScoords()

  # set radii for apertures around coordinate locations
  radiiDeg = np.array([4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12]) #degrees
  #radiiDeg = np.array([5.0])
  radii = radiiDeg*np.pi/180. # converted to radians

  numCV = 50 # number of clusters and voids in catalog
  for radNum, radius in enumerate(radii):
      print 'starting radius ',radiiDeg[radNum],' degrees: '
      mask     = np.zeros(hp.nside2npix(nside))
      cCentralVec = glgb2vec(cgl,cgb) #returns array of unit vectors
      vCentralVec = glgb2vec(vgl,vgb) #returns array of unit vectors
      for cvNum in np.arange(numCV):
        #print 'starting (cluster,void) number ',cvNum+1
        # cluster
        myPixels = hp.query_disc(nside,cCentralVec[cvNum],radius,nest=nested)
        mask[myPixels] = 1
        # void
        myPixels = hp.query_disc(nside,vCentralVec[cvNum],radius,nest=nested)
        mask[myPixels] = 1
      radString = str("%04.1f" %radiiDeg[radNum])
      pixNumStr = str(int(np.sum(mask)))
      print 'number of pixels for radius '+radString+ ': '+pixNumStr
      maskFile = 'ISWmask_'+radString+'deg_'+pixNumStr+'pix.fits'
      hp.write_map(ISWDir+maskFile,mask,nest=nested,coord='GALACTIC')


def showMap(mapFile,nested=False,return_projected_map=False,
            title = 'ISW map over SDSS region'):
  """
  Purpose:
      make orthographic plots of maps or masks over the SDSS region
  Note:
      This is translated from IDL function sdss_plot.pro.  The IDL
        graphics are superior to the ones available in python, and
        the plotting functions have more options.
  Args:
      mapFile:
      nested:
      return_projected_map: pass to orthview, get back numpy map
        note: I don't know how to use this now 2016.01.19
      title: the title of the plot
  Returns:
      if return_projected_map is set: the projected map in a numpy array
  """


  #subTitle = 'equitorial coordinates with RA flipped'
  map = hp.read_map(mapFile,nest=nested)
  projected = hp.orthview(map,rot=[180,35],coord=['G','C'],half_sky=True,
                          nest=nested,flip='geo',title=title,
                          return_projected_map=return_projected_map)
  hp.graticule(dpar=30,dmer=30)
  plt.show()

  return projected

def showProfile(profileFile):
  r,ISW = np.loadtxt(profileFile,unpack=True)
  plt.plot(r,ISW*1e6)
  plt.xlabel('r [Mpc]')
  plt.ylabel('ISW signal *1e-6 [DeltaT/T]')
  plt.title('ISW profile in file '+profileFile)
  plt.show()

def getCVmap(nside,centralVec,maxAngle,latitudes,longitudes,impactDomain,ISWRange,
             D_comov,nest=False,isVoid=False,doTangent=False):
  """
  Purpose:
      Create an ISW map from an ISW profile and other information
  Args:
      nside:
      centralVec:
      maxAngle:
      latitudes: array of latitudes of the Healpix pixels
      longitudes: array of longitudes of the Healpix pixels
      impactDomain:
      ISWRange:
      D_comov:
      nest:
      isVoid:
      doTangent:

  Returns:
      myPixels: an array of indices that correspond to the indices in an intact
        healpix map which are covered by this ISW map
      mapISW: an array of ISW values with unit DeltaT/T (no units) that correspond
        to the locations specified by the indices in the myPixels array
  """
  # create ISW signal interpolation function
  ISWinterp = interp1d(impactDomain,ISWRange)

  # find which pixels to work on
  myPixels = hp.query_disc(nside,centralVec,maxAngle,nest=nest)
  numPix = myPixels.size

  # create data array only of pixels for this cluster/void
  mapISW= np.zeros(numPix)
  for pixNum in range(numPix):
    myGb = latitudes[myPixels[pixNum]]
    myGl = longitudes[myPixels[pixNum]]
    myVec = glgb2vec(myGl,myGb) #returns unit vector
    angSep = np.arccos(np.dot(centralVec,myVec))
    if doTangent:
      radSep = angSep*D_comov
    else:
      radSep = 2*D_comov*np.sin(angSep/2.)
    if isVoid:
      mapISW[pixNum] = ISWinterp(radSep)
    else:
      mapISW[pixNum] = ISWinterp(radSep)*-1
      #clusters get the -1 rather than the voids due to sign error in PSG
      # that ended up in clusterVoid.overtemp function
  return myPixels, mapISW


################################################################################
# testing code

def test(nested=False,doPlot=False,nside=64,doTangent=False):
  """
  Note that this is not a rigorous testing function
  Purpose:
      reads overmass files from overmass directory and creates ISW maps and masks
        using overmass profiles

  Args:
      nested:
      doPlot:
      NSIDE: must be 64 or 1024
      doTangent: set this to approximate arc lengths using tangent lines.
        Default: false.  Arc lengths are properly accounted for.
  Returns:
      writes ISW maps and masks to disk
  """

  CMBtemp = 2.7260 # +-0.0013 K (WMAP) Fixen, 2009

  # create comoving distance function
  zVals, comDists = getComDist(zMax=1.0,nSteps=5000)
  comovInterp = interp1d(zVals,comDists)

  # load HEALpix coordinates file
  print 'NSIDE=',nside,' NESTED=',nested
  longitudes,latitudes = getMapCoords(nside,nested)

  # load GNS catalog coordinates
  cgl,cgb,vgl,vgb = getGNScoords()

  # collect filenames
  overmassDirectory = '/Data/PSG/'
  ISWDirectory = '/Data/PSG/hundred_point/'
  directoryFiles = listdir(overmassDirectory)
  overmassFiles = [file for file in directoryFiles if 'overmass' in file]

  # just do one file for now
  #overmassFiles = [file for file in overmassFiles if 'PSGplot060' in file]
  #overmassFiles = [file for file in overmassFiles if 'PSGplot100' in file]
  #overmassFiles = [file for file in overmassFiles if 'R010' in file]
  #overmassFiles = [file for file in overmassFiles if 'R020' in file]
  #overmassFiles = [file for file in overmassFiles if 'R030' in file]
  #overmassFiles = [file for file in overmassFiles if 'R040' in file]
  #overmassFiles = [file for file in overmassFiles if 'R050' in file]
  #overmassFiles = [file for file in overmassFiles if 'R060' in file]
  #overmassFiles = [file for file in overmassFiles if 'R070' in file]
  #overmassFiles = [file for file in overmassFiles if 'R080' in file]
  #overmassFiles = [file for file in overmassFiles if 'R090' in file]
  #overmassFiles = [file for file in overmassFiles if 'R100' in file]
  #overmassFiles = [file for file in overmassFiles if 'R110' in file]
  #overmassFiles = [file for file in overmassFiles if 'R120' in file]
  #overmassFiles = [file for file in overmassFiles if 'R130' in file]
  #overmassFiles = [file for file in overmassFiles if 'R140' in file]
  #overmassFiles = [file for file in overmassFiles if 'R150' in file]
  #overmassFiles = [file for file in overmassFiles if 'R160' in file]
  newProfile = False#True
  newMap = True#False


  # create healpix maps

  #zList = [0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75]
  zList = [0.52] # used by PSG for median of GNS catalog
  rmax = 800     # Mpc, 2x PSGplot max
  #rmax = 1200    # Mpc, twice the rmax of overmass profiles
  npoints = 101  # number of points in the ISW profile
  delta_z = 0.3  # for limits of integration when making ISW profile
  cutoff = 0.02  # Maps extend to radius where amplitude = maxAmp*cutoff

  # loop over zList to create maps at each redshift
  for zCent in zList:
    zStr = str(zCent)  # this is sloppy formatting. eg: Want "0.40", not "0.4"
    print 'starting with z_cent = '+zStr
    D_comov = comovInterp(zCent)
    print 'redshift: ',zCent,', comoving dist: ',D_comov,' Mpc'
    for omFile in overmassFiles:
      # get ISW profile
      ISWProfileFile = 'ISWprofile_z'+zStr+omFile[8:] # 'overmass' is at start of omFile and has 8 characters
      if newProfile:
        print 'reading file ',omFile
        impactDomain,ISWRange = makeISWProfile(overmassDirectory+omFile,zCent,ISWDirectory+ISWProfileFile,
                                               rmax=rmax,npoints=npoints,delta_z=delta_z)
      else:
        print 'loading file ',ISWProfileFile
        impactDomain,ISWRange = np.loadtxt(ISWDirectory+ISWProfileFile,unpack=True)
      print 'impactDomain: ',impactDomain,' Mpc'
      print 'ISWRange: ',ISWRange,' DeltaT/T'

      if newMap:
        # add point in outer region for kludgy extrapolation, just as in ISWprofile.clusterVoid.__init__
        impactDomain = np.concatenate([impactDomain,[2*impactDomain[-1]]])
        ISWRange = np.concatenate([ISWRange,[0]]) # ramps down to zero at r=2*impactDomain[-1]

        # find cutoff radius at cutoff*100% of maximum amplitude
        maxAmp = ISWRange[0]
        impactLookup = interp1d(ISWRange,impactDomain)#,kind='cubic')
        print 'maxAmp, cutoff, product: ',maxAmp,cutoff,maxAmp*cutoff
        maxRadius = impactLookup(maxAmp*cutoff)
        print 'max radius: ',maxRadius, ' Mpc'

        #doTangent = False
        if doTangent: # for easier calculations, use maxRadius as arc length instead of tangential length
          maxAngle = maxRadius/D_comov #radians
        else: # use maxRadius as a chord length
          maxAngle = 2*np.arcsin(maxRadius/(2*D_comov))
        print 'radius for disc: ',maxAngle*180/np.pi, ' degrees'

        # visually check accuracy of interpolation function
        if doPlot:
          # create ISW signal interpolation function
          ISWinterp = interp1d(impactDomain,ISWRange) # same line as in getCVmap
          impactTest = np.linspace(0,maxRadius,100)
          ISWTest = ISWinterp(impactTest)

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
        for cvNum in np.arange(numCV):
          print 'starting cv number ',cvNum+1
          cIndices,cISW = getCVmap(nside,cCentralVec[cvNum],maxAngle,latitudes,longitudes,impactDomain,ISWRange,
                                   D_comov,nest=nested,isVoid=False,doTangent=doTangent)
          mapArray[cIndices] += cISW
          mask[cIndices] = 1
          vIndices,vISW = getCVmap(nside,vCentralVec[cvNum],maxAngle,latitudes,longitudes,impactDomain,ISWRange,
                                   D_comov,nest=nested,isVoid=True,doTangent=doTangent)
          mapArray[vIndices] += vISW
          mask[vIndices] = 1

        # 'overmass' is at start of omFile and has 8 characters; 'txt' is at end
        if nside == 64:
          ISWMap = 'ISWmap_RING_z'+zStr+omFile[8:-3]+'fits'
          hp.write_map(ISWDirectory+ISWMap,mapArray*CMBtemp,nest=nested,coord='GALACTIC')
          ISWMask = 'ISWmask_RING_z'+zStr+omFile[8:-3]+'fits'
          hp.write_map(ISWDirectory+ISWMask,mask,nest=nested,coord='GALACTIC')
        elif nside == 1024:
          ISWMap = 'ISWmap_RING_1024_z'+zStr+omFile[8:-3]+'fits'
          hp.write_map(ISWDirectory+ISWMap,mapArray*CMBtemp,nest=nested,coord='GALACTIC')
          ISWMask = 'ISWmask_RING_1024_z'+zStr+omFile[8:-3]+'fits'
          hp.write_map(ISWDirectory+ISWMask,mask,nest=nested,coord='GALACTIC')




if __name__=='__main__':
  test()

