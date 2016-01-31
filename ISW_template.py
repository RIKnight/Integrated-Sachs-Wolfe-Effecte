#! /usr/bin/env python
"""
  NAME:
    ISW_template

  PURPOSE:
    ISW template objects will have the parameters taken from the GNS catalogs.
    These are meant to be used in a nonlinear template_fit minimization
    ISW template objects will have a method to produce a template based on
        the two parameters of the model
        Pixel values will be DeltaT/T * 2.726K

  USES:
    #ISWprofile.py
    #cosmography.py
    make_ISW_map.py

  MODIFICATION HISTORY:
    Written by Z Knight, 2016.01.29

"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.interpolate import interp1d
#from os import listdir
import time  # for measuring duration

#import ISWprofile as ISW
#import cosmography
import make_ISW_map as mm


def profileInterp(bigR):
    """
    Purpose:
        Interpolate between ISW profiles of various R values.
    Note:
        This function needs to be expanded to also include redshift as a second parameter.
        This will necessarily involve more sets of profiles at various redshifts.
        The current profile set is at redshift z=0.52
        Also note that this is not set up to handle bigR values outside the range [10,160]
    Args:
        bigR: the PSG R parameter in Mpc/h
    Returns:
        radii: an array of radius values in Mpc
        DeltaToverT: an array of DeltaT/T (no units) values corresponding to radius values
    """
    bigRvalues = np.array([10,20,30,40,50,60,70,80,90,100,
                           110,120,130,140,150,160]) # Mpc/h
    ISWDirectory = '/Data/PSG/hundred_point/'
    ISWFilenames = ['ISWprofile_R'+'{:0>3d}'.format(R)+'.txt' for R in bigRvalues]

    if bigR in bigRvalues:
        ISWProfileFile = ISWFilenames[np.where(bigRvalues == bigR)[0]]
        print 'loading file (exact match) ',ISWProfileFile
        impactDomain,ISWRange = np.loadtxt(ISWDirectory+ISWProfileFile,unpack=True)
    else: # interpolate between neighboring profiles
        lower = np.where(bigRvalues < bigR)[0]
        ISWProfileFileL = ISWFilenames[lower[-1]]
        print 'loading file (lower bound) ',ISWProfileFileL
        impactDomainL,ISWRangeL = np.loadtxt(ISWDirectory+ISWProfileFileL,unpack=True)
        upper = np.where(bigRvalues > bigR)[0]
        ISWProfileFileU = ISWFilenames[upper[0]]
        print 'loading file (upper bound)',ISWProfileFileU
        impactDomainU,ISWRangeU = np.loadtxt(ISWDirectory+ISWProfileFileU,unpack=True)

        slopes = (ISWRangeU-ISWRangeL) / (bigRvalues[upper[0]]-bigRvalues[lower[-1]])
        ISWRange = ISWRangeL + slopes*(bigR-bigRvalues[lower[-1]])  #linear interpolation
        impactDomain = impactDomainL
    return impactDomain,ISWRange



################################################################################
# the ISW template class

class ISWtemplate:


    def __init__(self,averageR=55,newMaps=True,noSave=False):  #should averageR be a function of z?
        """
        Purpose:
            Creates ISWtemplate object, based on parameters for a set of
                ISW creating clusters and voids
        Args:
            averageR: the value to scale the set R_i to so that <R> = averageR
                This also sets the average value <delta_in(R)> = lambda^star(averageR)
                where lambda^star(R) is derived from PSG style template fitting
                Default: 55 Mpc/h
            newMaps: set this to create new maps and indices.
                Otherwise, they will be loaded from files.
                Default: True
            noSave: set this to omit saving the maps and indices
                Default: False
        Uses:
            file tfSN_result.npy; created by template_fit_SN
        Returns:

        """
        # file names for saved maps and indices
        mapFile = 'ISW_template_mapFile.npy'
        indicesFile = 'ISW_template_indicesFile.npy'

        # healpix parameters
        nested=False # just don't change this.
        nside=64     # nor this.
        self.nside = nside

        # read catalogs
        clusterFile = '/Data/Gr08_clustercat.txt'
        voidFile    = '/Data/Gr08_voidcat.txt'
        cz,cra,cdec,cmean_r_sky,cmax_r_sky,cvol,cd_all,cd_pos,cd_max,cdenscontrast,cprob = np.loadtxt(clusterFile,skiprows=1,unpack=True)
        vz,vra,vdec,vmean_r_sky,vmax_r_sky,vvol,vd_all,vd_neg,vd_min,vdenscontrast,vprob = np.loadtxt(voidFile,skiprows=1,unpack=True)

        # transform catalog coordinates
        r = hp.rotator.Rotator(coord=['C','G']) # from equitorial to galactic
        cgl,cgb = r(cra,cdec,lonlat=True)
        vgl,vgb = r(vra,vdec,lonlat=True)

        # dump into object variables
        self.nClusters=50 # these are the lengths of the full catalog
        self.nVoids=50
        self.GLs = np.concatenate((cgl,vgl))
        self.GBs = np.concatenate((cgb,vgb))
        self.Zs  = np.concatenate((cz,vz))
        self.unscaledRs  = np.concatenate((cmean_r_sky,vmean_r_sky))
        self.unscaledDeltaIns = np.concatenate((cd_pos,vd_neg))
        self.poissonProbs = np.concatenate((cprob,vprob)) # the first place to start scaling by significance

        # convert catalog coordinates into unit vectors
        centralVecs = mm.glgb2vec(self.GLs,self.GBs)

        # other parameters
        self.CMBtemp = 2.7260 # +-0.0013 K (WMAP) Fixen, 2009
        doTangent = False # controls how radii from centers of clusters are mapped onto CMB sphere

        # load HEALpix coordinates file
        print 'NSIDE=',nside,' NESTED=',nested
        longitudes,latitudes = mm.getMapCoords(nside,nested)

        # create comoving distance function
        zVals, comDists = mm.getComDist(zMax=1.0,nSteps=5000)
        comovInterp = interp1d(zVals,comDists)

        # redshift is currently locked at this value to be compatible with exising ISWprofile files
        zCent = 0.52 # used by PSG for median of GNS catalog
        D_comov = comovInterp(zCent)
        print 'redshift: ',zCent,', comoving dist: ',D_comov,' Mpc'

            # I probably don't need the mask.  Remove this except for maskNum and ISWlabels
            # get mask
            #self.maskDir = '/Data/covariance_matrices/'
            #self.maskFiles = np.array(['ISWmask6110_RING.fits','ISWmask9875_RING.fits',
            #                           'ISWmask9875minus6110_RING.fits'])
            #self.maskLabels = np.array(['small','large','delta']) # must match what is in template_fit_SN

        self.ISWlabels  = np.array([10,40,60,80,100,120,160]) # must match what is in template_fit_SN
        maskNum = 0 #small mask  # note the small mask fits have a higher significance

            #print 'reading mask file ',self.maskFiles[maskNum]
            #mask = hp.read_map(self.maskDir+self.maskFiles[maskNum],nest=nested)
            #nPixels = np.sum(mask)

        # calculate scaled R and delta_in
        # for now, averageR is a single value.
        # May expand to be averageR(z) once more R(delta_in, z) averages have been found.
        ampSig = np.load('tfSN_result.npy')  #[[[amp,stddev]]]; created by template_fit_SN
            # first index is mask number: eg. 0: m6110, 1: m9875, 2: mDelta
            # second index is ISW map number: eg. 10,20,30,40,...
        self.avgAmps   = ampSig[maskNum,:,0] # amplitudes of PSG fitting
        self.avgSigmas = ampSig[maskNum,:,1] # standard deviations of PSG fitting
        self.avgR = averageR                        # R value for scaling other R values
        self.avgLambda = self.lambdaStar(averageR)  # delta_in(R) for scaling other delta_in(R) values

        self.scaledRs = self.unscaledRs*self.avgR/np.average(self.unscaledRs)
        self.scaledDeltaIns = self.unscaledDeltaIns*self.avgLambda/np.average(self.unscaledDeltaIns)

        # create array for each object ( much of this is copied from make_ISW_map.test() )
        if newMaps:
            startTime = time.time()
            mapsList = [] #empty list to be filled with 100 lists
            indicesList = []
            nMaps = self.nClusters+self.nVoids
            for cvNum in range(nMaps):
                print 'starting cluster or void number ',cvNum+1,' of ',nMaps
                if cvNum < self.nClusters:
                    isVoid = False
                else:
                    isVoid = True
                impactDomain,ISWRange = profileInterp(self.scaledRs[cvNum])

                # add point in outer region for kludgy extrapolation, just as in ISWprofile.clusterVoid.__init__
                impactDomain = np.concatenate([impactDomain,[2*impactDomain[-1]]])
                ISWRange = np.concatenate([ISWRange,[0]]) # ramps down to zero at r=2*impactDomain[-1]

                # find cutoff radius at cutoff*100% of maximum amplitude
                cutoff = 0.02
                maxAmp = ISWRange[0]
                impactLookup = interp1d(ISWRange,impactDomain)#,kind='cubic')
                print '  maxAmp, cutoff, product: ',maxAmp,cutoff,maxAmp*cutoff
                maxRadius = impactLookup(maxAmp*cutoff)
                print '  max radius: ',maxRadius, ' Mpc'
                if doTangent: # for easier calculations, use maxRadius as arc length instead of tangential length
                    maxAngle = maxRadius/D_comov #radians
                else: # use maxRadius as a chord length
                    maxAngle = 2*np.arcsin(maxRadius/(2*D_comov))
                print '  radius for disc: ',maxAngle*180/np.pi, ' degrees'


                myPixels,mapISW = mm.getCVmap(nside,centralVecs[cvNum],maxAngle,latitudes,longitudes,
                                              impactDomain,ISWRange,D_comov,
                                              nest=nested,isVoid=isVoid,doTangent=doTangent)
                mapsList.append(mapISW)
                indicesList.append(myPixels)
            self.maps = np.array(mapsList)
            self.indices = np.array(indicesList)
            if not noSave:
                np.save(mapFile,self.maps)
                np.save(indicesFile,self.indices)
            print 'time elapsed for 100 clusters/voids creation: ',(time.time()-startTime)/60.,' minutes'
        else: # load maps
            self.maps = np.load(mapFile)
            self.indices = np.load(indicesFile)
        print 'done initializing cluster and void maps.'


    # method for evaluating map at given parameter set
    def getMap(self,amplitude,exponent):
        """
        Purpose:
            Transform object data and parameters into a full healpix map
        Args:
            amplitude: the linear scaling factor
            exponent: the exponential scaling factor

        Returns:
            numpy array containing a healpix map
        """

        # choose something from catalog to use as significance of clusters/voids
        significance = np.log(self.poissonProbs)

        mapArray = np.zeros(hp.nside2npix(self.nside))
        nMaps = self.nClusters+self.nVoids
        for mapNum in range(nMaps):
            mapArray[self.indices[mapNum]] += amplitude*self.maps[mapNum]*significance[mapNum]**exponent
        return mapArray


    # create interpolation functions for going between average R and average delta_in(R)
    def lambdaStar(self,R):
        f = interp1d(self.ISWlabels,self.avgAmps)
        return f(R)
    def bigR(self,lambdaAmp):
        f = interp1d(self.avgAmps,self.ISWlabels)
        return f(lambdaAmp)



################################################################################
# testing code

def test(nested=False,newMaps=True,noSave=False):

    # test class ISWtemplate
    myTemplate = ISWtemplate(averageR=55,newMaps=newMaps,noSave=noSave)
    ISWmap = myTemplate.getMap(1.0,0.0) #amplitude,exponent

    # save map
    hp.write_map('firstMap.fits',ISWmap,nest=nested,coord='G')

    print 'done with test function.'




if __name__=='__main__':
  test()
