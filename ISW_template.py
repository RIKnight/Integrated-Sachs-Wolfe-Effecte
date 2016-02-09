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
    Merged template map and index save files into one; ZK, 2016.01.31

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
    bigRvalues = np.linspace(1,16,16)*10 # Mpc/h
    ISWDirectory = '/Data/PSG/hundred_point/'
    ISWFilenames = ['ISWprofile_z0.52_R'+'{:0>3.0f}'.format(R)+'.txt' for R in bigRvalues]

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


    def __init__(self,avgRc=30,avgRv=95, newMaps=True, #should avgR be a function of z?
                 templateFile = None):
        """
        Purpose:
            Creates ISWtemplate object, based on parameters for a set of
                ISW creating clusters and voids
        Args:
            avgRc,avgRv: the value for clusters (voids) to scale the R_i values
                to so that <R> = avgRc (avgRv)
                This also sets the average value <delta_in(R)> = lambda^star(avgR)
                where lambda^star(R) is derived from PSG style template fitting
                avgRc: for clusters only, must be in range [20.8,94.4]
                    Default: 30
                avgRv: for voids only, must be in range [27.6,96.1]
                    Default: 95
            newMaps: set this to create new maps and indices.
                Otherwise, they will be loaded from files.
                Default: True
            templateFile: the name of the file that holds template maps and indices
                If this is not set, then no template will be saved, but one can
                    still be loaded from the default file.
                Default: ISW_template_maps_and_indices.npy
        Uses:
            file tfSN_mask_size.npy; created by template_fit_SN
        Returns:

        """
        # file names for saved maps and indices
        #mapFile = 'ISW_template_mapFile.npy'
        #indicesFile = 'ISW_template_indicesFile.npy'

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
        #self.unscaledRs  = np.concatenate((cmean_r_sky,vmean_r_sky))
        self.unscaledRCs  = cmean_r_sky
        self.unscaledRVs  = vmean_r_sky
        #self.unscaledDeltaIns = np.concatenate((cd_pos,vd_neg))
        self.unscaledDeltaInCs = cd_pos
        self.unscaledDeltaInVs = vd_neg
        #self.poissonProbs = np.concatenate((cprob,vprob)) # the first place to start scaling by significance
        # actually combining those is bad since they have different ranges
        self.cprob = cprob
        self.vprob = vprob

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

        # load data from PSG fitting
        ampSig = np.load('tfSN_mask_size.npy')  #[[[amp,stddev]]]; created by template_fit_SN
            # first index is mask number: eg. 0: m6110, 1: m9875, 2: mDelta
            # second index is ISW map number: eg. 10,20,30,40,...
        self.Rvalues = np.linspace(1,16,16)*10 # must match what is in template_fit_SN
        maskNum = 1 #small mask  # note the small mask fits have a higher significance

        # calculate scaled R and delta_in
        self.avgAmps   = ampSig[maskNum,:,0]        # amplitudes of PSG fitting
        self.avgSigmas = ampSig[maskNum,:,1]        # standard deviations of PSG fitting
        self.avgRc = avgRc                          # R value for scaling other R values
        self.avgRv = avgRv                          # R value for scaling other R values
        self.avgLambdaC = self.lambdaStar(avgRc)    # delta_in(R) for scaling other delta_in(R) values
        self.avgLambdaV = self.lambdaStar(avgRv)    # delta_in(R) for scaling other delta_in(R) values

        #self.scaledRs = self.unscaledRs*self.avgR/np.average(self.unscaledRs)
        #self.scaledDeltaIns = self.unscaledDeltaIns*self.avgLambda/np.average(self.unscaledDeltaIns)
        self.scaledRCs = self.unscaledRCs*self.avgRc/np.average(self.unscaledRCs)
        self.scaledDeltaInCs = self.unscaledDeltaInCs*self.avgLambdaC/np.average(self.unscaledDeltaInCs)
        self.scaledRVs = self.unscaledRVs*self.avgRv/np.average(self.unscaledRVs)
        self.scaledDeltaInVs = self.unscaledDeltaInVs*self.avgLambdaV/np.average(self.unscaledDeltaInVs)
        #print 'for target value averageR = ',averageR
        #for cvNum in range(self.nClusters+self.nVoids):
        #    print 'cv ',cvNum,': R = ',self.scaledRs[cvNum],', delta_in(R) = ',self.scaledDeltaIns[cvNum]
        print 'for target value avgRc = ',avgRc
        for cvNum in range(self.nClusters):
            print 'cv ',cvNum,': R = ',self.scaledRCs[cvNum],', delta_in(R) = ',self.scaledDeltaInCs[cvNum]
        print 'for target value avgRv = ',avgRv
        for cvNum in range(self.nVoids):
            print 'cv ',cvNum,': R = ',self.scaledRVs[cvNum],', delta_in(R) = ',self.scaledDeltaInVs[cvNum]

        # list of scaling ratios (previously calculated)
        zRatios = [ 1.46487317, 1.27569965, 1.096803,  0.92780813, 0.76830525,
                    0.61786465, 0.47603369, 0.34236417]
        zVals = [.4,.45,.5,.55,.6,.65,.7,.75]
        zRatioInterp = interp1d(zVals,zRatios)

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
                    impactDomain,ISWRange = profileInterp(self.scaledRCs[cvNum])
                else:
                    isVoid = True
                    impactDomain,ISWRange = profileInterp(self.scaledRVs[cvNum-self.nClusters])

                # scale ISWRange for redshift dependence
                ISWRange *= zRatioInterp(self.Zs[cvNum]) #all profiles based on z=0.52

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
            if templateFile is not None:
                self.saveTemplate(templateFile)
                #np.save(mapFile,self.maps)
                #np.save(indicesFile,self.indices)
            print 'time elapsed for 100 clusters/voids creation: ',(time.time()-startTime)/60.,' minutes'
        else: # do not make new maps; load maps
            if templateFile == None:
                templateFile = 'ISW_template_maps_and_indices.npy'
            self.loadTemplate(templateFile)
            #self.maps = np.load(mapFile)
            #self.indices = np.load(indicesFile)
        print 'done initializing cluster and void maps.'

    # methods for saving and loading map and index data
    def saveTemplate(self,saveFile):
        toSave = np.array([self.maps,self.indices])
        np.save(saveFile,toSave)
    def loadTemplate(self,loadFile):
        loaded = np.load(loadFile)
        self.maps = loaded[0]
        self.indices = loaded[1]

    # create interpolation functions for going between average R and average delta_in(R)
    def lambdaStar(self,R):
        f = interp1d(self.Rvalues,self.avgAmps)
        return f(R)
    def bigR(self,lambdaAmp):
        f = interp1d(self.avgAmps,self.Rvalues)
        return f(lambdaAmp)


    # method for evaluating map at given parameter set
    def getMap(self,amplitude,exponent):
        """
        Purpose:
            Transform object data and parameters into a full healpix map
        Args:
            amplitude: the linear scaling factor
            exponent: the exponential scaling factor

        Returns:
            numpy array containing a healpix ISW map with values in Kelvin
        """

        # choose something from catalog to use as significance of clusters/voids
        #significance = np.log(self.poissonProbs)
        cSig = np.zeros(self.nClusters) #np.log(self.cprob/np.median(self.cprob))
        vSig = np.zeros(self.nVoids)    #np.log(self.vprob/np.median(self.vprob))
        cMedian = np.median(self.cprob)
        vMedian = np.median(self.vprob)
        for cNum in range(self.nClusters):
            # have to loop through since numpy.power and numpy.log can't handle
            # fractional powers of negative numbers, but regular python can.  Annoying but true.
            cSig[cNum] = (self.cprob[cNum]/cMedian)**exponent
        for vNum in range(self.nVoids):
            vSig[vNum] = (self.vprob[vNum]/vMedian)**exponent
        significance = np.concatenate((cSig,vSig))
        #print 'significance: ',significance

        mapArray = np.zeros(hp.nside2npix(self.nside))
        nMaps = self.nClusters+self.nVoids
        for mapNum in range(nMaps):
            mapArray[self.indices[mapNum]] += amplitude*self.maps[mapNum]*significance[mapNum] #**exponent
        return mapArray *self.CMBtemp




################################################################################
# testing code

def test(avgRc=30,avgRv=95,nested=False,newMaps=True,
         templateFile='ISW_template_maps_and_indices.npy'):
    """

    Args:
        avgRc,avgRv: the value for clusters (voids) to scale the R_i values
                to so that <R> = avgRc (avgRv)
                This also sets the average value <delta_in(R)> = lambda^star(avgR)
                where lambda^star(R) is derived from PSG style template fitting
                avgRc: for clusters only, must be in range [20.8,94.4]
                    Default: 30
                avgRv: for voids only, must be in range [27.6,96.1]
                    Default: 95
        nested:
        newMaps:
        templateFile:

    Returns:

    """

    # test class ISWtemplate
    myTemplate = ISWtemplate(avgRc=avgRc,avgRv=avgRv,newMaps=newMaps,templateFile=templateFile)
    ISWmap = myTemplate.getMap(1.0,0.0) #amplitude,exponent

    # save map
    hp.write_map('firstMap.fits',ISWmap,nest=nested,coord='G')

    print 'done with test function.'




if __name__=='__main__':
  test()
