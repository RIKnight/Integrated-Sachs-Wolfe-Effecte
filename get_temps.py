#! /usr/bin/env python
"""
    NAME:
        get_temps

    PURPOSE:
        Loads map and mask from fits files and coordinates from text files,
            measures temperatures of cluster and void stacks,
            and determines S/N ratio using random coordinates

    USES:
        make_ISW_map.py
        Gr08_clustercat.txt: cluster catalog
        Gr08_voidcat.txt: void catalog
        planckSmoothed.fits (for testing)
        pMaskSmoothed.fits (for testing)

    MODIFICATION HISTORY:
        Ported from IDL programs get_temps.pro, getsdssrnd.pro, checknoise.pro;
            by Z Knight, 2016.03.21

"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import numpy.random as rnd

import make_ISW_map as mm


def get_temps(mapFile,radius=4.0,plot=False,nested=False,nside=2048,maskFile='nofile'):
    """
    Purpose:
        Function for loading a planck temperature map and finding the
        temperatures at the locations of the GNS08 voids and clusters,
        using a compensated tophat filter
        Optionally displays the measured values as plots
    Args:
        mapFile: the healpix FITS file to analyze
        radius: the radius in degrees of the inner ring of the filter.
            Default = 4.0 degrees
        plot: set this to True to plot the cluster and void temperatures
            as kludgy lines
            ... not yet implemented! ...
        nested: the healpy NESTED vs. RING parameter
            Default: False
        nside: the nside parameter of the map in mapFile
            if maskFile is also used, this should have the same nside.
            Default: 2048
        maskFile: set this to the name of the mask to be applied.
            Acts as a weight for creating weighted averages.
            Must have same ordering, nsize, coordsys as the map1_in file
            Default: 'nofile' for no masking/weighting done.
    Note: IDL version has /POINT, which has not been ported here.

    Procedure:
        loads cluster and void catalogs and the specified map.
        Uses a compensated tophat filter of the specified radius to measure the
            temperatures at the catalog coordinates

    Example:
        temps = get_temps('awesome_map.fits', 3.6, maskFile='awesome_mask.fits')


    Returns: a 2 element array that contains the average cluster
        temperature followed by the average void temperature

    """

    # set mask control
    if maskFile == 'nofile':
        domask = False
    else:
        domask = True

    # get object coordinates from catalogs
    cgl,cgb,vgl,vgb = mm.getGNScoords()
    ncat = cgl.__len__()
    cCentralVec = mm.glgb2vec(cgl,cgb) #returns array of unit vectors
    vCentralVec = mm.glgb2vec(vgl,vgb) #returns array of unit vectors

    # create inner and outer radii for filter
    print 'Radius for compensated top-hat filter is',radius,' degrees.'
    radius1 = radius*np.pi/180. # deg to rad
    radius2 = radius1*np.sqrt(2) # for equal areas between

    # read maps
    print 'reading file ',mapFile
    CMBmap,CMBheader = hp.read_map(mapFile,nest=nested,h=True)
    if domask:
        print 'reading file ',maskFile
        mask,mheader = hp.read_map(maskFile,nest=nested,h=True)
        print 'applying mask to map . . .'
        CMBmap *= mask

    # create storage
    clistin    = np.zeros(ncat)  #within inner ring
    vlistin    = np.zeros(ncat)
    clistout   = np.zeros(ncat)  #between rings
    vlistout   = np.zeros(ncat)
    cweightin  = np.zeros(ncat)
    vweightin  = np.zeros(ncat)
    cweightout = np.zeros(ncat)
    vweightout = np.zeros(ncat)

    # loop through catalog objects
    for cvNum in range(ncat):
        print 'processing cluster and void ',cvNum+1,' of ',ncat,'...'
        c_listpix1 = hp.query_disc(nside,cCentralVec[cvNum],radius1,nest=nested)
        c_listpix2 = hp.query_disc(nside,cCentralVec[cvNum],radius2,nest=nested)
        v_listpix1 = hp.query_disc(nside,vCentralVec[cvNum],radius1,nest=nested)
        v_listpix2 = hp.query_disc(nside,vCentralVec[cvNum],radius2,nest=nested)

        clistin[cvNum]  = np.sum(CMBmap[c_listpix1])
        vlistin[cvNum]  = np.sum(CMBmap[v_listpix1])
        clistout[cvNum] = np.sum(CMBmap[c_listpix2]) - clistin[cvNum]
        vlistout[cvNum] = np.sum(CMBmap[v_listpix2]) - vlistin[cvNum]

        if domask:
            cweightin[cvNum]  = np.sum(mask[c_listpix1])
            vweightin[cvNum]  = np.sum(mask[v_listpix1])
            cweightout[cvNum] = np.sum(mask[c_listpix2]) - cweightin[cvNum]
            vweightout[cvNum] = np.sum(mask[v_listpix2]) - vweightin[cvNum]
        else:
            cweightin[cvNum]  = c_listpix1.size()
            vweightin[cvNum]  = v_listpix1.size()
            cweightout[cvNum] = c_listpix2.size()
            vweightout[cvNum] = v_listpix2.size()

    # collect weighted averages over pixels
    csumin  = clistin /cweightin
    csumout = clistout/cweightout
    vsumin  = vlistin /vweightin
    vsumout = vlistout/vweightout
    csum = csumin-csumout
    vsum = vsumin-vsumout

    # get averages, standard deviations, standard errors of means
    cavg = np.average(csum)
    vavg = np.average(vsum)
    #cstd = np.std(csum)
    #vstd = np.std(vsum)
    #csem = cstd/np.sqrt(ncat)
    #vsem = vstd/np.sqrt(ncat)

    return cavg,vavg

def getSDSSrnd():
    """
    Purpose:
        Creates a random point within approximate boundaries of the SDSS sky

    Procedure:
        uses boundaries 120 <= RA < 240, 0 <= Dec < 60
        RA is chosen from a uniform distribution
        Dec is the inverse cosine of a uniform distribution
        Totheter these create a uniform spherical distribution

    Returns:
        returns Right Ascention and Declination (in degrees) of one point on the sky

    """
    # region limits given in degrees
    RAMin = 120
    RAMax = 240
    DecMin = 30 #degrees from north pole
    DecMax = 90

    radperdeg = np.pi/180
    cosdecmin = np.cos(DecMin*radperdeg)
    cosdecmax = np.cos(DecMax*radperdeg)

    RA = rnd.uniform()*(RAMax-RAMin)+RAMin
    Dec = np.arccos(rnd.uniform()*(cosdecmax-cosdecmin)+cosdecmin) / radperdeg
    Dec = 90-Dec # back to standard declination definition

    return [RA, Dec]

def checknoise(mapFile,radius=4.0,ntries=1000,maskFile='nofile',noSNR=False,
               ncat=50,nside=2048,nested=False):
    """
    Purpose:
        Function for loading a planck map and finding the temperatures at
        the locations of random locations, using a compensated tophat filter
        Attempts to measure the noise associated with the tophat filter
        Optionally calculates SNR for GNS08 stacks
    Args:
        mapFile: the healpix file to analyze
        radius: the radius in degrees of the inner ring of the filter.
            Default = 4.0
        ntries: the number of repetitions
            Default=1000
        maskFile: set this to the name of the mask to be applied.
            Acts as a weight for creating weighted averages.
            Must have same ordering, nsize, coordsys as the map1_in file
            Default: no masking/weighting done.
        noSNR: set this to true to omit calculating SNR for GNS stacks
        ncat: the number of locations to average over
            Default: 50
        nside: the NSIDE parameter for finding healpix pixels
            should match that in data files if noSNR is not used
            Default: 2048
        nested: the NESTED vs RING healpy parameter
            Default: False

    Procedure:
        obtains 50 random locations within the approximate boundaries of the SDSS and
            measures the temperature of the map at those locaitons, and averages them.
        repeats this NTRIES number of times, then averages the results, as well as
        calculates the standard deviation of the distribution of results

    Returns: returns a two element array containing sigma significance (S/N)
        for clusters, voids in GNS08
        If noSNR is set then returns two element array containing the
            average and standard deviation of the simulated measurements

    """

    # set mask control
    if maskFile == 'nofile':
        domask = False
    else:
        domask = True

    # create inner and outer radii for filter
    print 'Radius for compensated top-hat filter is',radius,' degrees.'
    radius1 = radius*np.pi/180. # deg to rad
    radius2 = radius1*np.sqrt(2) # for equal areas between

    # read maps
    print 'reading file ',mapFile
    CMBmap,CMBheader = hp.read_map(mapFile,nest=nested,h=True)
    if domask:
        print 'reading file ',maskFile
        mask,mheader = hp.read_map(maskFile,nest=nested,h=True)
        print 'applying mask to map . . .'
        CMBmap *= mask

    # create storage
    clistin    = np.zeros(ncat)  #within inner ring
    clistout   = np.zeros(ncat)  #between rings
    cweightin  = np.zeros(ncat)
    cweightout = np.zeros(ncat)

    measurements = np.zeros(ntries)

    for trial in range(ntries):
        print 'measurement trial ',trial+1,' of ',ntries
        for n in range(ncat):
            myRADec = getSDSSrnd()

            # transform catalog coordinates
            r = hp.rotator.Rotator(coord=['C','G']) # from equitorial to galactic
            gl,gb = r(myRADec[0],myRADec[1],lonlat=True)
            centralVec = mm.glgb2vec(gl,gb) #returns unit vector

            listpix1 = hp.query_disc(nside,centralVec,radius1,nest=nested)
            listpix2 = hp.query_disc(nside,centralVec,radius2,nest=nested)
            clistin[n]  = np.sum(CMBmap[listpix1])
            clistout[n] = np.sum(CMBmap[listpix2]) - clistin[n]
            if domask:
                cweightin[n]  = np.sum(mask[listpix1])
                cweightout[n] = np.sum(mask[listpix2]) - cweightin[n]
            else:
                cweightin[n]  = listpix1.__len__()
                cweightout[n] = listpix2.__len__()

        # collect weighted averages over pixels
        csumin  = clistin /cweightin
        csumout = clistout/cweightout
        csum = csumin-csumout
        print 'random sample: avg:',np.mean(csum),' muK'#, stddev: ',np.std(clist),' muK'
        measurements[trial] = np.mean(csum)

    myavg = np.mean(measurements)
    mystd = np.std(measurements)
    print  'measurements: avg:',myavg,' muK, stddev: ',mystd,' muK'#, sem: ',mystd/np.sqrt(ncat)

    # unfortutnately this reads the fits files again
    if not noSNR:
        GNS08_temps = get_temps(mapFile,radius=radius,nside=nside,
                                maskFile=maskFile,nested=nested)
        print 'map: ',mapFile
        signif = GNS08_temps/mystd
        print 'For GNS08 stacks: clusters: ',signif[0],'; voids: ',signif[1]
        return signif

    return myavg,mystd






################################################################################
# testing code

def test(nested=False):

    # test map previously masked and smoothed by gauss_beam 30'
    #testMap = '/Data/planckMaskedSmoothed.fits'
    # test map previously smoothed by gauss_beam 30'
    testMap = '/Data/planckSmoothed.fits'
    # test mask previously smoothed by gauss_beam 30'
    testMask = '/Data/pMaskSmoothed.fits'

    # test get_temps
    #result = get_temps(testMap,radius=3.5,maskFile=testMask,nside=2048,nested=nested)
    #print 'get_temps result: ',result

    # test getSDSSrnd
    result = getSDSSrnd()
    print 'getSDSSrnd result: ',result

    # 2d histogram in healpix map; check uniformity
    nside = 32
    nrnd = 1000#0
    title = 'test of random coordinate generator'
    mapArray = np.zeros(hp.nside2npix(nside))
    for drawNum in range(nrnd):
        print 'draw ',drawNum+1,' of ',nrnd
        randomCoord = getSDSSrnd()
        print 'random coord: ',randomCoord
        randomVec = mm.glgb2vec(randomCoord[0],randomCoord[1]) #returns array of unit vectors
        randomPix = hp.query_disc(nside,randomVec,1e-5,nest=nested,inclusive=True)
        print 'random pixel: ',randomPix[0]
        mapArray[randomPix[0]] += 1.0
    projected = hp.orthview(mapArray,rot=[180,35],coord=['C'],half_sky=True, #,coord=['G','C']
                          nest=nested,flip='geo',title=title)
    hp.graticule(dpar=30,dmer=30)
    plt.show()

    # test checknoise
    cSN,vSN = checknoise(testMap,radius=3.5,ntries=100,maskFile=testMask)
    print 'Signal to Noise ratios: ', cSN, vSN

if __name__=='__main__':
  test()

