#! /usr/bin/env python
"""
  NAME:
    redshift_check

  PURPOSE:
    opens ISW profile files created for various redshifts and compares them.
        Looks for ratios and... ??

  USES:

  MODIFICATION HISTORY:
    Written by Z Knight, 2016.02.04

"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.interpolate import interp1d
from os import listdir
import time  # for measuring duration

#import ISWprofile as ISW
import make_ISW_map as mm

def test():

    # get filenames
    profileDirectory = '/Data/PSG/hundred_point/'
    dirFiles = listdir(profileDirectory)
    profileFiles = [profileDirectory+file for file in dirFiles if 'profile' in file]
    profileFiles = [file for file in profileFiles if '.txt' in file]# and 'z0.52' not in file]
    profileFiles = np.sort(profileFiles)

    R060Files = [file for file in profileFiles if 'R060' in file]
    print 'R060 files: ',R060Files
    R050Files = [file for file in profileFiles if 'R050' in file]
    print 'R050 files: ',R050Files

    nProfiles = R060Files.__len__()
    R060Profiles = np.zeros((nProfiles,101,2))
    R050Profiles = np.zeros((nProfiles,101,2))
    for profileNum in range(nProfiles):
        R060Profiles[profileNum] = np.loadtxt(R060Files[profileNum])
        R050Profiles[profileNum] = np.loadtxt(R050Files[profileNum])

    #for profileNum in range(nProfiles):
    #    plt.plot(R060Profiles[profileNum,:,0],R060Profiles[profileNum,:,1])
    #plt.show()

    # check ratios
    plt.figure(1)
    plt.subplot(211)
    for profileNum in range(nProfiles):
        plt.plot(R060Profiles[profileNum,:,0],R060Profiles[profileNum,:,1]/R060Profiles[0,:,1])
    plt.ylabel('r [Mpc]')
    plt.xlabel('amplitude(i)/amplitude(0)')
    plt.subplot(212)
    for profileNum in range(nProfiles):
        plt.plot(R050Profiles[profileNum,:,0],R050Profiles[profileNum,:,1]/R050Profiles[0,:,1])
    plt.ylabel('r [Mpc]')
    plt.xlabel('amplitude(i)/amplitude(0)')
    plt.show()

    print R060Profiles[:,0,1]/R060Profiles[3,0,1]

if __name__=='__main__':
  test()

