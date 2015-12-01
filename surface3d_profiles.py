#! /usr/bin/env python

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from os import listdir

fig = plt.figure()
ax = fig.gca(projection='3d')
Y = np.linspace(10,160,16) # copies PSG values
X = np.logspace(-1,1,10)*40 # 4 to 400 Mpc
X, Y = np.meshgrid(X, Y)
Z = np.zeros([16,10])

ISWDirectory = '/shared/Data/PSG/'
directoryFiles = listdir(ISWDirectory)
ISWFiles = [file for file in directoryFiles if 'ISWprofile' in file]
ISWFiles.sort()
#print 'ISWFiles: ',ISWFiles

for fileNum in range(16):
  #for ISWFile in ISWFiles:
  impactDomain,ISWRange = np.loadtxt(ISWDirectory+ISWFiles[fileNum],unpack=True)
  Z[fileNum] = ISWRange * 1e6 #for plotting


surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_zlim(-1.2, 0.0)

ax.set_xlabel('r [Mpc]')
ax.set_ylabel('R [Mpc/h]')
ax.set_zlabel('DeltaT/T [10^-6]')
ax.set_title('ISW Profile for voids with delta_in = -1')

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

