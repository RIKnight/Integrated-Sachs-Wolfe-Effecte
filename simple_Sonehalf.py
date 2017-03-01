#! /usr/bin/env python
"""
Name:
  simple_Sonehalf
Purpose:
  create simplistic CMB simulations and analyze S_{1/2} properties
  shows trend when increasing l_min
  Also plots C_2 and S_{1/2} together on 2d plot
Uses:
  healpy
  get_crosspower.py
  spice, ispice.py
  legprodint.py (legendre product integral)
  ramdisk.sh    (creates and deletes ramdisks)
  plot2Ddist    (plot 2D distributions)
    Note: this is a modified version that does not include pymc
Inputs:
  Data files as specified in get_crosspower.loadCls function
Outputs:
  creates plot of S_{1/2} distributions with varying values of l_min
Modification History:
  Written by Z Knight, 2016.09.07
  Added cutSky option; ZK, 2016.11.16
  Removed titles from plots; ZK, 2016.11.17
  Changed histogram linewidth to 2, l to \ell; 
    Added C_2 plotting; ZK, 2016.12.13
  Added newSC2 and modified plotting; ZK, 2017.01.19
  Added plot2Ddist for joint 2d p-value calculation; ZK, 2017.02.23
  Added nGrid, bw_method, axSize parameters for plot2Ddist; ZK, 2017.02.24
  Added second call to plot2Ddist and created inset zoom; ZK, 2017.02.27
  Added pValue for calculation of C2, Sonehalf p-values; ZK, 2017.02.28
  
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time # for measuring duration
import subprocess # for calling RAM Disk scripts
import get_crosspower as gcp
from numpy.polynomial.legendre import legval # for C_l -> C(theta) conversion
from numpy.polynomial.legendre import legfit # for C(theta) -> C_l conversion
from ispice import ispice
from legprodint import getJmn
from scipy.interpolate import interp1d
from matplotlib import cm # color maps for 2d histograms
import corner             # for corner plot
import plot2Ddist         # for 2d distribution plotting, etc.


def getSsim(ell,Cl,lmax=100,cutSky=False):
  """
  Purpose:
    create simulated S_{1/2} from input power spectrum
  Note:
    this calculates Jmn every time it is run so should not be used for ensembles
  Procedure:
    simulates full sky CMB, measures S_{1/2}
  Inputs:
    ell: the l values for the power spectrum
    Cl: the power spectrum
    lmax: the maximum ell value to use in calculation
      Default: 100
    cutSky: set to True to convert to real space, apply mask, etc.
      Default: False
      Note: true option not yet implemented
  Returns:
    simulated S_{1/2}
  """
  # get Jmn matrix for harmonic space S_{1/2} calc.
  myJmn = getJmn(lmax=lmax)[2:,2:] # do not include monopole, dipole

  #alm_prim,alm_late = hp.synalm((primCl,lateCl,crossCl),lmax=lmax,new=True)
  almSim = hp.synalm(Cl,lmax=lmax) # question: does this need to start at ell[0]=1?
  ClSim = hp.alm2cl(almSim)

  return np.dot(ClSim[2:],np.dot(myJmn,ClSim[2:]))


def pValue(myArray,threshold):
  """
  Purpose:
    create p-value using ensemble and reference point
  Inputs:
    myArray: numpy array containing ensemble values
    threshold: value to compare to ensemble to calculate p-value of
  Returns:
    the p-value

  """
  nUnder = 0 # will also include nEqual
  nOver  = 0
  for sim in myArray:
     if sim > threshold: 
       nOver  += 1
     else: 
       nUnder += 1
  #print "nUnder: ",nUnder,", nOver: ",nOver
  return nUnder/float(nUnder+nOver)


################################################################################
# testing code

def test(nSims=100,lmax=100,lmin=2,partialMax=4,useCLASS=1,useLensing=1,
         cutSky=True,myNSIDE=128,newSC2=True,saveFile='simpleSonehalfC2.npy',
         nGrid=100):
  """
    Purpose:
      function for testing S_{1/2} calculations
    Inputs:
      nSims: the number of simulations to do
        Overriden if newSC2 = False
        Default: 100
      lmax: the highest l to use in the calculation
        Default: 100
      lmin: the lowest l to use in the calculation
        Default: 2
      partialMax: the maximum l to use for partial Sonehalf plots
        must be more than lmin
        Overriden if newSC2 = False
        Default: 4
      useCLASS: set to 1 to use CLASS Cl, 0 for CAMB
        Default: 1
      useLensing: set to 1 to use lensed Cls
        Default: 1
      cutSky: set to True to do cut-sky sims
        Default: True
      myNSIDE: HEALPix parameter for simulated maps if cutSky=True
        Default: 128
      newSC2: set to True to simulate new ensemble and save S,C2 results 
        in file, False to skip simulation and load previous results
        If false, values of nSims and partialMax will come from file
        Default: True
      saveFile: filename to save S,C2 result if newSC2 is true, to load if false
        Default: 'simpleSonehalfC2.npy'
      nGrid: to pass to plot2Ddist; controls grid for binning for contours
        Default: 100
  """
  # get power spectrum
  # starts with ell[0]=2
  ell,fullCl,primCl,lateCl,crossCl = gcp.loadCls(useCLASS=useCLASS,useLensing=useLensing)

  # fill beginning with zeros
  startEll = int(ell[0])
  ell = np.append(np.arange(startEll),ell)
  Cl  = np.append(np.zeros(startEll),fullCl)
  #conv = ell*(ell+1)/(2*np.pi)

  # Note: optimizeSx2 includes a multiplication of Cl by (beam*window)**2 at this point, 
  #   but in this program I'm omitting it.  Why?  Effects are small, esp. at low ell

  # get Jmn matrix for harmonic space S_{1/2} calc.
  myJmn = getJmn(lmax=lmax) # do not include monopole, dipole

  if cutSky:
    # yeah.. disk access is annoying so...
    RAMdisk     = '/Volumes/ramdisk/'
    ClTempFile  = RAMdisk+'tempCl.fits'
    mapTempFile = RAMdisk+'tempMap.fits'
    mapDegFile  = RAMdisk+'smicaMapDeg.fits' #created by sim_stats.getSMICA
    maskDegFile = RAMdisk+'maskMapDeg.fits'  #created by sim_stats.getSMICA

    # create RAM Disk for SpICE and copy these files there using bash
    RAMsize = 4 #Mb
    ramDiskOutput = subprocess.check_output('./ramdisk.sh create '+str(RAMsize), shell=True)
    print ramDiskOutput
    diskID = ramDiskOutput[31:41] # this might not grab the right part; works for '/dev/disk1'
    subprocess.call('cp smicaMapDeg.fits '+RAMdisk, shell=True)
    subprocess.call('cp maskMapDeg.fits ' +RAMdisk, shell=True)

    ispice(mapDegFile,ClTempFile,maskfile1=maskDegFile,subav="YES",subdipole="YES")
    Clsmica = hp.read_cl(ClTempFile)
  else:
    ClTempFile  = 'tempCl.fits'
    mapTempFile = 'tempMap.fits'
    mapDegFile  = 'smicaMapDeg.fits' #created by sim_stats.getSMICA 
    maskDegFile = 'maskMapDeg.fits'  #created by sim_stats.getSMICA
    ispice(mapDegFile,ClTempFile,subav="YES",subdipole="YES")
    Clsmica = hp.read_cl(ClTempFile)


  # collect results
  if newSC2:
    sEnsemblePartial = np.zeros([nSims,partialMax+1])
    C2Ensemble = np.zeros(nSims)
    for i in range(nSims):
      print "starting sim ",i+1," of ",nSims,"... "

      almSim = hp.synalm(Cl,lmax=lmax) # should start with ell[0] = 0
      if cutSky:
        mapSim = hp.alm2map(almSim,myNSIDE,lmax=lmax)
        hp.write_map(mapTempFile,mapSim)
        ispice(mapTempFile,ClTempFile,maskfile1=maskDegFile,subav="YES",subdipole="YES")
        ClSim = hp.read_cl(ClTempFile)
      else:  
        ClSim = hp.alm2cl(almSim)

      for myLmin in range(lmin,partialMax+1):
        sEnsemblePartial[i,myLmin] = np.dot(ClSim[myLmin:lmax+1],
                                  np.dot(myJmn[myLmin:,myLmin:],ClSim[myLmin:lmax+1]))
      C2Ensemble[i] = ClSim[2]
    # save results
    np.save(saveFile,np.hstack((np.array([C2Ensemble]).T,sEnsemblePartial)) )

  else: # load from file
    sEnsemblePartial = np.load(saveFile)
    C2Ensemble = sEnsemblePartial[:,0]
    sEnsemblePartial = sEnsemblePartial[:,1:]
    nSims = sEnsemblePartial.shape[0]
    partialMax = sEnsemblePartial.shape[1]-1

  if cutSky:
    # free the RAM used by SpICE's RAM disk
    ramDiskOutput = subprocess.check_output('./ramdisk.sh delete '+diskID, shell=True)
    print ramDiskOutput





  # plot results
  
  print 'plotting S_{1/2} distributions... '

  #myBins = np.logspace(2,7,100)
  myBins = np.logspace(2,6,100)
  #plt.axvline(x=6763,color='b',linewidth=3,label='SMICA inpainted')
  #plt.axvline(x=2145,color='g',linewidth=3,label='SMICA masked')
  #plt.hist(sEnsembleFull,bins=myBins,color='b',histtype='step',label='full sky')
  #plt.hist(sEnsembleCut, bins=myBins,color='g',histtype='step',label='cut sky')

  myColors = ('g','b','r','c','m','k')#need more?  prob. not.
  myLines  = ('-','--','-.')#need more?
  for myEll in range(lmin,partialMax+1):
    plt.hist(sEnsemblePartial[:,myEll],bins=myBins,histtype='step',
        label=r'sims: $\ell_{\rm min}$ = '+str(myEll),
        color=myColors[myEll-lmin],linestyle=myLines[myEll-lmin],linewidth=2)

    Sonehalf = np.dot(Clsmica[myEll:lmax+1],
                  np.dot(myJmn[myEll:,myEll:],Clsmica[myEll:lmax+1])) *1e24
    plt.axvline(x=Sonehalf,linewidth=3,label=r'SMICA: $\ell_{\rm min}$='+str(myEll),
        color=myColors[myEll-lmin],linestyle=myLines[myEll-lmin])
    # calculate and print p-value
    pval = pValue(sEnsemblePartial[:,myEll],Sonehalf)
    print 'l_min: ',myEll,', Sonehalf: ',Sonehalf,', p-value: ',pval

  plt.gca().set_xscale("log")
  plt.legend()
  plt.xlabel(r'$S_{1/2} (\mu K^4)$')
  plt.ylabel('Counts')
  plt.xlim((500,10**6))
  if cutSky:
    sName = ' cut-sky'
  else:
    sName = ' full-sky'
  #plt.title(r'$S_{1/2}$ of '+str(nSims)+sName+' simulated CMBs')
  plt.show()


  print 'plotting C_2 vs. S_{1/2} histogram... '

  SMICAvals = (np.log10(2145),171.8) # KLUDGE!!! #moved to earlier in program
  SonehalfLabel = "$log_{10}(\ S_{1/2}\ /\ (\mu K)^4\ )$"
  C2Label       = "$C_2\ /\ (\mu K)^2$"
  C2Label3      = "$C_2\ /\ (10^3 (\mu K)^2)$"

  log10SonehalfEnsemble = np.log10(sEnsemblePartial[:,lmin])
  myBinsLog10S = np.linspace(2,6,100)
  myBinsC2     = np.linspace(0,3000,100)
  cmap = cm.magma#Greens#Blues

  plt.hist2d(log10SonehalfEnsemble,C2Ensemble,bins=[myBinsLog10S,myBinsC2],cmap=cmap)
  plt.plot(SMICAvals[0],SMICAvals[1],'cD')
  plt.colorbar()
  plt.xlabel(SonehalfLabel)
  plt.ylabel(C2Label)
  plt.show()

  
  print 'plotting C_2 vs. S_{1/2} contours... '

  H,xedges,yedges=np.histogram2d(log10SonehalfEnsemble,C2Ensemble,bins=(myBinsLog10S,myBinsC2))
  H = H.T  # Let each row list bins with common y range
  myXedges = (xedges[1:]+xedges[:-1])/2 #find midpoint of linspace for plotting
  myYedges = (yedges[1:]+yedges[:-1])/2
  hMax = np.max(H)
  #levels = [hMax*0.0009,hMax*0.009,hMax*0.09,hMax*0.9,hMax]
  #levels = [hMax*0.01,hMax*0.05,hMax*0.1,hMax*0.5,hMax*0.9,hMax]
  levels = np.logspace(np.log10(0.01*hMax),np.log10(0.9*hMax),5)

  norm = cm.colors.Normalize(vmax=abs(H).max(),vmin=0)
  #cmap = cm.PRGn

  #plt.figure()
  #plt.imshow(H,origin='lower',norm=norm,cmap=cmap)#,extent=extent) #extent is a coordinate zoom
  #plt.imshow(H,norm=norm,cmap=cmap,extent=(2,6,0,3000)) #should match linspace above
  #v = plt.axis()
  CS = plt.contour(myXedges,myYedges,H,levels,colors='k',thickness=2)
  plt.clabel(CS, inline=1, fontsize=10)
  #plt.axis(v)

  plt.colorbar()
  #plt.title('do i want a title here?')
  plt.xlim(2.8,5.8)
  plt.xlabel(SonehalfLabel)
  plt.ylabel(C2Label)
  plt.plot(SMICAvals[0],SMICAvals[1],'cD')
  plt.show()

  
  print 'plotting corner plot... '

  toPlot = np.vstack((log10SonehalfEnsemble,C2Ensemble))
  toPlot = toPlot.T
  figure = corner.corner(toPlot,labels=[SonehalfLabel,C2Label],
                         show_titles=False,truths=SMICAvals,
                         range=((2.5,6),(0,3000)) )
  plt.show()


  print 'plotting contours again but now using plot2Ddist (please wait)... '

  doTime = True
  startTime = time.time()
  scatterstyle = {'color':'r','alpha':0.5}
  styleargs = {'color':'k','scatterstyle':scatterstyle}
  bw_method = 0.05 #'scott'
  axSize= "20%" #1.5
  nstart=600

  # create separate figures to contain separate plots
  """plt.figure(1)
  ax1=plt.gca()
  plt.figure(2)
  ax2=plt.gca()
  plt.figure(3)
  ax3=plt.gca()"""
  #fig = plt.figure() #should be the same one used by plot2Ddist

  # divide C2Ensemble by 1000 since that is approximate factor between ranges of C2,Sonehalf
  # presumably useful for accuracy in contour plotting via kernel density estimation
  fig1,axeslist = plot2Ddist.plot2Ddist([log10SonehalfEnsemble,C2Ensemble/1000],
                         truevalues=[SMICAvals[0],SMICAvals[1]/1000],
                         labels=[SonehalfLabel,C2Label3],contourNGrid=nGrid,
                         bw_method=bw_method,axSize=axSize,nstart=nstart,
                         returnfigure=True,**styleargs)
                         #bw_method=bw_method,axSize=axSize,axeslist=[ax1,ax2,ax3],**styleargs)
  ax1,ax2,ax3=axeslist
  timeInterval1 = time.time()-startTime
  if doTime: 
    print 'time elapsed: ',int(timeInterval1),' seconds'
    print 'starting second plot2Ddist call... '

  ax1.set_xlim(left=2.9,right=6.1)
  ax1.set_ylim(top=5.5)
  ax1.plot(SMICAvals[0],SMICAvals[1]/1000,'cD')

  #inset plot
  left, bottom, width, height = [0.2, 0.4, 0.3, 0.3]
  ax4 = fig1.add_axes([left,bottom,width,height])
  #ax4.plot(range(10))
  plt.figure(5)
  ax5=plt.gca()
  plt.figure(6)
  ax6=plt.gca()

  plot2Ddist.plot2Ddist([log10SonehalfEnsemble,C2Ensemble/1000],
                         truevalues=[SMICAvals[0],SMICAvals[1]/1000],
                         contourNGrid=nGrid,
                         bw_method=bw_method,axSize=axSize,nstart=nstart,
                         axeslist=[ax4,ax5,ax6],contourFractions=[0.91,0.93,0.95,0.97,0.99],
                         labelcontours=False,**styleargs)

  timeInterval2 = time.time()-startTime
  if doTime: 
    print 'time elapsed for both: ',int(timeInterval2),' seconds'

  ax4.set_xlim(left=3.15,right=3.45)
  ax4.set_ylim(top=0.5)
  ax4.plot(SMICAvals[0],SMICAvals[1]/1000,'cD')
  ax4.xaxis.set_ticks((3.2,3.3,3.4))


  #plt.figure(1)
  #plt.xlim(2.9,6.1)
  #plt.ylim(-0.03,5.5)
  plt.show()


  # calculate and print 1D p-values
  pValueS12 = pValue(log10SonehalfEnsemble,SMICAvals[0])
  pValueC2  = pValue(C2Ensemble,SMICAvals[1])

  print 'S_{1/2} p-value = ',pValueS12
  print 'C_2 p-value     = ',pValueC2
  print ''


if __name__=='__main__':
  test()


