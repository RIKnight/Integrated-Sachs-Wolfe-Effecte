# Integrated-Sachs-Wolfe-Effecte
A combined repository containing all of my S_{1/2} Anomaly and Stacking Anomaly code which is related to the ISW effect.

Readme updated February 12, 2018, by Z Knight

This repository containss mostly Python code, as well as some helper programs and data files.

Other software used:
  numpy
  scipy
  CAMB
  HEALPix
  PolSPICE
  ramdisk.sh
  optimzeSx.so

Python files:

ISW_template.py:
  ISW template objects will have the parameters taken from the GNS catalogs.
    These are meant to be used in a nonlinear template_fit minimization
    ISW template objects will have a method to produce a template based on
        the two parameters of the model
        Pixel values will be DeltaT/T * 2.726K
        
ISWprofile.py:
  This program creates plots of mass overdensities for superclusters and supervoids,
    following the forumulation of Papai and Szapudi, 2010. (herafter PS)
  It starts with a matter power spectrum from CAMB, contained in file test_matterpower_2015.dat.
  From there it computes <delta(r)delta_in(R)> and <delta_in(R)^2>, as specified
    in the paper.  These will be numerically integrated and dependant on radius r.
  Then for a given value of delta_in(R), the conditional expectation <delta(r)|delta_in(R)>
    is calculated.
  This is then used to create the mass overdensity as a function of R and r.
  
KS_showdown.py:
  Evaluates KS statistic on an ensemble of CMB realizations
    Compares different methods of template fitting for ISW map
    
SN_mode_filter.py:
  Uses covariance matrices for signal (s) and noise (n) with T=s+n model
        and a mask for selecting a data vector from a healpix map to find
        a rotation such that covariance matrices <s_i*S_j>, <n_i*n_j>
        are diagonal
    Creates an ensemble of simulations from power spectra for s and n,
        extracts data vector from these, rotates into diagonal frame,
        squares and averages these to find (S+N)/N eigenvalues.
        
chisquare_test.py:
  Program to check that T*C^-1*T follows chi square distribution as expected
  
cosmography.py:
  calculate various distance measures in an expanding universe
  
gacf.py:
  functions to implement Gaussian Auto-Correlation Function
	as presented in
	ned.ipac.caltech.edu/level5/March02/White/White4.html
  
get_crosspower.py:
  load total, primordial, and late Cl power spectra and create cross power:
    C_l^tot = C_l^(prim,prim) + 2*C_l^(prim,late) + C_l^(late,late)
    
get_temps.py:
  Loads map and mask from fits files and coordinates from text files,
            measures temperatures of cluster and void stacks,
            and determines S/N ratio using random coordinates

gittesting.py:
  This file is just for testing git.
  
inversion_testing.py:
  Program to explore covariance matrix inversion
  
ispice.py:
  This program is not mine, but is included here because I use it.
  ispice defines tools to run Spice from Python,
  either in the Planck HFI DMC (aka piolib, objects managed by database)
  or using FITS files
  
legendre_test.py:
  to test accuracy of legendre transforms
  
legprodint.py:
  create the I_m,n factor that is the integral of a product of legendre polynomials:
    I_m,n(x) = \int_-1^x P_m(x')P_n(x')dx'
  Follows Appendix A (which has typos) of Copi et. al., 2009
  Then modified for use in S_{1/2} as in Copi et. al., 2013
  
likelihood.py:
  Evaluate the CMB likelihood functions
  
make_Cmatrix.py:
  Program to create a covariance matrix for a given set of HEALpix pixels.
  
make_ISW_map.py:
  create a HEALpix spherical map containing an ISW map from one or more
      superclusters or supervoids
    Pixel values will be DeltaT/T * 2.726K
    
make_overmass.py:
  This program creates sets of ISW profiles following the formulation of
      Papai and Szapudi, 2010. (PS)

make_sims.py:
  to load simulated ISW maps and simulated CMB maps and add them
  
mask_check.py:
  Program to calculate amplitude of template on CMB using various masks
  
optSxJackknife.py:
  do Jackknife testing to check for stability of optimized x, S_x, P(x)
  
optimizeSx1.py:
  explore the presumed arbitrary cut off point for S_{1/2} by optimizing
    PTE(S_x) for random CMB realizations
    
optimizeSx2.py:
  explore the presumed arbitrary cut off point for S_{1/2} by optimizing
    PTE(S_x) for random CMB realizations
     
plot2Ddist.py:
  The plot2Ddist function plots the joint distribution of 2 variables, 
    with estimated density contours and marginal histograms.
  Designed to plot parameter distributions for MCMC samples.
  
quadoctcorr.py:
  extract l=2,3 a_lm.s from SMICA to compare C(theta) against simulations
  
redshift_check.py:
  opens ISW profile files created for various redshifts and compares them.
        Looks for ratios and... ??

scaledlegs.py:
  show legendre polynomials * (2l+1) together
  
sim_stats.py:
  create simulated early and late CMB maps and analyze them for CMB anomalies
  
simple_Sonehalf.py:
  create simplistic CMB simulations and analyze S_{1/2} properties
  shows trend when increasing l_min
  Also plots C_2 and S_{1/2} together on 2d plot
  
surface3d_profiles.py:
  plotting ISW profiles as surfaces; functions of R and r
  
template.py:
  Just a template I used for quickly starting python files
  
template_fit.py:
  Program to calculate amplitude of template on CMB
  
template_fit_SN.py:
  Program to calculate amplitude of template on CMB using non-direct methods
  
template_fit_eig.py:
  Program to calculate amplitude of template on CMB using non-direct methods
  
  
  
Non-python program files:

optimizeSx.c:
  * create S_x functions from input file
  * create Pval(x) for each S_x using ensemble
  * find global minimum for each Pval(x)
  *   -> can be used to create distribution S(xValMinima)
  
ramdisk.sh:
  # From http://tech.serbinn.net/2010/shell-script-to-create-ramdisk-on-mac-os-x/
  
spice:
  the PolSPICE program
  
  
  
