#! /usr/bin/env python
"""
  This program creates plots of mass overdensities for superclusters and supervoids,
    following the forumulation of Papai and Szapudi, 2010. (herafter PS)
  It starts with a matter power spectrum from CAMB, contained in file test_matterpower_2015.dat.
  From there it computes <delta(r)delta_in(R)> and <delta_in(R)^2>, as specified
    in the paper.  These will be numerically integrated and dependant on radius r.
  Then for a given value of delta_in(R), the conditional expectation <delta(r)|delta_in(R)>
    is calculated.
  This is then used to create the mass overdensity as a function of R and r.

  Note: formulae for ddin and dindin are based on the symmetrized Fourier convention.
  Power spectrum results from CAMB are "conventionally normalized", according to the
    CAMB documentation, but the authors do not specify which convention they are 
    referring to.  It may or may not be the same one I'm using.

  Written by Z Knight, April,May 2015
  Fixed missing factor of 1/c in 2nd term of dphi_deta; 
    Removed c**2 factor in phi; 
    Fixed etamin(zmax),etamax(zmin) problem in overtemp; ZK, 2015.09.02
  Fixed Mpc to Mpc/h conversion error in ClusterVoid.phi; ZK, 2015.10.07
  Fixed accidentally inverted 1/3 factor in ddin; ZK, 2015.10.08
  Added epsrel,epsabs to overmass2; wanted to mimic overmass accuracy
    but hardly changed it; but:sped up calculation tremendously; ZK, 2015.10.12

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sint
from scipy.interpolate import interp1d


################################################################################
# cosmological parameters
# these are being moved into the ClusterVoid class
#H_0        = 67.04   # km/sec/Mpc
#c_light    = 2.998e5 # km/sec
#km_per_Mpc = 3.086e19
#Omega_L    = 0.6817
#Omega_M    = 0.3183  # sum of DM and b
#Omega_DM   = 0.2678
#Omega_b    = 0.04902
#Omega_r    = 8.24e-5
##Omega_k explicitly assumed to be zero and omitted in formulae

################################################################################
# statistical quantites derived from power spectrum

# file name containing the power spectrum
#Pk_file = 'test_matterpower_2011.dat'
Pk_file = 'test_matterpower_2015.dat'

# load power spectrum
# message from CAMB params.ini:
#  Matter power spectrum output against k/h in units of h^{-3} Mpc^3
powspec = np.loadtxt(Pk_file)
#print 'Power spectrum data shape: ', powspec.shape

# create power spectrum interpolation function
powspecinterp = interp1d(powspec[:,0],powspec[:,1],kind='cubic')


# define a function for the partial window factor
def windowfac(k,R):
  """ This function implements the sin and cosine part of the Fourier transform of 
      the window function
    k is the wavenumber in the Fourier transform in h/Mpc
    R is the radius of the real space tophat window function in Mpc/h
  """
  kR = k*R
  return np.sin(kR)/(kR**3) - np.cos(kR)/(kR**2)

# the integrand for ddin
def int_ddin(k,r,R):
  """ the integrand for the <delta(r)delta_in(R)> integral
    k is the wavenumber in the Fourier transform in h/Mpc
    r is the radius at which to delta(r) is represented in Mpc/h
    R is the radius of the real space tophat window function in Mpc/h
  """
  return windowfac(k,R)*powspecinterp(k)*(k/r)*np.sin(k*r)

# the integrand for dindin
def int_dindin(k,R):
  """ the integrand for the <delta_in(R)**2> integral
    k is the wavenumber in the Fourier transform in h/Mpc
    R is the radius of the real space tophat window function in Mpc/h
  """
  return (windowfac(k,R))**2*powspecinterp(k)*k**2

# the integral ddin
def ddin(kmin,kmax,r,R):
  """ the <delta(r)delta_in(R)> integral
    kmin,kmax are the lower,upper limits of integration in h/Mpc
    r is the radius at which to delta(r) is represented in Mpc/h
    R is the radius of the real space tophat window function in Mpc/h
  """
  result = sint.quad(lambda k: int_ddin(k,r,R), kmin,kmax, limit=500)
  #normalized = result[0]*(4*np.pi)/(3*(2*np.pi))**(3/2.) #this was an error that was hard to find
  normalized = result[0]*(12*np.pi)/((2*np.pi)**(3/2.))
  #print 'R = ',R,'r = ',r,' ddin result: ',normalized
  return normalized

# the integral dindin
def dindin(kmin,kmax,R):
  """ the <delta_in(R)**2> integral
    kmin,kmax are the lower,upper limits of integration in h/Mpc
    R is the radius of the real space tophat window function in Mpc/h
  """
  result = sint.quad(lambda k: int_dindin(k,R), kmin,kmax)
  normalized = result[0]*9*(4*np.pi)/((2*np.pi)**(3/2.))
  #print 'dindin result: ',normalized
  return normalized

# the conditional expectation of delta(r) given delta_in(R)
def ce_delta(r,R,delta_in):
  """ the <delta(r) | delta_in(R)> conditional expectation function
    r is the radius at which to delta(r) is represented in Mpc/h
    R is the radius of the real space tophat window function in Mpc/h
    delta_in is delta_in(R), the given average of delta inside radius R.
      This value will be different for every supercluster or supervoid.
  """
  kmin = powspec[0,0]
  kmax = powspec[-1,0]
  my_ddin = ddin(kmin,kmax,r,R)
  my_dindin = dindin(kmin,kmax,R)
  #print my_ddin,my_dindin,delta_in
  return (my_ddin/my_dindin)*delta_in


################################################################################
# overmass functions

# overmass integrand
def int_overmass(kmin,kmax,r,R):
  """ the integrand for the overmass integral
    kmin,kmax are the lower,upper limits of integration in h/Mpc
    r is the radius at which to delta(r) is represented in Mpc/h
    R is the radius of the real space tophat window function in Mpc/h

    This function is only part of the integrand, omitting the parts that
      do not depend on r.
  """
  return r**2*ddin(kmin,kmax,r,R)

# overmass is integrated overdensity
def overmass2(R,delta_in,rmax=400,npoints=20):
  """ the volume integral of <delta(r) | delta_in(R)>
    R is the radius of the real space tophat window function in Mpc/h
    delta_in is delta_in(R), the given average of delta inside radius R.
      This value will be different for every supercluster or supervoid.
    rmax is the maximum radius to make calculaitons out to
      default value is rmax=400 Mpc/h
    npoints is the number of points in the summing process
      default value is npoints=20

    returns array of r values and array of M(r) values
  """
  rdomain = np.linspace(rmax/npoints**2,rmax,npoints) # rmax/npoints**2 to avoid 0.0
  M_vals = np.zeros(npoints)

  kmin = powspec[0,0]
  kmax = powspec[-1,0]

  # do integral for each point in rdomain, adding up domain intervals as it progresses
  for index in range(npoints):
    r0 = rdomain[index-1] # starting point for each piece of the integral
    r = rdomain[index]
    print 'index: ',index,', r0: ', r0,', r: ', r

    if index > 0:
      result = sint.quad(lambda rprime: int_overmass(kmin,kmax,rprime,R),r0,r,epsrel=0.02,epsabs=0.02)
      M_vals[index] = M_vals[index-1] + result[0] 
    else:
      M_vals[index] = 0

  # multiply by non-r dependant factors
  my_dindin = dindin(kmin,kmax,R)
  M_vals = (M_vals/my_dindin)*delta_in *4*np.pi

  return rdomain, M_vals

# overmass is integrated overdensity
def overmass(R,delta_in,rmax=400,npoints=50):
  """ Rewritten version of overmass 
    the volume integral of <delta(r) | delta_in(R)>
    R is the radius of the real space tophat window function in Mpc/h
    delta_in is delta_in(R), the given average of delta inside radius R.
      This value will be different for every supercluster or supervoid.
    rmax is the maximum radius to make calculaitons out to
      default value is rmax=400 Mpc/h
    npoints is the number of points in the summing process
      default value is npoints=50

    returns array of r values and array of M(r) values
      r will be in Mpc/h
      M(r) will be in (Mpc/h)**3
  """
  rmin = rmax*1.0/npoints**2 # avoid dividing by zero
  rdomain = np.linspace(rmin,rmax,npoints)
  delta_r_vals = np.zeros(npoints)
  M_vals = np.zeros(npoints)
  kmin = powspec[0,0]
  kmax = powspec[-1,0]

  # note: this does not use function ce_delta, since it evaluates dindin for each value of r

  # approximate the integral with a sum
  dr = (rmax*1.0-rmin)/npoints
  for index in range(npoints):
    r = rdomain[index]
    # calculate only the part that depends on r
    delta_r_vals[index] = ddin(kmin,kmax,r,R)
    #delta_r_vals[index] = ce_delta(r,R,delta_in)
    if index > 0:
      M_vals[index] = M_vals[index-1]+delta_r_vals[index]*r**2*dr
    else:
      M_vals[index] = 0
  #print 'delta_r_vals: ',delta_r_vals
  my_dindin = dindin(kmin,kmax,R)
  return rdomain, M_vals*4*np.pi /my_dindin*delta_in

def save_overmass(filename, R=60,rmax=4000,delta_in=0.5,npoints=100,doplot = False):
  """
    wrapper to run the overmass function with specified parameters and save results into text file

    filename is the name of the file to save the arrays to
    R is the radius of the real space tophat window function in Mpc/h
      default value is 60 Mpc/h
    delta_in is delta_in(R), the given average of delta inside radius R.
      This value will be different for every supercluster or supervoid.
    rmax is the maximum radius to make calculaitons out to
      default value is rmax=4000 Mpc/h
    npoints is the number of points in the summing process
      default value is npoints=100
    doplot is an optional flag for creating a M(r) plot
  
    returns array of r values and array of M(r) values; just as overmass does
    data saved in file will have two columns: r, M(r)
      r will be in Mpc/h
      M(r) will be in (Mpc/h)**3

    # 2015.04.29: testing indicates that in order to match the figure in PSG11, 
    #   I need to divide R and rmax by h (~0.65); Lloyd and I can't figure out why. 

  """
  #h = 0.65
  #R = R/h
  #rmax = rmax/h
  
  rdomain, my_overmass = overmass(R,delta_in,rmax,npoints)
  np.savetxt(filename,np.vstack((rdomain,my_overmass)).T)
  
  if doplot:
    plt.plot(rdomain,my_overmass)
    plt.title('Overmass profile, R = '+str(R)+', delta_in = '+str(delta_in))
    plt.xlabel('r')
    plt.ylabel('M(r)')
    plt.show()

  return rdomain,my_overmass


################################################################################
# the cluster/void class

class ClusterVoid:
  """ 
    class to create cluster or void objects.
    each cluster/void is based on an (r, M(r)) table file, like those created by
      function save_overmass
    each object also is placed at a certain redshift

    note: the h = H_0/100 parameter is used in methods and variables related to
      the center of the cluster/void, and not for methods and variables related
      to the line of sight (all should be labeled in comments)
  """
  # cosmological parameters
  H_0        = 67.04   # km/sec/Mpc
  c_light    = 2.998e5 # km/sec
  km_per_Mpc = 3.086e19
  Omega_L    = 0.6817
  Omega_M    = 0.3183  # sum of DM and b
  Omega_DM   = 0.2678
  Omega_b    = 0.04902
  Omega_r    = 8.24e-5
  # Omega_k explicitly assumed to be zero in formulae and omitted

  def __init__(self,massfile,zcenter):
    """
      massfile is the name of an (r,M(r)) overmass file
      data saved in file must have two columns: r, M(r)
        r should be in Mpc/h
        M(r) should be in (Mpc/h)**3
      zcenter is the redshift of the center of the cluster or void
    """
    self.zcenter = zcenter
    self.eta_cent = self.conftime(1/(1.0+zcenter)) #conftime takes scale factor argument
    self.rcent,self.Mcent = np.loadtxt(massfile,unpack=True)

    # add point in outer region for kludgy extrapolation
    self.rcent = np.concatenate([self.rcent,[2*self.rcent[-1]]])
    self.Mcent = np.concatenate([self.Mcent,[0]]) # ramps down to zero at r=2*rcent[-1]

    # create potential from overmass
    self.rdomain,self.newtpot = self.potential()

    # create scale factor / conformal time reverse lookup table 
    #self.adomain = np.logspace(-10,0,100)
    self.adomain = np.logspace(-1,0,100)
    self.etarange = self.conftime(self.adomain)


  # maybe this method should be moved into the init method?  Is it useful otherwise?
  def potential(self,npoints=101):
    """
      Method to calculate the newtonian potential at radius r.
      Follows PS eq.n 29
      npoints determines the number of points in the map
        default is 101

      returns a pair of vectors each npoints long: r, phi(r)
        r will be in Mpc/h
        phi(r) will be in units (km/s)**2
    """

    # write in terms of h=H_0/100 and not H_0 so that h factors cancel in return value
    #prefac = -3*self.Omega_M/(8*np.pi) *self.H_0**2  # units: (km/s)**2*Mpc**-2
    prefac = -3*self.Omega_M/(8*np.pi) *100**2  # units: (km/s)**2*(h/Mpc)**2

    # create overmass interpolation function
    Minterp = interp1d(self.rcent,self.Mcent)#,kind='cubic')

    # r, phi(r) arrays
    rdomain = np.linspace(self.rcent[0],self.rcent[-1],npoints)
    newtpot = np.zeros(npoints)

    for rindex in range(rdomain.size):
      result = sint.quad(lambda r: Minterp(r)/r**2, rdomain[rindex],rdomain[-1],limit=500)
      newtpot[rindex] = result[0]

    return rdomain,newtpot*prefac


  def hubbleparam(self,a):
    """ the hubble parameter as a function of scale factor and cosmological 
        parameters.  Curvature is assumed to be flat and is omitted from
        the calculation.
        returns hubble parameter in units km/s/Mpc
    """
    omega_scaled = self.Omega_M*a**-3 + self.Omega_r*a**-4 + self.Omega_L
    return self.H_0 * np.sqrt(omega_scaled)

  def conftime(self,a):
    """ the conformal time / horizon distance for a given scale factor a 
        a can be a single scale factor or an array of them
        returns conftime in units Mpc/c (where c=1 so units are Mpc)
        I call this a conformal time but really it's a comoving distance in Mpc
          from a'=0 to a'=a.
        To convert to Mpc/h/c, multiply by H_0/100
    """
    a = np.array(a)
    eta = np.zeros(a.size)
    int_eta = lambda a: a**-2*self.hubbleparam(a)**-1
    for aindex in range(a.size):
      if a.size == 1:
        result = sint.quad(int_eta,0,a)
      else:
        result = sint.quad(int_eta,0,a[aindex])
      eta[aindex] = result[0]
    return eta*self.c_light

  def int_D1(self,a):
    """ the integrand for the growth factor equation """
    H = self.hubbleparam(a)
    return (a*H/self.H_0)**-3

  def D1(self,a):
    """ growth factor; Dodelson eq.n 7.77 
        a can be a single scale factor or an array of them
    """
    a = np.array(a)
    prefac = 5*self.Omega_M*self.hubbleparam(a)/(2*self.H_0)
    growth = np.zeros(a.size)
    for index in range(a.size):
      if a.size == 1:
        result = sint.quad(self.int_D1, 0, a)
      else:
        result = sint.quad(self.int_D1, 0, a[index])
      growth[index] = result[0]
    return prefac*growth

  def dD1_deta(self,a):
    """ derivative of the growth factor wrt conformal time eta 
        returns dD1/deta in units c/Mpc
    """
    H = self.hubbleparam(a)
    return ( -a*H*self.D1(a) + 5*self.Omega_M*self.H_0**2/(2*a*H) )/self.c_light

  def phi(self,eta,impact):
    """ 
      the potential of a supervoid or supercluster centered at conformal time self.eta_cent
      eta is the conformal time / horizon distance at which to calculate phi
        eta must be in units Mpc/c (where c=1... so units are Mpc)
      impact is the impact parameter of the line of sight to the center of the cluster/void
        impact must be in units Mpc
      returns the value of potential at eta,impact in units (km/s)**2
    """
    # self.eta_cent should be in Mpc/c (where c=1... so units are Mpc)
    # self.rdomain should be in Mpc/h
    # self.newtpot should be in units (km/s)**2

    # create potential interpolation function
    phi_interp = interp1d(self.rdomain,self.newtpot)#,kind='cubic')

    #r = np.sqrt(impact**2 + self.c_light**2*(eta-self.eta_cent)**2 )
    r = np.sqrt(impact**2 + (eta-self.eta_cent)**2 )
    #rph = r*100/self.H_0 # unfortunately I went along time before noticing this error.
    rph = r*self.H_0/100 # convert from Mpc to Mpc/h
    phi = np.zeros(rph.size)
    #print 'eta: ',eta
    #print 'rph: ',rph,', self.rdomain[-1]: ',self.rdomain[-1]
    for index in range(rph.size):
      if rph[index] < self.rdomain[-1]:
        phi[index] = phi_interp(rph[index])  ### 2016.01.31 getting error here about "below the interpolation range"
                                             ### occurred with z_cent=0.6, delta_z=0.3, overmass_R010, point 1 of 101
                                             ### worked fine for z_cent=0.4, 0.45, 0.5, 0.55
    return phi

  def dphi_deta(self,eta,impact):
    """
      the deriviative of potential with respect to conformal time, eta
      This is the derivative of phi(a) = phi(a_0)*(a_0*D1(a))/(a*D1(a_0)),
        where a_0 = 1
        derived from Dodelson eq.n 7.4
      eta is the conformal time / horizon distance at which to calculate phi
        eta must be in units Mpc/c (where c=1 so units are Mpc)
      impact is the impact parameter of the line of sight to the center of the cluster/void
        impact must be in units Mpc
      range is limited by 0.1 < a < 1 (9 > z > 0)
      returns the value of dphi/deta at coordinates eta,impact in units (km/s)**2/Mpc
    """
    # interpolate a from a/eta reverse lookup table (limited range)
    scale = interp1d(self.etarange,self.adomain,kind='cubic')
    a = scale(eta)

    H = self.hubbleparam(a)
    prefac = self.phi(eta,impact)/self.D1(1)
    return prefac*( self.dD1_deta(a)/a - H*self.D1(a)/self.c_light )

  def overtemp(self,impact,delta_z=0.2):
    """
      function to calculate the quantity DeltaT/T; the ISW signal.
      impact is the impact parameter of the line of sight to the center of the cluster/void
        impact must be in units Mpc
      delta_z is the redshift radius around the object's central redshift for
        which to run the integral
        default is delta_z = 0.2
      returns DeltaT/T (no units)
    """
    zmin = self.zcenter-delta_z
    zmax = self.zcenter+delta_z
    etamin = self.conftime(1/(1.0+zmax)) #eta increases in opposite direction as z
    etamax = self.conftime(1/(1.0+zmin))
    prefac = -2/self.c_light**2
    result = sint.quad(lambda etaprime: self.dphi_deta(etaprime,impact),etamin,etamax)
    return prefac*result[0]


################################################################################
# testing code

def test():
  """ function for testing the other functions in this file """

  """
  # display power spectrum plot
  plt.loglog(powspec[:,0],powspec[:,1])
  plt.title('CAMB matter power spectrum')
  plt.xlabel('Wavenumber k [h/Mpc]')
  plt.ylabel('Power Spectrum P(k) [(Mpc/h)^3]')
  plt.show()
  

  # test the windowfac function
  k = 1
  R = np.arange(1500)/100.
  window = windowfac(k,R)
  plt.semilogy(k*R,window**2)
  plt.title('( the transformed window function / 4*pi*R^2 )^2')
  plt.show()

  # test int_ddin
  kmin = powspec[0,0]
  kmax = 0.5 #powspec[-1,0]
  x1 = np.linspace(kmin,kmax,10)
  y1 = int_ddin(x1,150,60)
  x2 = np.linspace(kmin,kmax,100)
  y2 = int_ddin(x2,150,60)
  x3 = np.linspace(kmin,kmax,1000)
  y3 = int_ddin(x3,150,60)

  plt.plot(x1,y1)
  plt.plot(x2,y2)
  plt.plot(x3,y3)
  plt.title('int_ddin test; R=60, r=150')
  plt.xlabel('k')
  plt.ylabel('int_ddin')
  plt.show()

  # test int_dindin
  kmin = powspec[0,0]
  kmax = 0.5 #powspec[-1,0]
  x1 = np.linspace(kmin,kmax,10)
  y1 = int_dindin(x1,60)
  x2 = np.linspace(kmin,kmax,100)
  y2 = int_dindin(x2,60)
  x3 = np.linspace(kmin,kmax,1000)
  y3 = int_dindin(x3,60)

  plt.plot(x1,y1)
  plt.plot(x2,y2)
  plt.plot(x3,y3)
  plt.title('int_dindin test; R=60, (r not used)')
  plt.xlabel('k')
  plt.ylabel('int_dindin')
  plt.show()
  
  # test ddin
  kmin = powspec[0,0]
  kmax = powspec[-1,0]
  R = 60 
  npoints = 100
  #rdomain = np.linspace(0.1,800,npoints)
  rdomain = np.linspace(600,1000,npoints)
  #print 'rdomain: ',rdomain
  ddin_vals = np.zeros(npoints)
  #print 'ddin_vals: ',ddin_vals
  for index in range(npoints):
    ddin_vals[index] = ddin(kmin,kmax,rdomain[index],R) *rdomain[index]**2 # r**2 for visibility

  plt.plot(rdomain,ddin_vals)
  plt.title('ddin test; R= 60')
  plt.xlabel('r')
  plt.ylabel('<delta(r)delta_in(R)>')
  plt.show()

  # test dindin
  kmin = powspec[0,0]
  kmax = powspec[-1,0]
  R = 60 
  dindin_val = dindin(kmin,kmax,R)
  print 'dindin(R=60) = ',dindin_val

  # test ce_delta
  r = 20
  R = 60
  delta_in = 0.5 # starting guess for a supercluster

  my_ce_delta = ce_delta(r,R,delta_in)
  print 'for R = ',R,', r = ',r,':'
  print 'conditional expectation of delta: ',my_ce_delta
  
  # test overmass
  # 2015.04.29: testing indicates that in order to match the figure in PSG11, 
  #   I need to divide R and rmax by h (~0.65); Lloyd and I can't figure out why. 

  #h = 0.65
  #R = 60/h
  #rmax = 800/h
  
  R = 60 # without the h
  rmax = 400
  delta_in = 0.5
  npoints = 100
  rdomain, my_overmass = overmass(R,delta_in,rmax,npoints)
  #rdomain, my_overmass = overmass2(R,delta_in)

  plt.plot(rdomain,my_overmass)
  plt.title('Overmass profile, R = '+str(R)+' Mpc/h, delta_in = '+str(delta_in))
  plt.xlabel('r [Mpc/h]')
  plt.ylabel('M(r) [(Mpc/h)**3]')
  plt.show()
  


  """
  # test ClusterVoid class
  # need to create file using save_overmass before doing this
  #file400  = 'R60_rmax400_din0p5_npoints100.txt'
  file4000 = 'R60_rmax4000_din0p5_npoints100.txt'
  file800  = 'overmass_R60_rmax800overh_din1.txt'
  file800h = 'overmass_R60overh_rmax800overh_din1.txt'
  #file400  = 'overmass_R14.6_d0.46.txt'
  #zcent = 0.458
  file400  = '/shared/Data/PSG/overmass_din1_R060.txt'
  zcent = 0.52
  myCluster = ClusterVoid(file400,zcent)

  """
  # test potential
  #r,phi = myCluster.potential(npoints=101)
  r = myCluster.rdomain; phi = myCluster.newtpot
  plt.plot(r,phi*1e6)
  plt.xlabel('radius from center [Mpc/h]')
  plt.ylabel('potential [1e-6]')
  plt.title('Newtonian potential via PS eq.n 29')
  plt.show()
  
  
  # test hubbleparam
  adomain = np.logspace(-7,0,100)
  Hrange = myCluster.hubbleparam(adomain)
  plt.loglog(adomain,Hrange)
  plt.xlabel('scale factor')
  plt.ylabel('Hubble parameter [km s**-1 Mpc**-1]')
  plt.show()
  
  # test conftime
  adomain = np.logspace(-7,0,100)
  etarange = myCluster.conftime(adomain)
  plt.loglog(adomain,etarange)
  plt.xlabel('scale factor')
  plt.ylabel('conformal time [Mpc]')
  plt.show()
  
  # test D1 (growth factor)
  adomain = np.logspace(-7,0,100)
  D1range = myCluster.D1(adomain)
  plt.loglog(adomain,D1range)
  plt.xlabel('scale factor')
  plt.ylabel('growth factor')
  plt.show()

  # test dD1_deta (derivative of growth factor)
  adomain = np.logspace(-7,0,100)
  dD1range = myCluster.dD1_deta(adomain)
  plt.loglog(adomain,dD1range)
  plt.xlabel('scale factor')
  plt.ylabel('d(growth factor) / d(conformal time) [Mpc**-1]')
  plt.show()
  
  
  # test phi
  delta_z = 0.14 #0.8e-6
  scalefac = 1e-5
  zdomain = np.linspace(zcent-delta_z,zcent+delta_z,101)
  zplottable = (zdomain-zcent) # *scalefac
  adomain = 1/(1.0+zdomain)
  etadomain = myCluster.conftime(adomain) #Mpc
  etacent = myCluster.conftime(1/(1+zcent))
  etaplottable = (etadomain-etacent) #Mpc


  #plt.plot(zdomain,etadomain)
  #plt.xlabel('zdomain')
  #plt.ylabel('etadomain')
  #plt.show()

  
  phirange1 = myCluster.phi(etadomain,10)
  phirange2 = myCluster.phi(etadomain,50)
  phirange3 = myCluster.phi(etadomain,150)

  
  #print 'phirange1: ',phirange1
  #print 'phi assymetry: ',phirange1-phirange1[::-1]

  yplotmin = -8
  yplotmax = 0
  fig, ax1 = plt.subplots()
  #ax1.plot(zplottable,phirange1*scalefac)
  ax1.plot(zplottable,phirange2*scalefac)
  ax1.plot(zplottable,phirange3*scalefac)
  ax1.set_xlabel('redshift-'+str(zcent) )
  ax1.set_ylabel('gravitational potential ['+"%e"%(scalefac**-1)+' (km/s)**2]')
  ax1.axis([zplottable[0],zplottable[-1],yplotmin,yplotmax])

  ax2=ax1.twiny()
  ax2.plot(etaplottable,phirange1*scalefac,'r')
  ax2.set_xlabel('conformal time - '+"%d"%etacent+'[Mpc]')
  ax2.axis([etaplottable[0],etaplottable[-1],yplotmin,yplotmax])
  #ax2.set_title('phi(r) for b = 10, 50, 150 Mpc')
  plt.show()

  
  # test dphi_deta
  # uses same test data as phi, above
  dphirange1 = myCluster.dphi_deta(etadomain,10)
  dphirange2 = myCluster.dphi_deta(etadomain,50)
  dphirange3 = myCluster.dphi_deta(etadomain,150)
  #scalefac = 1#1e4
  etaplottable = (etadomain-etacent)

  #print 'dphi_deta assymetry: ',dphirange1-dphirange1[::-1]

  yplotmin = 0
  yplotmax = 75
  fig, ax1 = plt.subplots()
  #ax1.plot(zplottable,dphirange1)
  ax1.plot(zplottable,dphirange2)
  ax1.plot(zplottable,dphirange3)
  ax1.set_xlabel('redshift-'+str(zcent) )
  ax1.set_ylabel('dphi/deta ['+str(scalefac**-1)+' (km/s)**2 Mpc**-1]')
  ax1.axis([zplottable[0],zplottable[-1],yplotmin,yplotmax])

  ax2=ax1.twiny()
  ax2.plot(etaplottable,dphirange1,'r')
  ax2.set_xlabel('conformal time-'+"%d"%etacent+' Mpc]')
  ax2.axis([etaplottable[0],etaplottable[-1],yplotmin,yplotmax])
  #ax2.set_title('dphi(r)/deta for b = 10, 50, 150 Mpc')
  plt.show()
  """
  
  # test overtemp
  delta_z = 0.14
  # overtemp(impact,delta_z) where impact has units Mpc
  #overtemp1 = myCluster.overtemp(10,delta_z)
  #overtemp2 = myCluster.overtemp(50,delta_z)
  #overtemp3 = myCluster.overtemp(150,delta_z)
  #print 'overtemp(10): ',overtemp1,', overtemp(50): ',overtemp2,', overtemp(150): ',overtemp3

  npoints = 10
  impactDomain = np.linspace(10,800,npoints) #Mpc
  ISWRange = np.zeros(npoints)
  for pointnum in range(npoints):
    print 'calculating point number ',pointnum+1,' of ',npoints
    ISWRange[pointnum] = myCluster.overtemp(impactDomain[pointnum],delta_z)

  plt.plot(impactDomain,ISWRange)
  plt.xlabel('r [Mpc]')
  plt.ylabel('ISW: DeltaT / T')
  plt.show()
  

if __name__=='__main__':
  test()
