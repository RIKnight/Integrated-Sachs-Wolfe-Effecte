#! /usr/bin/env python
"""
Name:
  legprodint
Purpose:
  create the I_m,n factor that is the integral of a product of legendre polynomials:
    I_m,n(x) = \int_-1^x P_m(x')P_n(x')dx'
  Follows Appendix A (which has typos) of Copi et. al., 2009
  Then modified for use in S_{1/2} as in Copi et. al., 2013
Uses:

Inputs:

Outputs:

Modification History:
  Written by Z Knight, 2016.07.22
  Added S_{1/2} mod; ZK, 2016.07.24

"""
import numpy as np

def legendreSeries(x,lmax=101):
  """
  Purpose:
    creates set of legendre polynomials evaluated at position x
  Inputs:
    x: the point to evaluate at.  Should be -1 <= x <= 1.
      note: no error given if x is not in this range
    lmax: set this to the highest l value to return.
      note: lowest is l=0
      Default: lmax=101
  Returns:
    numpy array of numbers representing P_l(x), indices ranging from 0 to lmax

  """
  Pl = np.zeros(lmax+1)
  Pl[0] = 1
  Pl[1] = x
  for ell in range(2,lmax+1):
    Pl[ell] = ell**(-1) * ( (2*ell-1)*x*Pl[ell-1] - (ell-1)*Pl[ell-2] )
  
  return Pl


def getImn(makeNew=True,endX=0.5,lmax=100,fileName='legProdInt.npy'):
  """
  Purpose:
    Creates or loads Imn(x) array.
  Inputs:
    makeNew: set to True to recalculate array values
      if false, array will be loaded from file
      Default: True
    endX = the ending point of the integral
      Default: 0.5 (for use in S_{1/2})
    lmax = the highest ell value to include in the calculation
      Note: lmin=0, as this is used in recursion relations
      Default: 100
    fileName: string containin the file name to load/save array
      Default: legProdInt.npy

  """
  Imn = np.array([])
  if makeNew:
    # initialize
    Imn = np.zeros([lmax+2,lmax+1]) #or would empty be better?
    Imn[0,0] = endX+1
    Imn[1,1] = (endX**3+1)/3.
    # evaluate Legendre polynomials at x = endX
    Pl = legendreSeries(endX,lmax=lmax+1) # +1 for Imm recursion relation
    # fill in array
    for n in range(lmax+1):
      for m in range(lmax+2): # final index for final Imn[n+1,n-1]
        if m==0 and n==0 or m==1 and n==1:
          break
        if m==n: # eq.n A8
          Imn[m,n] = ( (Pl[n+1]-Pl[n-1])*(Pl[n]-Pl[n-2])
                      - (2*n-1)*Imn[n+1,n-1] + (2*n+1)*Imn[n,n-2]
                      + (2*n-1)*Imn[n-1,n-1] ) / (2*n+1)
        else:    # eq.n A6
          Imn[m,n] = ( m*Pl[n]*(Pl[m-1]-endX*Pl[m]) 
                      - n*Pl[m]*(Pl[n-1]-endX*Pl[n]) ) \
                      / (n*(n+1)-m*(m+1))
    np.save(fileName,Imn)
  else: #load from file
    Imn = np.load(fileName)

  return Imn[:-1,:]

def getJmn(makeNew=True,endX=0.5,lmax=100):
  """
  wrapper around getImn to do ell,ell' scaling
  """
  myImn = getImn(makeNew=makeNew,endX=endX,lmax=lmax)
  ellFactor = np.array([(2*ell+1) for ell in range(lmax+1)])/(4*np.pi)

  return myImn*np.outer(ellFactor,ellFactor)


################################################################################
# testing code

def test(useCLASS=1,useLensing=0,classCamb=1,nSims=1000,lmax=100):
  """
    code for testing the other functions in this module

  """
  # test getImn
  myImn = getImn(makeNew=True,lmax=100)
  print 'myImn: ',myImn



if __name__=='__main__':
  test()


