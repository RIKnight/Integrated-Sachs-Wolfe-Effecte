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
  Added testing comparing against integration method; ZK, 2016.08.26
  Fixed serious misuse of "break" statement; ZK, 2016.08.26
  Added doSave to getImn, getJmn; ZK, 2016.09.12

"""
import numpy as np
from numpy.polynomial.legendre import legval # used for testing

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


def getImn(makeNew=True,endX=0.5,lmax=100,fileName='legProdInt.npy',doSave=True):
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
    doSave: set to True to save a file, if makeNew is also True.
      Default: True
  Returns:
    Imn(endX)

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
        #print m,n
        #if m==0 and n==0 or m==1 and n==1:
          #break
        if m==n: # eq.n A8
          if m!=0 and m!=1:
            Imn[m,n] = ( (Pl[n+1]-Pl[n-1])*(Pl[n]-Pl[n-2])
                        - (2*n-1)*Imn[n+1,n-1] + (2*n+1)*Imn[n,n-2]
                        + (2*n-1)*Imn[n-1,n-1] ) / (2*n+1)
        else:    # eq.n A6
            Imn[m,n] = ( m*Pl[n]*(Pl[m-1]-endX*Pl[m]) 
                        - n*Pl[m]*(Pl[n-1]-endX*Pl[n]) ) \
                        / (n*(n+1)-m*(m+1))
    if doSave:
      np.save(fileName,Imn)
  else: #load from file
    Imn = np.load(fileName)

  return Imn[:-1,:]

def getJmn(makeNew=True,endX=0.5,lmax=100,doSave=True):
  """
  wrapper around getImn to do ell,ell' scaling
  """
  myImn = getImn(makeNew=makeNew,endX=endX,lmax=lmax,doSave=doSave)
  ellFactor = np.array([(2*ell+1) for ell in range(lmax+1)])/(4*np.pi)

  return myImn*np.outer(ellFactor,ellFactor)


################################################################################
# testing code

def test(useCLASS=1,useLensing=0,classCamb=1,nSims=1000,lmax=100):
  """
    code for testing the other functions in this module

  """
  lmax = 5
  myX = 0.5
  myPl = legendreSeries(myX,lmax=lmax)
  print 'myPl: ',myPl

  newPl = np.zeros(lmax+1)
  for ell in range(lmax+1):
    c = np.zeros(lmax+1)
    c[ell] = 1
    newPl[ell] = legval(myX,c)
  print 'newPl: ',newPl


  lmax = 5
  # test getImn
  myImn = getImn(makeNew=True,lmax=lmax)
  print 'myImn: ',myImn

  # compare result against integration method
  newImn = np.zeros([lmax+1,lmax+1])
  nTerms = 10000
  dx = 1.5 / nTerms
  for m in range(lmax+1):
    for n in range(lmax+1):
      if m>n: #use symmetry
        newImn[m,n] = newImn[n,m]
      else:
        for term in range(nTerms):
          xVal = -1 + dx*(term+0.5) # evaluating at center of each bin
          c = np.zeros([2,lmax+1])
          c[0,m] = 1
          c[1,n] = 1
          newImn[m,n] += legval(xVal,c[0])*legval(xVal,c[1])*dx
  print 'newImn: ',newImn
  print 'newImn-myImn: ',newImn-myImn

if __name__=='__main__':
  test()


