#! /usr/bin/env python
""" 
	functions to implement Gaussian Auto-Correlation Function
	as presented in
	ned.ipac.caltech.edu/level5/March02/White/White4.html
        Written by Z Knight, 2015.11.09
        Fixed integer division problem; ZK, 2015.11.13
"""
import numpy as np
import matplotlib.pyplot as plt

def C_GACF(C_0,theta,theta_C):
	""" the gacf """
	return C_0 * np.exp(-1* theta**2 / (2.*theta_C**2))

def C_ell(C_0,ell,theta_C):
	""" the legendre transform of the gacf """
	eFactor = np.exp(-1/2. *ell*(ell+1)*theta_C**2)
	#eFactor = np.exp(-1/2. *(ell+0.5)**2*theta_C**2) # a nearly identical variation
	return 2*np.pi*C_0*theta_C**2*eFactor


def test():
	print 'testing the gacf functions...'

	# test C_GACF
	C_0 = 1
	thetaMax = 10
	npoints = 100
	theta_C_set = np.linspace(0.1,5,5)

	thetaDomain = np.linspace(-1*thetaMax,thetaMax,npoints)
	for theta_C in theta_C_set:
		CG = C_GACF(C_0,thetaDomain,theta_C)
		plt.plot(thetaDomain,CG)
	plt.title('C_GACF with 5 different values of theta_C')
	plt.show()

	# test C_ell
	ellMax = npoints-1
	ellDomain = np.arange(ellMax)
	for theta_C in theta_C_set:
		Cl = C_ell(C_0,ellDomain,theta_C)
		plt.plot(ellDomain,Cl)
	plt.title('C_l with 5 different values of theta_C')
	plt.show()



if __name__=='__main__':
	test()
