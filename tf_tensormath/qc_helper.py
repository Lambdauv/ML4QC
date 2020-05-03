import tensorflow as tf
import numpy as np
from ML4QC.tf_tensormath import basicOps
def genAllPaulyCombine(Nqubits):
	d=2**Nqubits
	sigmax = tf.constant([[0,1],[1,0]],dtype=tf.complex128)
	sigmaz = tf.constant([[1,0],[0,-1]],dtype=tf.complex128)
	sigmay = sigmaz@sigmax/1j
	sigmas = [tf.eye(2,dtype=tf.complex128),sigmax,sigmay,sigmaz]
	outputlist = []
	#TODO: Try to make the following loop scale automatically with Nqubits.
	for p in sigmas:
		for q in sigmas:
			outputlist.append(basicOps.tensorDirectProd(p,q))
	return tf.stack(outputlist,0)

def jumpOp(c,rho):
	return (2*basicOps.matSandwicher(c,rho)-basicOps.matAntiCommutator(tf.linalg.adjoint(c)@c,rho))/2

def stepMasterEq(H,cs,rho,hbar=1,dt=1e-2):
	jumps = [jumpOp(c,rho) for c in cs]
	drho = (-1j/hbar*basicOps.matCommutator(H,rho)+sum(jumps))*dt
	return drho+rho

def aveFidelityTplus(Utarget,rhos0,HT,csT,rhosT): #the state input into the policy is the current benchmarking densities, action will be H and csT
	rhosTplusLs = [stepMasterEq(HT,csT,rhoT) for rhoT in rhosT]
	d=np.int(np.sqrt(len(rhosTplusLs)))
	recovMats = [tf.linalg.trace(basicOps.matSandwicher(Utarget,tf.linalg.adjoint(rhos0[k]))@rhosTplusLs[k]) for k in range(d)]
	subfide = sum(recovMats)
	return ((subfide+d**2)/(d**2*(d+1)),rhosTplusLs)