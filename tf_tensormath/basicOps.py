import tensorflow as tf
import numpy as np

def tensorDirectProd(t1,t2):
	tstT = tf.tensordot(t1,t2,axes=0)
	matdim = np.int32(np.sqrt(tf.size(tstT).numpy()))
	return tf.Variable([[tstT[q//2][p//2][q%2][p%2] for q in range(matdim)] for p in range(matdim)])

def matCommutator(m1,m2):
	return m1@m2-m2@m1
def matAntiCommutator(m1,m2):
	return m1@m2+m2@m1
def matSandwicher(u,o):
	return u@o@tf.linalg.adjoint(u)




