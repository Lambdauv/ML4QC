from ML4QC.tf_tensormath import qc_helper
from ML4QC.tf_tensormath import basicOps
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
class PulseResponseEnv(py_environment.PyEnvironment):
    def __init__(self,Utarget,rho0s):
        rhoshape = rho0s.shape
        self.Nqubits = np.int(np.log(rhoshape[0])/np.log(4))
        self._action_spec = array_spec.BoundedArraySpec(shape=(3*self.Nqubits+1,),dtype=np.float32,minimum = -np.ones((3*self.Nqubits+1,),dtype=np.float32), maximum = np.ones((3*self.Nqubits+1,),dtype=np.float32), name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(2,rhoshape[0],2**self.Nqubits,2**self.Nqubits),dtype=np.float32,minimum = -np.ones((2,rhoshape[0],2**self.Nqubits,2**self.Nqubits),dtype=np.float32), maximum = np.ones((2,rhoshape[0],2**self.Nqubits,2**self.Nqubits),dtype=np.float32),name='rhoTs')#{"runtime":array_spec.BoundedArraySpec(shape=(1,),dtype=np.int32,minimum = np.int32(0), maximum = np.int32(256)),"rhoTs":array_spec.BoundedArraySpec(shape=(8,2**8,2**8),dtype=np.float32,minimum = np.zeros((8,2**8,2**8),dtype=np.float32), maximum = np.ones((8,2**8,2**8),dtype=np.float32))}
        self._state = tf.stack([np.float32(tf.math.real(rho0s)),np.float32(tf.math.imag(rho0s))], 0)#{"runtime":np.zeros(1,dtype=np.int32),"rhoTs":rho0s}
        self._episode_ended = False
        self.Utarget = Utarget
        self.rho0s = rho0s
        self._Tstep = 0
        self._reward = 0
    def action_spec(self):
        return self._action_spec
    def observation_spec(self):
        return self._observation_spec
    def _reset(self):
        self._state = tf.stack([np.float32(tf.math.real(self.rho0s)),np.float32(tf.math.imag(self.rho0s))], 0)
        self._episode_ended = False
        self._Tstep = 0
        self._reward = 0
        return ts.restart(self._state,batch_size=self.batch_size)
    def _step(self,action):
        if self._episode_ended:
            return self._reset()
        self.move(action)
        if self.game_over():
            self._episode_ended = True
            return ts.termination(self._state,reward = np.float32(np.absolute(self._reward)))
        else:
            return ts.transition(self._state,reward = 0.0)
    def move(self,action):
        self._Tstep += 1
        rho0Ls = [self.rho0s[k,:,:] for k in range(self.Nqubits**4)]
        rhoTLs = [np.complex128(self._state[0,k,:,:])+1j*np.complex128(self._state[1,k,:,:]) for k in range(self.Nqubits**4)]
        HT,csT = self.genHandC(action)
        Ufidelity,rhoTplusLs = qc_helper.aveFidelityTplus(self.Utarget,rho0Ls,HT,csT,rhoTLs)
        rhoTplusMat = np.array(rhoTplusLs)
        self._state = tf.stack([np.float32(tf.math.real(rhoTplusMat)),np.float32(tf.math.imag(rhoTplusMat))], 0)
        self._reward = Ufidelity
    def game_over(self):
        return (self._Tstep == 1024)
    def genHandC(self,action):
        actionVec = action
        sigmax = tf.constant([[0,1],[1,0]],dtype=np.complex128)
        sigmaz = tf.constant([[1,0],[0,-1]],dtype=np.complex128)
        sigmay = sigmaz@sigmax/1j
        sigxsigx=basicOps.tensorDirectProd(sigmax,sigmax)
        sigysigy=basicOps.tensorDirectProd(sigmay,sigmay)
        sigzs = [basicOps.tensorDirectProd(sigmaz,tf.eye(2,dtype=np.complex128)),basicOps.tensorDirectProd(tf.eye(2,dtype=np.complex128),sigmaz)]
        sigxs = [basicOps.tensorDirectProd(sigmax,tf.eye(2,dtype=np.complex128)),basicOps.tensorDirectProd(tf.eye(2,dtype=np.complex128),sigmax)]
        sigys = [basicOps.tensorDirectProd(sigmay,tf.eye(2,dtype=np.complex128)),basicOps.tensorDirectProd(tf.eye(2,dtype=np.complex128),sigmay)]
        Ht = 0.5*actionVec[0]*(sigxsigx+sigysigy)+sum([0.5*actionVec[k+1]*sigzs[k]-actionVec[k+3]*(sigxs[k]+sigys[k]) for k in range(2)])
        cst = [actionVec[k+5]*(sigys[k]*1j+sigxs[k])/2 for k in range(2)]
        return (Ht, cst)