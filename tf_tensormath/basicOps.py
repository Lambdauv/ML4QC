# coding=utf-8
# Copyright 2020 Jie Luo.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3

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




