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

import numpy as np
import torch
def coefMatGen(qfreqs,gqq,dramps,drFreq):
    qfreqs=qfreqs-drFreq
    diaglin = np.insert(qfreqs,0,0)
    coefMat = np.diagflat(diaglin, 0)
    coefMatSz = coefMat.shape
    gqqSz = gqq.shape
    coefMat[1:,1:]+=np.triu(gqq,1)
    coefMat[1:,1:]+=np.transpose(np.triu(gqq,1))
    coefMat[0,1:]+=dramps
    coefMat[1:,0]+=dramps
    return coefMat    
def singleFreqSol(psi0,qfreqs,gqq,dramps,drFreq,t):
    coefM=coefMatGen(qfreqs,gqq,dramps,drFreq)
    eigVals, eigVecs = np.linalg.eig(coefM)
    psi0EigWeights = psi0.dot(eigVecs)
    matformT = t.reshape((1,-1))
    matformEigVal = eigVals.reshape((-1,1))
    timeEvalFact = np.exp(-2*np.pi*matformEigVal.dot(matformT)*1j)
    timeWeight = np.diagflat(psi0EigWeights).dot(timeEvalFact)
    timePsi = eigVecs.dot(timeWeight)
    return timePsi
def multiFreqSol(psi0,qfreqs,gqq,dramps,drFreqs,t):
    return np.transpose(np.array([singleFreqSol(psi0,qfreqs,gqq,dramps,drF,t) for drF in drFreqs]) , (1, 0, 2))
def torchInput3DTensor(multifreqTimePsi):
    return torch.from_numpy(np.array(np.square(np.absolute(multifreqTimePsi[1:,:,:]))))
