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
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse
from torch.utils.tensorboard import SummaryWriter
import ML4QC.torch_tensormath.torch_helper as torch_helper
import ML4QC.torch_tensormath.qc_helper as qc_helper
class crossTalkConvNet(nn.Module):
    def __init__(self,inputDims):
        super(crossTalkConvNet, self).__init__()
        self.writer = SummaryWriter()
        self.numConvStack = 1
        self.inputDims = inputDims
        self.Ntimes = self.inputDims[1]
        self.Nfreqs = self.inputDims[0]
        self.kernelDim1 = (16, 32)
        self.kernelDim2 = (7, 13)
        self.kernelDim3 = (4, 8)
        self.kernelDim4 = (2, 4)
        self.padding = (0, 0)
        self.stride1 = (4, 4)
        self.stride2 = (2, 2)
        self.stride3 = (2, 2)
        self.stride4 = (1, 1)
        self.conv1OutSz = torch_helper.conv2dOutSize(self.kernelDim1,self.padding,self.stride1,(1,1),self.inputDims)
        self.conv2OutSz = torch_helper.conv2dOutSize(self.kernelDim2,self.padding,self.stride2,(1,2),self.conv1OutSz)
        self.conv3OutSz = torch_helper.conv2dOutSize(self.kernelDim3,self.padding,self.stride3,(1,4),self.conv2OutSz)
        self.conv1 = nn.Conv2d(8, 16, self.kernelDim1,stride=self.stride1,padding=self.padding,dilation=(1,1)) 
        self.conv2 = nn.Conv2d(16, 32, self.kernelDim2,stride=self.stride2,padding=self.padding,dilation=(1,2)) 
        self.conv3 = nn.Conv2d(32, 64, self.kernelDim3,stride=self.stride3,padding=self.padding,dilation=(1,4)) 
        self.fc1 = nn.Linear(64*int(self.conv3OutSz[0]*self.conv3OutSz[1]),8*2+28*0+8)
    
    def forward(self,inputImg):
        x = inputImg
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x 

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def targetFormater(self,drFreqList,t):
        Psi0 = np.zeros((1,9))
        Psi0[0,0] = 1
        nfreqs = drFreqList.size
        ntimes = t.size
        g0qqLS = np.zeros((8,8))
        qFreqsLS = ((np.random.rand(8,)-0.5)*0.01+1)*5-((np.random.rand(8,)-0.5)*0.1+1)*1e-5/2*1j
        randomAmpInd = np.random.rand(1,)
        randomAmpLs = 20*np.random.rand(1,)*0.5e-3
        driveTransferMatLs = np.random.rand(8,8)/10
        g0qq = g0qqLS
        qFreqs = qFreqsLS
        driveTransferMat = driveTransferMatLs
        convTensors = torch.zeros(8*1,nfreqs,ntimes)
        p = int(np.round(7*randomAmpInd[0]))
        driveTransferMat[p,p] = 1
        driveTransferMat[:,p] = driveTransferMat[:,p]/np.linalg.norm(driveTransferMat[:,p])
        driveAmps = 10*0.5e-3* driveTransferMat[:,p]
        PsiT = qc_helper.multiFreqSol(Psi0,qFreqs,g0qq,driveAmps,drFreqList,t)
        convTensor = 1*qc_helper.torchInput3DTensor(PsiT)
        convTensors[0:8,:,:] = convTensor
        target = (driveTransferMat[:,p].flatten())*1e2
        target = np.append(target,np.real(qFreqs)/5)
        target = np.append(target,-np.imag(qFreqs)*100*t[-1])
        targetLabels = (torch.from_numpy(target.reshape(1,-1)))
        inputImagBatch = convTensors.unsqueeze(dim=0).double()
        labelBatch= targetLabels.unsqueeze(dim=0).double()
        outputs = self(inputImagBatch)
        return (outputs, labelBatch)

    def randomDataGen(self,configs):
        Nepoch = configs["Nepoch"]
        Nstep = configs["Nstep"]
        ReportNstep = configs["ReportNstep"]
        BatchSize = configs["BatchSize"]
        Ntimes = self.Ntimes
        Nfreqs = self.Nfreqs
        Psi0 = np.zeros((1,9))
        Psi0[0,0] = 1
        t = np.linspace(0,2000,Ntimes)
        drFreqList = np.linspace(0.96,1.04,Nfreqs)*5
        g0qqLS = np.zeros((Nstep,BatchSize,8,8))
        qFreqsLS = ((np.random.rand(Nstep,BatchSize,8)-0.5)*0.01+1)*5-((np.random.rand(Nstep,BatchSize,8)-0.5)*0.1+1)*1e-5/2*1j
        randomAmpInd = np.random.rand(Nstep,BatchSize,1)
        randomAmpLs = 20*np.random.rand(Nstep,BatchSize,1)*0.5e-3
        driveTransferMatLs = np.random.rand(Nstep,BatchSize,8,8)/10
        inputImagBatch = torch.empty(BatchSize,8*1,Nfreqs,Ntimes)
        labelBatch = torch.empty(BatchSize,8*2+8+28*0+64*0)
        for epoch in range(Nepoch):
            for step in range(Nstep):
                for batch in range(BatchSize):
                    g0qq = g0qqLS[step,batch,:,:]
                    qFreqs = qFreqsLS[step,batch,:]
                    driveTransferMat = driveTransferMatLs[step,batch,:,:]
                    convTensors = torch.zeros(8*1,Nfreqs,Ntimes)
                    p = int(np.round(7*randomAmpInd[step,batch][0]))
                    driveTransferMat[p,p] = 1
                    driveTransferMat[:,p] = driveTransferMat[:,p]/np.linalg.norm(driveTransferMat[:,p])
                    driveAmps = 10*0.5e-3* driveTransferMat[:,p]
                    PsiT = qc_helper.multiFreqSol(Psi0,qFreqs,g0qq,driveAmps,drFreqList,t)
                    convTensor = 1*qc_helper.torchInput3DTensor(PsiT)
                    convTensors[0:8,:,:] = convTensor
                    target =  (driveTransferMat[:,p].flatten())*1e2
                    target=np.append(target,np.real(qFreqs)/5)
                    target=np.append(target,-np.imag(qFreqs)*100*t[-1])
                    targetLabels = (torch.from_numpy(target.reshape(1,-1)))
                    inputImagBatch[batch,:,:,:] = convTensors
                    labelBatch[batch,:] = targetLabels
        return {"labelBatch":labelBatch,"inputImagBatch":inputImagBatch}

    def train(self,configs,data=None):
        lr = configs["learningRate"]
        Nepoch = configs["Nepoch"]
        Nstep = configs["Nstep"]
        ReportNstep = configs["ReportNstep"]
        BatchSize = configs["BatchSize"]
        Ntimes = self.Ntimes
        Nfreqs = self.Nfreqs
        device = configs["device"]
        net = nn.DataParallel(self)
        net.to(device)
        if data==None:
            data = self.randomDataGen(configs)   
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        running_loss = [0,0,0.0]
        for epoch in range(Nepoch):
            running_loss[2] = 0.0
            for step in range(Nstep):
                inputImgs = data["inputImagBatch"].double().to(device)
                targetBatch = data["labelBatch"].double().to(device)
                optimizer.zero_grad()
                outputs = net(inputImgs)
                loss = criterion(outputs, targetBatch)
                loss.backward()
                optimizer.step()
                self.writer.add_histogram('conv1_weights',net.module.conv1.weight, step*(epoch+1))
                self.writer.add_histogram('conv2_weights',net.module.conv2.weight, step*(epoch+1))
                self.writer.add_histogram('conv3_weights',net.module.conv3.weight, step*(epoch+1))
                self.writer.add_histogram('fc1_weights',net.module.fc1.weight, step*(epoch+1))
                running_loss[2] += loss.item()
                running_loss[0] =  epoch + 1
                running_loss[1] = step + 1
                self.writer.add_scalar('loss vs step',loss.item(), step*(epoch+1))
                if step % ReportNstep == ReportNstep-1:  
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, step + 1, running_loss[2] / ReportNstep))
                    running_loss[2] = 0.0
        self.load_state_dict(net.module.state_dict())
        return self

    def save(self,filename):
        torch.save({
            'model_state_dict': self.state_dict()
            }, filename)
    def load(self,filename):
        netload = torch.load(filename)
        self.load_state_dict(netload['model_state_dict'])

