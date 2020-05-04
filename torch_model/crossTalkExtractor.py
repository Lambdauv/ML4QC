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

import torch
import numpy as np
import torch.nn as nn
import ML4QC.torch_tensormath.torch_helper as torch_helper
class crossTalkConvNet(nn.Module):
    def __init__(self,inputDim,numConvStack=1):
        super(Net, self).__init__()
        # 8 input image channel, 64 output channels, 10x5 square convolution
        # kernel
        self.numConvStack = 1
        self.inputDims = inputDim
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
 #       self.conv4OutSz = conv2dOutSize(self.kernelDim4,self.padding,self.stride4,(1,8),self.conv3OutSz)
        self.conv1 = nn.Conv2d(8, 16, self.kernelDim1,stride=self.stride1,padding=self.padding,dilation=(1,1)) # Image dimensions reduce by (7*1)x3
        self.conv2 = nn.Conv2d(16, 32, self.kernelDim2,stride=self.stride2,padding=self.padding,dilation=(1,2)) # Image dimensions reduce by (7*2)x3
        self.conv3 = nn.Conv2d(32, 64, self.kernelDim3,stride=self.stride3,padding=self.padding,dilation=(1,4)) # Image dimensions reduce by (7*4)x3
#        self.conv4 = nn.Conv2d(128, 128, self.kernelDim4,stride=self.stride4,padding=self.padding,dilation=(1,8)) # Image dimensions reduce by (7*8)x3
#         self.bn1 = nn.BatchNorm2d(24)
#         self.bn2 = nn.BatchNorm2d(24)
#         self.bn3 = nn.BatchNorm2d(24)
        # an affine operation: y = Wx + b
      #  self.fc1 = nn.Linear(64*int(self.conv3OutSz[0]*self.conv3OutSz[1]), 128+0*(8*2+28*2))  # 2048*512 from image dimension. This needs to be a
#         self.fc2 = nn.Linear(1024, 256)
#         self.fc3 = nn.Linear(128, )
#        self.fc4 = nn.Linear(8*3+28*2,128)
#         self.fc5 = nn.Linear(128,128)
       # self.fc6 = nn.Linear(128,8*2+28*0+8)
        self.fc6 = nn.Linear(64*int(self.conv3OutSz[0]*self.conv3OutSz[1]),8*2+28*0+8)
    def forward(self,inputImg):
#         paramSz = inputParams.size()
#         paramSeq = inputParams.view(-1, paramSz[0])
       # paramSeq = inputParams
        x = inputImg
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
#        x = F.relu(self.conv4(x))
        x = x.view(-1, self.num_flat_features(x))
        #x = torch.cat((paramSeq,x),1)
        #x = F.relu(self.fc1(x))
       # x = F.hardshrink(self.fc1(x), lambd=0.5) 
#         x = F.relu(self.fc2(x))
        #convOut = F.relu(self.fc1(x))
#        x = torch.cat((paramSeq,convOut),1)
#        x = F.relu(self.fc4(x))
#         x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x   #torch.cat((x,convOut),1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features