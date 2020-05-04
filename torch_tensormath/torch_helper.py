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

def conv2dOutSize(kernalSz,padding,stride,dilation,InputImagShape):
    Hout = np.floor((InputImagShape[0]+2*padding[0]-dilation[0]*(kernalSz[0]-1)-1)/stride[0])+1
    Wout = np.floor((InputImagShape[1]+2*padding[1]-dilation[1]*(kernalSz[1]-1)-1)/stride[1])+1
    return (Hout, Wout)