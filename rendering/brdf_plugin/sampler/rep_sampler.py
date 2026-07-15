import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import os
import yaml
from brdf_plugin.utils.ops import double_sided

from brdf_plugin.material.anisotropicLatent import AnisotropicLatentTexturedModel
from brdf_plugin.material.isotropicLatent import LatentTexturedModel
from brdf_plugin.material.svpbr import SvPBRBRDF
from brdf_plugin.utils.cuda_manage import print_cuda_memory_info
import torch.nn.functional as F

'''
class Model_T(nn.Module):
    def __init__(self,input_dim):
        super(Model_T, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x=F.silu(x)
        x = F.silu(self.fc1(x))
        x=F.silu(self.fc2(x))
        x=self.fc3(x)
        w=x[:,0:2]
        r=F.softplus(x[:,2:3])
        wi=w/torch.sqrt(torch.sum(w*w,dim=1,keepdim=True)+r*r)
        return wi,w
'''

class Model_T(nn.Module):
    def __init__(self,input_dim):
        super(Model_T, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x=self.fc1(x)
        x=F.silu(x)
        x=self.fc2(x)
        x=F.silu(x)
        x=self.fc3(x)
        w=x[:,0:2]
        r=F.softplus(x[:,2:3])
        wi=w/torch.sqrt(torch.sum(w*w,dim=1,keepdim=True)+r*r)
        return wi,w

class Model_pdf(nn.Module):
    def __init__(self,input_dim):
        super(Model_pdf, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x=F.relu(x)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x

class Model_compress(nn.Module):
    def __init__(self,input_dim):
        super(Model_compress, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 3)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x=F.relu(x)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        x=self.activation(x)
        return x
    