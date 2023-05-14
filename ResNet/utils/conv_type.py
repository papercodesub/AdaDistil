from itertools import combinations
import copy
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
LearnedBatchNorm = nn.BatchNorm2d
from utils.options import args
N = args.N
M = args.M
class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

DenseConv = nn.Conv2d

class NMConv(nn.Conv2d):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask =  self.init()  

    def init(self):
        return nn.Parameter(torch.ones(self.weight.shape), requires_grad = False)

    def forward(self, x):
        sparseWeight = self.mask * self.weight 
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x
