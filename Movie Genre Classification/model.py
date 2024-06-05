import torch
import numpy as np
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class Model(nn.Module):
    def __init__(self, num_classes=27):
        super(Model, self).__init__()
        self.l1=nn.LazyLinear(out_features=120)
        self.l2=nn.LazyLinear(out_features=200)
        self.l3=nn.LazyLinear(out_features=50)
        self.l4=nn.LazyLinear(out_features=num_classes)

        
    
    def forward(self, x):
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=self.l4(x)
        x=F.log_softmax(x, dim=1)
        return x

