#%%
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim # Optimizer

import torchvision
from torchvision import transforms

import numpy as np
import cv2
from scipy import ndimage

import PIL
from PIL import Image

import matplotlib.pyplot as plt
from typing import List

class GramMatrix(nn.Module):
    ## Fucking hard to debug
    def forward(self, input : torch.Tensor)->torch.Tensor:
        b,c,h,w = input.size() # Batch : int , Channel : int , height : int, width : int 
        fe : torch.Tensor = input.view(b, c, h*w) ## make a pointer, ( batch , channel, resolution )
        Gram : torch.Tensor = torch.bmm(fe, torch.transpose(fe, 1, 2)) # Transpose the channel and h*w
        Gram.div_(h*w)
        return Gram

class GramMSELoss(nn.Module):
    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        out : torch.Tensor = nn.MSELoss().forward(GramMatrix(input),target)
        return out

def savePlotResult(loss_list : List[torch.Tensor], label : str, output_path : str):
    plt.plot(loss_list, label=label)
    plt.legend()

    plt.savefig(f"{output_path}{label}.jpg")
    plt.close() ## Prevent to override the figure window


#%%
preprocessingImage = transforms.Compose([
    transforms.Lambda(lambd= lambda x : x.div(255.)), # For each pixels do division,
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]), ## Using imageNet mean and std,
    transforms.Lambda(lambd=lambda x : x[torch.LongTensor([2,1,0])] ) ## Turn to BGR

])

postPb = transforms.Compose([transforms.ToPILImage()])




