import torch
import torch.nn.functional as F
from typing import List

def getPatches(feature_map : List[torch.Tensor], k : int = 5, stride : int = 1):
    patches : List [torch.Tensor] = []
    for i in range(0, feature_map.shape[3] - k, stride ): # Loop towards the chanell
        for j in range(0, feature_map.shape[3] - k, stride):
            patch = feature_map[:, :, i:i+k, j:j+k]
            patches.append(patch)

def dividePatches(style_patches):
    ...
