import torch.nn as nn
from mymodel.visualFrontend import visualFrontend
import torch.nn.functional as F
import math
import torch



class SWWNetModel(nn.Module):
    def __init__(self):
        super(SWWNetModel, self).__init__()
        self.v = visualFrontend()


    def forward(self, audioFeature, visualFeature):
        print(audioFeature.shape,visualFeature.shape)
        audioFeature = torch.unsqueeze(audioFeature,1)
        out_wake_word = self.v(visualFeature,audioFeature)

        return out_wake_word


