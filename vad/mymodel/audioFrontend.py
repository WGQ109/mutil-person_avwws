import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.dconv = nn.Conv1d(13,512,kernel_size=4,stride=4,padding=0,bias=False)
        self.lybn = nn.LayerNorm(512)
        return

    def forward(self, inputBatch):
        inputBatch = inputBatch.transpose(1,2)
        x = self.dconv(inputBatch)
        x = x.transpose(1, 2)
        x = self.lybn(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        return x