import torch.nn as nn
from studentmodel.audioFrontend import TCResNet
from studentmodel.mobilenet import MobileNetV2
from studentmodel.visualFrontend import visualFrontend
import torch.nn.functional as F
import math
import torch


class PositionalEncoding(nn.Module):


    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float()*(math.log(10000.0)/dModel))
        pe[:, 0::2] = torch.sin(position/denominator)
        pe[:, 1::2] = torch.cos(position/denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pe", pe)


    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.shape[0],:,:]
        return outputBatch

class STUSWWNetModel(nn.Module):
    def __init__(self):
        super(STUSWWNetModel, self).__init__()
        self.positionalEncoding = PositionalEncoding(dModel=128, maxLen=600)
        encoderLayer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=128*4, dropout=0.1)
        self.selfAV = nn.TransformerEncoder(encoderLayer, num_layers=4)
        self.videoEncoder = visualFrontend()
        self.linear1 = nn.Linear(128, 5)
        self.norm = nn.LayerNorm(128)
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(16, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.mobilenet = MobileNetV2()
        self.tc = TCResNet(40, [16, 32, 32, 64, 64, 128, 128], 5)


    def forward_video_frontend(self, x):
        x = x / 255
        x = (x - 0.4161) / 0.1688
        x = self.videoEncoder(x)
        return x

    def kmax_pooling(self, x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def forward_video_backend(self,x):
        x = x.transpose(0,1)
        self.positionalEncoding = PositionalEncoding(dModel=128, maxLen=600)
        x = self.selfAV(src=x)
        x = x.transpose(0,1)
        x = self.kmax_pooling(x, 1, 25)
        x = self.linear1(x)
        x = torch.mean(x, 1)
        return x



    def forward(self, audioFeature, visualFeature):
        video_wake_word = self.forward_video_frontend(visualFeature)
        video_wake_word = video_wake_word.transpose(0,1)
        out_wake_word2 = self.forward_video_backend(video_wake_word)
        audioFeature = torch.unsqueeze(audioFeature, 1)
        audioFeature = audioFeature.transpose(2, 3)
        audioFeature = self.tc(audioFeature)
        return audioFeature,out_wake_word2


if __name__ == '__main__':
    import torch

    a = torch.randn(5, 6)

    x = [5, 4, 3, 2, 1]
    mask = torch.zeros(5, 6, dtype=torch.float)
    for e_id, src_len in enumerate(x):
        mask[e_id, src_len:] = 1
    mask = mask.to(device='cpu')
    print(mask)
    a.data.masked_fill_(mask==0, -float('inf'))
    print(a)

