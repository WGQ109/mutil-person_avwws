import torch.nn as nn
from mymodel.audioFrontend import audioEncoder
from mymodel.attentionLayer import SelfAttention,attentionLayer
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

class SWWNetModel(nn.Module):
    def __init__(self):
        super(SWWNetModel, self).__init__()

        self.audioEncoder = audioEncoder(layers = [3, 4, 6, 3],  num_filters = [64, 128, 256, 512])
        self.positionalEncoding = PositionalEncoding(dModel=512, maxLen=600)
        encoderLayer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.selfAV = nn.TransformerEncoder(encoderLayer, num_layers=4)
        self.linear1 = nn.Linear(512, 5)
        self.norm = nn.LayerNorm(512)
        self.audioConv = nn.Conv1d(512, 512, kernel_size=4, stride=4, padding=0)
        self.crossA2V = attentionLayer(d_model=512, nhead=8)
        self.crossV2A = attentionLayer(d_model=512, nhead=8)

    def forward_audio_frontend(self, x):
        x = self.audioEncoder(x)
        return x

    def kmax_pooling(self, x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def forward_audio_backend(self,x,mask):
        x = x.transpose(0,1)
        self.positionalEncoding = PositionalEncoding(dModel=512, maxLen=600)
        x = self.selfAV(src=x)
        x = x.transpose(0,1)
        x = self.kmax_pooling(x, 1, 25)
        x = self.linear1(x)
        x = torch.mean(x, 1)
        return x

    def forward_video_backend(self,x,mask):
        x = x.transpose(0,1)
        self.positionalEncoding = PositionalEncoding(dModel=512, maxLen=600)
        x = self.selfAV(src=x)
        x = x.transpose(0,1)
        x = self.kmax_pooling(x, 1, 25)
        x = self.linear1(x)
        x = torch.mean(x, 1)
        return x


    def forward(self, audioFeature, visualFeature,mask):
        #audioEmbed = self.forward_audio_frontend(audioFeature)
        '''


        '''
        audioEmbed = audioFeature.transpose(1, 2)
        audioEmbed = self.audioConv(audioEmbed)
        audioEmbed = audioEmbed.transpose(1, 2)






        out_wake_word = self.forward_audio_backend(audioFeature,mask)

        return out_wake_word,audioEmbed


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

