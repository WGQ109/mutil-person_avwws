import torch.nn as nn
from mymodel.audioFrontend import ResNet
import torch
from mymodel.attentionLayer import SelfAttention

class select_people(nn.Module):
    def __init__(self,len):
        super(select_people, self).__init__()
        self.audioEncoder  = ResNet()
        self.attention_cross = SelfAttention(512,8,0.1,len)

    def forward_audio_frontend(self, x):
        x = self.audioEncoder(x)
        return x


    def forward(self, audioFeature, visualFeature):
        print('ll',audioFeature.shape,visualFeature.shape)
        audioEmbed = self.forward_audio_frontend(audioFeature)
        visualEmbed = visualFeature
        visualEmbed = visualEmbed.contiguous().view(-1, visualEmbed.shape[2], visualEmbed.shape[3])
        audioEmbed = audioEmbed.transpose(1, 2)
        out = self.attention_cross(audioEmbed, visualEmbed)
        return out