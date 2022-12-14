import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mymodel.audioFrontend import ResNet,audioEncoder
from mymodel.visualFrontend import visualFrontend
from mymodel.attentionLayer import attentionLayer
from mymodel.attentionLayer import SelfAttention
from mymodel.attentionLayer import PositionalEncoding

class SWWNetModel(nn.Module):
    def __init__(self):
        super(SWWNetModel, self).__init__()
        # Visual Temporal Encoder
        self.visualFrontend  = visualFrontend() # Visual Frontend 
        # self.visualFrontend.load_state_dict(torch.load('visual_frontend.pt', map_location="cuda"))
        # for param in self.visualFrontend.parameters():
        #     param.requires_grad = False

        # Audio Temporal Encoder 
        self.audioEncoder  = ResNet()
        self.audioConv = nn.Conv1d(512,512,kernel_size=4,stride=4,padding=0)
        # Audio-visual Cross Attention
        self.crossA2V = attentionLayer(d_model = 512, nhead = 8)
        self.crossV2A = attentionLayer(d_model = 512, nhead = 8)


        self.jointConv = nn.Conv1d(2 * 512, 512, kernel_size=1, stride=1, padding=0)

        # Audio-visual Self Attention
        self.selfAV = attentionLayer(d_model = 512, nhead = 8)
        self.attention_cross = SelfAttention(512, 8, 0.1)

        self.linear1 = nn.Linear(512, 5)

        self.norm = nn.LayerNorm(512)
        self.audiolayer = audioEncoder(layers=[3, 4, 6, 3], num_filters=[64, 128, 256, 512])
        self.dropout = nn.Dropout(0.4)
        self.model1 = nn.Sequential(
            nn.Linear(512, 64),
            nn.Dropout(0.1),  # 以0.5的概率断开
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 6),
            nn.LeakyReLU(inplace=True),
        )

        self.EncoderPositionalEncoding = PositionalEncoding(dModel=512, maxLen=2500)


    def forward_visual_frontend(self, x):
        x = self.visualFrontend(x)
        return x

    def forward_audio_frontend(self, x):
        '''
        x = x.unsqueeze(1).transpose(2, 3)
        x = self.audiolayer(x)
        #print(x.shape)
        x = x.transpose(0,1)
        '''
        x = self.audioEncoder(x)
        x = x.permute(2,0,1)
        return x

    def forward_cross_attention(self, x1, x2):
        x1 = x1.permute(2,1,0)
        x2 = x2.permute(1,0,2)

        #x1_c = self.crossA2V(src = x1, tar = x2)
        #x2_c = self.crossV2A(src = x2, tar = x1)
        #return x1_c, x2_c
        return x1,x2


    def forward_audio_visual_backend(self, x1, x2):
        x = torch.cat((x1,x2), 2)
        x = x.transpose(0,1).transpose(1,2)
        x = self.jointConv(x)
        x = x.transpose(1,2).transpose(0,1)
        x = self.norm(x)
        x = self.selfAV(src = x, tar = x)

        x = self.linear1(x)
        '''
        x = F.relu(x)
        x = self.dropout(x)

        x = self.model1(x)
        '''

        x = torch.mean(x, 0)
        return x

    def forward(self, audioFeature, visualFeature):

        audioEmbed = self.forward_audio_frontend(audioFeature)
        visualEmbed = visualFeature

        visualEmbed = torch.sum(visualEmbed,1)

        '''
         visualEmbed = visualEmbed.contiguous().view(-1, visualEmbed.shape[2], visualEmbed.shape[3])
        visualEmbed.requires_grad = True

        audioEmbed = audioEmbed.transpose(1,0)
        visualEmbed = self.attention_cross(audioEmbed, visualEmbed)
        '''
        audioEmbed = audioEmbed.transpose(1, 0)
        audioEmbed = audioEmbed.transpose(1, 0)
        visualEmbed = visualEmbed.transpose(1, 0)
        '''
        visualEmbed = self.EncoderPositionalEncoding(visualEmbed)
        audioEmbed = self.EncoderPositionalEncoding(audioEmbed)
        '''

        visualEmbed = self.selfAV(src=visualEmbed, tar=visualEmbed)
        audioEmbed = self.selfAV(src=audioEmbed, tar=audioEmbed)
        out_wake_word = self.forward_audio_visual_backend(audioEmbed, visualEmbed)



        '''
        ########这里有一点不好，应该把self-attention放在选择以后########
        ########就算视频的人脸选择错误按照音频的特征也应该可以在一个比较高的准确率下对于人脸进行识别###########
        audioEmbed = self.forward_audio_frontend(audioFeature)
        # visualEmbed = self.forward_visual_frontend(visualFeature)
        visualEmbed = visualFeature
        visualEmbed = visualEmbed.contiguous().view(-1, visualEmbed.shape[2], visualEmbed.shape[3])
        visualEmbed.requires_grad = True
        visualEmbed = visualEmbed.transpose(1, 0)
        print(visualEmbed)
        #audioEmbed = audioEmbed.permute(2,1,0)
        #print(visualEmbed.shape,audioEmbed.shape)
        visualEmbed = self.selfAV(src=visualEmbed, tar=visualEmbed)
        print(visualEmbed)
        audioEmbed = self.selfAV(src=audioEmbed, tar=audioEmbed)
        visualEmbed = visualEmbed.transpose(0, 1)
        audioEmbed = audioEmbed.transpose(0, 1)

        # 这里要把视频转换为b*t-f

        # 后面考虑对音频信号进行归一化和标准化，这里先不对它进行处理

        visualEmbed = self.attention_cross(audioEmbed, visualEmbed)
        print(visualEmbed)
        visualEmbed = visualEmbed.transpose(1, 0)
        audioEmbed = audioEmbed.transpose(1, 0)
        #print(visualEmbed.shape, audioEmbed.shape)
        # audioEmbed,visualEmbed = self.forward_cross_attention(audioEmbed,visualEmbed)

        out_wake_word = self.forward_audio_visual_backend(audioEmbed, visualEmbed)
        
        
        '''

        return out_wake_word