import torch.nn as nn
from mymodel.audioFrontend import ResNet
from mymodel.attentionLayer import SelfAttention,attentionLayer
import torch.nn.functional as F
import torch
from mymodel.SelectOnePeople import select_people

class SWWNetModel(nn.Module):
    def __init__(self):
        super(SWWNetModel, self).__init__()

        self.audioEncoder = ResNet()
        self.jointConv = nn.Conv1d(2 * 512, 512, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(512)
        self.crossA2V = attentionLayer(d_model=512, nhead=8)
        self.crossV2A = attentionLayer(d_model=512, nhead=8)
        self.selfAV = attentionLayer(d_model=512, nhead=8)
        self.linear1 = nn.Linear(512, 5)
        self.norm = nn.LayerNorm(512)


    def forward_audio_frontend(self, x):
        x = self.audioEncoder(x)
        return x

    def forward_visual_frontend(self, x):
        x = self.visualFrontend(x)
        return x


    def forward_cross_attention(self, x1, x2):
        x1 = self.selfAV(src=x1, tar=x1)
        x2 = self.selfAV(src=x2, tar=x2)

        x1_c = self.crossA2V(src=x1, tar=x2)
        x2_c = self.crossV2A(src=x2, tar=x1)
        return x1_c, x2_c


    def forward_audio_visual_backend(self, x1, x2):
        x = torch.cat((x1, x2), 2)
        x = x.transpose(1, 2)
        x = self.jointConv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = F.relu(x)
        x = self.selfAV(src=x, tar=x)
        x = self.linear1(x)
        x = torch.mean(x, 1)
        return x

    def forward(self, audioFeature, visualFeature):

        audioEmbed = self.forward_audio_frontend(audioFeature)
        audioEmbed = audioEmbed.transpose(1, 2)
        audioEmbed, visualEmbed = self.forward_cross_attention(audioEmbed, visualFeature)
        out_wake_word = self.forward_audio_visual_backend(audioEmbed, visualEmbed)
        return out_wake_word


if __name__ == '__main__':
    import torch
    a = torch.arange(0,16)    #此时a的shape是(1,20)
    b = a.view(8,2)
    c = b.view(4,2,2)
    e = c.transpose(1,0)
    d = a.view(2,4,2)
    print(c)
    print(d.shape)    #输出为(4,5)
    print(e.shape)
