import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention
import torch
import matplotlib.pyplot as plt
import numpy as np


class attentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(attentionLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor

        src2 = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout,len):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # d_model // h 仍然是要能整除，换个名字仍然意义不变
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.trans_k_to_v = nn.Linear(hid_dim,hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))
        self.list = list()
        self.list2 = list()
        self.len = len

    def forward(self, query, key, mask=None):

        bsz = query.shape[0]
        msz = key.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)

        K = K.view(bsz, msz//bsz, -1, self.hid_dim)
        key = key.view(bsz, msz // bsz, -1, self.hid_dim)
        Q = torch.unsqueeze(Q, 1)

        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.cuda()

        # 如果没有mask，就生成一个
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # 每一帧的得分
        total_attention = torch.sum(energy, dim=-2)


        temp_total_attention = torch.squeeze(total_attention,0)
        temp_total_attention = torch.softmax(temp_total_attention,0)
        print(temp_total_attention.shape)
        temp_total_attention = temp_total_attention.cpu().tolist()


        plt.title('Scores by person and speaker')
        Bottom = temp_total_attention[0]

        N = len(Bottom)
        ind = np.arange(N)

        plt.ylabel('Scores')
        width = 1  # 设置条形图一个长条的宽度

        if self.len == 1:
            p1 = plt.bar(ind, Bottom, width, color='blue')
            plt.legend((p1[0]), ('p1'), loc=1)
        elif self.len == 2:
            Center = temp_total_attention[1]

            p1 = plt.bar(ind, Bottom, width, color='blue')
            p2 = plt.bar(ind, Center, width, bottom=Bottom, color='green')
            plt.legend((p1[0], p2[0]), ('p1', 'p2'), loc=2)
        elif self.len == 3:
            Center = temp_total_attention[1]
            Top = temp_total_attention[2]
            d = []
            for i in range(0, len(Bottom)):
                sum = Bottom[i] + Center[i]
                d.append(sum)

            p1 = plt.bar(ind, Bottom, width, color='blue')
            p2 = plt.bar(ind, Center, width, bottom=Bottom, color='green')
            p3 = plt.bar(ind, Top, width, bottom=d, color='red')
            plt.legend((p1[0], p2[0], p3[0]), ('p1', 'p2', 'p3'), loc=3)
        plt.savefig(r'C:\Users\WGQ\Desktop\qtchoose\static/'+'choose' + '.jpg')
        plt.close()


        # 然后对得分进行softmax
        attention = torch.mean(total_attention, dim=-1)

        _,max_index = torch.max(attention,-1)
        max_index = max_index.cpu().detach().numpy()
        for i in range(bsz):
            k = key[i,max_index[i],:,:]
            self.list2.append(k)
        val = torch.tensor([item.cpu().detach().numpy() for item in self.list2]).cuda()
        val.requires_grad = True
        self.list2.clear()
        return val
