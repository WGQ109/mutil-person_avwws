import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention
import torch
import numpy as np
import math

class attentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(attentionLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, 4*d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(4*d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor

        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C

        src2 = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1) # T, B, C -> B, T, C
        return src

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
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

    def forward(self, query, key, mask=None):
        bsz = query.shape[0]
        msz = key.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)

        K = K.view(bsz, msz//bsz, -1, self.hid_dim)
        key = key.view(bsz, msz // bsz,  -1, self.hid_dim)

        Q = torch.unsqueeze(Q, 1)

        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.cuda()

        # 如果没有mask，就生成一个
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # 每一帧的得分
        total_attention = torch.sum(energy, dim=-2)
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
'''
bsz = query.shape[0]
        msz = key.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        key = key.view(msz // bsz, bsz, -1, self.hid_dim)
        K = K.view(msz//bsz, bsz, -1, self.hid_dim)


        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.cuda()
        #print(energy)
        # 如果没有mask，就生成一个
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：
        attention = torch.softmax(energy, dim=-1)
        #print(energy)
        #print(attention.shape)
        total_attention = torch.sum(attention,dim=-2)

        #print(total_attention.shape)
        ll, l = torch.max(total_attention, 0)
        l = l.cpu().detach().numpy()
        array = l
        for i in range(len(array)):
            self.list.append(int(np.argmax(np.bincount(array[i]))))
        #print(self.list)
        for i in range(bsz):
            k = key[self.list[i],i,:,:]
            self.list2.append(k)
        val = torch.tensor([item.cpu().detach().numpy() for item in self.list2]).cuda()
        val.requires_grad = True
        self.list.clear()
        self.list2.clear()
        return val
'''




'''
        # Q,K,V计算与变形：

bsz = query.shape[0]
msz = key.shape[0]

Q = self.w_q(query)
K = self.w_k(key)


Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
           self.n_heads).permute(0, 2, 1, 3)
K = K.view(msz // bsz, bsz, -1, self.n_heads, self.hid_dim //
           self.n_heads).permute(0, 1, 3, 2, 4)


# Q, K相乘除以scale，这是计算scaled dot product attention的第一步

energy = torch.matmul(K,Q.permute(0,1,3,2)) / self.scale.cuda()

# 如果没有mask，就生成一个

if mask is not None:
    energy = energy.masked_fill(mask == 0, -1e10)

# 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：

energy,_ = torch.max(energy,0)


attention = self.do(torch.softmax(energy, dim=-1))

# 第三步，attention结果与V相乘

x = torch.matmul(attention, K)

# 最后将多头排列好，就是multi-head attention的结果了

x = x.permute(0, 1, 3, 2, 4).contiguous()

x = x.view(msz // bsz, bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

energy, _ = torch.max(x, 0)

energy = self.fc(energy)

return energy


'''

class PositionalEncoding(nn.Module):
    """
    A layer to add positional encodings to the inputs of a Transformer model.
    Formula:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float() * (math.log(10000.0) / dModel))
        pe[:, 0::2] = torch.sin(position / denominator)
        pe[:, 1::2] = torch.cos(position / denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.shape[0], :, :]
        return outputBatch