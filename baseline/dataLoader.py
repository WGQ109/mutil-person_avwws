import librosa
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
import numpy as np
from scipy import signal
from cvtransforms import *
import python_speech_features
from scipy.io import wavfile
import copy

# 这个函数主要就是读取npy文件，并且把图片归一化
def load_file(video_path,audio_path,filename):

    cap = np.load(video_path + '/' + filename + '.npy')
    numFrames = cap.shape[1]
    sampFreq, audio = wavfile.read(audio_path + '/' + filename + '.wav')

    fps = 25
    audio = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025 * 25 / fps, winstep=0.010 * 25 / fps)
    audio = np.abs(audio)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage = maxAudio - audio.shape[0]
        audio = np.pad(audio, ((0, shortage), (0, 0)), 'wrap')
    audInp = audio[:int(round(numFrames * 4)), :]
    xx = copy.deepcopy(audInp)

    audInp = xx / np.max(np.abs(xx))  # 幅值归一化
    cap = cap.transpose(1, 0, 2)
    vidInp = cap
    return audInp,vidInp


class MyDataSet():
    """
    Dataset的目标是根据你输入的索引输出对应的image和label，而且这个功能是要在__getitem__()函数中完成的，
    所以当你自定义数据集的时候，首先要继承Dataset类，还要复写__getitem__()函数。
    """
    def __init__(self, folds, audio_path, video_path):
        # 数据的类型，训练集，测试集还是验证集/
        self.folds = folds
        # 数据存放的路径
        self.audio_path = audio_path
        self.path = video_path
        # 找到对应路径下的txt文档，后续的处理需要根据txt文档来进行处理
        self.filenames = self.path + '/' + self.folds + '.txt'
        # 这里定义一个字典类型的数据来分别存放数据以及数据的标签（方便后续的collate_fn函数的改写工作）
        self.list = {}
        # i分别用于记录测试集，训练集和验证集的数据的数目
        i = 0

        # 读取txt中的文件内容并且在数据集中找到相应的数据
        with open(self.filenames) as myfile:
            data_dir = myfile.read().splitlines()

        for elem in data_dir:
            if int(elem[1:5]) < 5000:
                self.list[i] = [elem[:-4]]
                label_target = elem[6:12]
                # 这里相当于对标签进行了处理了，现在就只需要对图片进行处理了
                for kk,label in enumerate(label_target):
                    if int(label) == 1:
                        self.list[i].append([kk])
                self.list[i].append([elem])
                i = i + 1
        print(len(self.list))

    # index is belong to --len--
    def __getitem__(self, index):
        inputs = load_file(self.path ,self.audio_path, self.list[index][0])
        labels = self.list[index][1]
        k = self.list[index][2]
        return inputs, labels,k

    def __len__(self):
        return len(self.list)

def mycollate_fn(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param data: 元组，第一个元素：句子序列数据，第二个元素：长度 第2维：句子标签
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
    # (tensor([1, 2, 3, 5]), 4, 0)
    # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
    # pack_padded_sequence函数要求要按照序列的长度倒序排列

    batch_data.sort(key=lambda xi: len(xi[0][0]), reverse=True)
    # 注意列表转tensor和数组转tensor的区别，使用的函数是不一样的
    audio_sent_seq = [torch.tensor(xi[0][0],dtype=torch.float32) for xi in batch_data]
    batch_data.sort(key=lambda xi: len(xi[0][1]), reverse=True)
    video_sent_seq = [torch.tensor(xi[0][1], dtype=torch.float32) for xi in batch_data]
    wake_word_label = [torch.tensor(xi[1][:],dtype=torch.long) for xi in batch_data]
    a = [xi[2][:] for xi in batch_data]
    audio_padded_sent_seq = pad_sequence(audio_sent_seq, batch_first=True, padding_value=0)
    video_padded_sent_seq = pad_sequence(video_sent_seq, batch_first=True, padding_value=0)
    wake_word_sent_seq = pad_sequence(wake_word_label,batch_first=True, padding_value=0)
    video_padded_sent_seq = video_padded_sent_seq.transpose(1, 2)
    return audio_padded_sent_seq,video_padded_sent_seq,wake_word_sent_seq,a



if __name__ == '__main__':
    # 这里的path末端一定要加上/
    audio_path = r'D:/1233'
    video_path = r'D:/npy'
    dsets = {x: MyDataSet(x, audio_path, video_path) for x in ['train', 'val', 'test']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=32, shuffle=True, num_workers=0,
                                                   collate_fn=mycollate_fn) for x in ['train', 'val', 'test']}
    # 测试的过程，后续也可以直接用到主函数里面
    for phase in ['train', 'val', 'test']:
        for batch_idx, (audio_inputs, video_inputs,wake_word_labels,speaker_laebels) in enumerate(dset_loaders[phase]):
            if phase == 'train':
                # 这里还应该加上对图片的增强的操作的
                print('train audio',audio_inputs.shape)
                print('train video', video_inputs.shape)

            elif phase == 'val' or phase == 'test':
                print('val audio',audio_inputs.shape)
                print('val video', video_inputs.shape)

            else:
                raise Exception('the dataset doesn\'t exist')
