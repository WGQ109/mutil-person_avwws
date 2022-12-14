import torch
from torch.nn.utils.rnn import pad_sequence
from cvtransforms import *
import librosa


def load_file(video_path,audio_path,filename):

    cap = np.load('E:/dataset/dataset1/select_video_npy/'+ filename + '.npy')
    y = np.load('E:/dataset/dataset1/new_people_npy/'+ filename + '.npy')
    audio = np.load('E:/dataset/dataset1/wav2dev_large/' + filename + '.npy')
    sig,sr = librosa.load('E:/dataset/dataset1/single_audio_wav/' + filename + '.wav',sr=16000)
    mfcc = librosa.feature.mfcc(sig, sr=16000, n_mfcc=40)
    mfcc = mfcc.T
    x = np.array(mfcc, np.float32, copy=False)
    '''
        A = np.random.rand(125,112,112)
    vidInp = A
    '''
    vidInp = cap


    return audio,vidInp,x,y


class MyDataSet():
    def __init__(self, folds, audio_path, video_path):
        self.folds = folds
        self.audio_path = audio_path
        self.path = video_path
        self.filenames = self.path + '/' + self.folds + '.txt'
        self.list = {}
        i = 0

        with open(self.filenames) as myfile:
            data_dir = myfile.read().splitlines()

        for elem in data_dir:
            if int(elem[1:5]) > 0 and int(elem[1:5]) <= 2000:
                self.list[i] = [elem[:-4]]
                label_target = elem[6:11]
                # 这里相当于对标签进行了处理了，现在就只需要对图片进行处理了
                for kk, label in enumerate(label_target):
                    if int(label) == 1:
                        self.list[i].append([kk])
                self.list[i].append([elem])
                i = i + 1
        print(len(self.list))

    def __getitem__(self, index):
        inputs = load_file(self.path ,self.audio_path, self.list[index][0])
        labels = self.list[index][1]
        k = self.list[index][2]

        return inputs, labels, k

    def __len__(self):
        return len(self.list)

def mycollate_fn(batch_data):

    batch_data.sort(key=lambda xi: len(xi[0][0]), reverse=True)
    audio_sent_seq = [torch.tensor(xi[0][0],dtype=torch.float32) for xi in batch_data]
    batch_data.sort(key=lambda xi: len(xi[0][1]), reverse=True)
    video_sent_seq = [torch.tensor(xi[0][1], dtype=torch.float32) for xi in batch_data]
    batch_data.sort(key=lambda xi: len(xi[0][2]), reverse=True)
    stuaudio_sent_seq = [torch.tensor(xi[0][2], dtype=torch.float32) for xi in batch_data]
    batch_data.sort(key=lambda xi: len(xi[0][3]), reverse=True)
    stuvideo_sent_seq = [torch.tensor(xi[0][3], dtype=torch.float32) for xi in batch_data]

    wake_word_label = [torch.tensor(xi[1][:],dtype=torch.long) for xi in batch_data]
    a = [xi[2][:] for xi in batch_data]
    audio_padded_sent_seq = pad_sequence(audio_sent_seq, batch_first=True, padding_value=0)
    video_padded_sent_seq = pad_sequence(video_sent_seq, batch_first=True, padding_value=0)
    wake_word_sent_seq = pad_sequence(wake_word_label,batch_first=True, padding_value=0)
    stuaudio_word_sent_seq = pad_sequence(stuaudio_sent_seq, batch_first=True, padding_value=0)
    stuvideo_word_sent_seq = pad_sequence(stuvideo_sent_seq, batch_first=True, padding_value=0)

    return audio_padded_sent_seq,video_padded_sent_seq,wake_word_sent_seq,stuaudio_word_sent_seq,stuvideo_word_sent_seq,a
