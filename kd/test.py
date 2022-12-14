from general import *
from studentmodel.SWWNetModel import *
import torch

np.random.seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
gpuAvailable = torch.cuda.is_available()
device = torch.device('cuda:0' if gpuAvailable else 'cpu')
audio_path = r'E:/dataset/dataset1/single_audio_wav'
video_path = r'E:/dataset/dataset1/new_video_npy1'
PATH = r'D:\get\talk\checkpoints\models\train-step_0076-loss_0.918.pt'
dsets = {x: MyDataSet(x, audio_path, video_path) for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=2, shuffle=False, num_workers=0,
                                               collate_fn=mycollate_fn) for x in ['train', 'val']}

'''
model = SWWNetModel()
model.load_state_dict(torch.load(PATH))
model = model.to(device)
'''



model = STUSWWNetModel()
import numpy as np

A = np.random.rand(1, 112, 40)
A = torch.from_numpy(A)
A = A.float()
A1 = np.random.rand(1,111, 112, 112)
A1 = torch.from_numpy(A1)
A1 = A1.float()
A2 = np.random.rand(1, 112, 512)
A2 = torch.from_numpy(A2)
A2 = A2.float()
model = STUSWWNetModel()

from thop import profile

flops, params = profile(model, [A,A1])
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
model_dict = model.state_dict()
pretrained_dict = torch.load(PATH)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model = model.to(device)

acc,validationLoss = evaluate_network(model,dset_loaders['val'],device)
print(validationLoss)

'''

'''
