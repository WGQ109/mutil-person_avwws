from general import *
from mymodel.SWWNetModel import *
import torch

np.random.seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
gpuAvailable = torch.cuda.is_available()
device = torch.device('cuda:0' if gpuAvailable else 'cpu')
audio_path = r'D:/dataset/single_audio_wav'
video_path = r'D:/dataset/new_video_npy1'
PATH = r'D:\get\20220412-1talk\checkpoints\models\train-step_0034-loss_0.980.pt'
dsets = {x: MyDataSet(x, audio_path, video_path) for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=2, shuffle=False, num_workers=0,
                                               collate_fn=mycollate_fn) for x in ['train', 'val']}

'''
model = SWWNetModel()
model.load_state_dict(torch.load(PATH))
model = model.to(device)
'''

model = SWWNetModel()
model_dict = model.state_dict()
pretrained_dict = torch.load(PATH)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model = model.to(device)

acc,validationLoss = evaluate_network(model,dset_loaders['val'],device)
print(validationLoss)