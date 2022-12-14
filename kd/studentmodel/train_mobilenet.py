
import torch
import torch.nn as nn
import torchvision.models as models
from studentmodel.mobilenet import MobileNetV2


class MobilenetVisualFrontend(nn.Module):
    def __init__(self, nClasses, vidfeaturedim):
        super(MobilenetVisualFrontend, self).__init__()
        self.nClasses = nClasses
        self.inputDim = vidfeaturedim
        self.mobilenet = MobileNetV2()
        # Conv3D
        checkpoint = torch.load('mobilenet_v2_small.pth')

        self.mobilenet.load_state_dict(checkpoint, strict=False)
        self.linear = nn.Linear(1280,5)

        # freeze features weights
        for param in self.mobilenet.features.parameters():
            param.requires_grad = False


        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(32, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.transconv2d = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(4,4), stride=(4,4), bias=False),
            nn.BatchNorm2d(32, momentum=0.01, eps=0.001),
            nn.ReLU()
        )

        self.backend_conv1 = nn.Sequential(
            nn.Conv1d(1280,640, 5, 2, 0, bias=False),
            nn.BatchNorm1d(640),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(640,  320, 5, 2, 0, bias=False),
            nn.BatchNorm1d(320),
            nn.ReLU(True),
        )
        self.backend_conv2 = nn.Sequential(
            nn.Linear(320, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(True),
            nn.Linear(160, 5)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3])
        batchsize = x.shape[0]
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 32, x.size(3), x.size(4))
        x = self.mobilenet(x)
        x = x.contiguous()
        x = x.view(batchsize, -1, x.size(1))
        x = torch.mean(x, 1)
        x = self.linear(x)
        return x


if __name__ == '__main__':

    import numpy as np
    net = MobilenetVisualFrontend(5,128).cuda()
    A = np.random.rand(1, 150, 112, 112)
    A = torch.from_numpy(A)
    A = A.float().cuda()
    out = net(A)


