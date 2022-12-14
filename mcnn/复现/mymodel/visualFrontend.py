
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetLayer(nn.Module):
    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1, 1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return

    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch


class ResNet(nn.Module):
    """
    An 18-layer ResNet architecture.
    """

    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4, 4), stride=(1, 1))

        return

    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch


class visualFrontend(nn.Module):
    """
    A visual feature extraction module. Generates a 512-dim feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.
    """

    def __init__(self):
        super(visualFrontend, self).__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet = ResNet()
        self.conv3d_1_1 = nn.Sequential(nn.Conv3d(1, 32, kernel_size=(9, 3, 3), stride=(2, 1, 1), padding=(0, 0, 0), bias=False),
                        nn.BatchNorm3d(32, momentum=0.01, eps=0.001),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0)))

        self.conv3d_1_2 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=(9, 3, 3), stride=(2, 1, 1), padding=(0, 0, 0), bias=False),
                        nn.BatchNorm3d(32, momentum=0.01, eps=0.001),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0)))

        self.conv3d_2 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=(4, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0), bias=False),
                        nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0)))
        self.linear3 = nn.Linear(64, 5)
        self.linear4 = nn.Linear(256, 64)
        self.avgpool2d = nn.AvgPool2d(kernel_size=(25, 6), stride=(1, 1))
        self.avgpool3d = nn.AvgPool3d(kernel_size=(3, 2, 4), stride=(1, 1, 1))

        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(21,8), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(6,1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(6,1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(6,1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv2d_4 = nn.Conv2d(256, 256, kernel_size=(6, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(64)
        self.norm3 = nn.LayerNorm(64)
        return

    def kmax_pooling(self, x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def forward(self, inputBatch, inputBatch2):
        '''
        inputBatch = inputBatch.reshape(inputBatch.shape[0], 1, inputBatch.shape[1],inputBatch.shape[2],inputBatch.shape[3])
        batchsize = inputBatch.shape[0]
        batch = self.frontend3D(inputBatch)
        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0] * batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        outputBatch = self.resnet(batch)
        outputBatch = outputBatch.reshape(batchsize, -1, 512)
        outputBatch = outputBatch.transpose(1, 2)
        outputBatch = outputBatch.transpose(1, 2).transpose(0, 1)
        '''

        batchsize = inputBatch.shape[0]
        num_time = inputBatch.shape[1]
        q1,q2 = inputBatch.shape[2], inputBatch.shape[3]
        inputBatch1 = inputBatch.transpose(0,1)
        if num_time < 100:
            from torch.nn.utils.rnn import pad_sequence
            a = torch.zeros(100, batchsize, q1, q2).float().cuda()
            c = pad_sequence([a, inputBatch1],padding_value=0)
            inputBatch = c[:,1,:,:]
            inputBatch = inputBatch.transpose(0,1)
        inputBatch = inputBatch.reshape(inputBatch.shape[0], 1, inputBatch.shape[1], inputBatch.shape[2],inputBatch.shape[3])
        outputBatch = self.conv3d_1_1(inputBatch)
        outputBatch = self.conv3d_1_2(outputBatch)
        outputBatch = self.conv3d_2(outputBatch)
        outputBatch = self.avgpool3d(outputBatch)
        outputBatch = outputBatch.reshape(batchsize, -1)
        outputBatch = self.relu(outputBatch)
        outputBatch = self.norm3(outputBatch)
        outputBatch = self.dropout(outputBatch)
        outputBatch = self.linear3(outputBatch)


        batchsize = inputBatch2.shape[0]
        num_time = inputBatch2.shape[2]
        q1 = inputBatch2.shape[3]
        if num_time < 600:
            inputBatch2 = inputBatch2.transpose(0, 2)
            from torch.nn.utils.rnn import pad_sequence
            a = torch.zeros(600, 1, batchsize, q1).float().cuda()
            c = pad_sequence([a, inputBatch2],padding_value=0)
            inputBatch2 = c[:,1,:,:]
            inputBatch2 = inputBatch2.transpose(0,2)
        inputBatch2 = self.conv2d(inputBatch2)
        inputBatch2 = self.conv2d_1(inputBatch2)
        inputBatch2 = self.conv2d_2(inputBatch2)
        inputBatch2 = self.conv2d_3(inputBatch2)
        inputBatch2 = self.conv2d_4(inputBatch2)

        inputBatch2 = self.kmax_pooling(inputBatch2, 2, 25)
        inputBatch2 = self.avgpool2d(inputBatch2)
        inputBatch2 = inputBatch2.reshape(batchsize, -1)
        inputBatch2 = self.relu(inputBatch2)
        inputBatch2 = self.norm(inputBatch2)
        inputBatch2 = self.dropout(inputBatch2)

        #
        inputBatch2 = self.linear4(inputBatch2)
        inputBatch2 = self.relu(inputBatch2)
        inputBatch2 = self.norm2(inputBatch2)
        inputBatch2 = self.dropout(inputBatch2)
        inputBatch2 = self.linear3(inputBatch2)

        return inputBatch2,outputBatch
