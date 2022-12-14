from torch import nn
import torch


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,  alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(16 * alpha, round_nearest)
        last_channel = _make_divisible(128 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [5, 32, 2, 2],
            [5, 64, 2, 2],
            [5, 128, 2, 2],
        ]

        features = []

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == '__main__':

    net = MobileNetV2()

    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = "mobilenet_v2.pth"
    checkpoint = torch.load(model_weight_path, map_location="cpu")
    #print(checkpoint.keys())
    for k in list(checkpoint.keys()):
        words2 = k.split('.')
        #print(words2)
        # retain only encoder_q up to before the embedding layer
        if k.startswith('features.0'):
            # remove prefix
            del checkpoint[k]
        elif k.startswith('classifier'):
            print(k)
            del checkpoint[k]
        else:
            t_words2 = str(int(words2[1]) - 1)
            print(words2[1])
            temp_name = words2[0]+'.'+t_words2
            t_temp_name = words2[0]+'.'+words2[1]
            print(k)
            temp_name = temp_name + k[len(t_temp_name):]
            print(temp_name)
            checkpoint[temp_name] = checkpoint[k]
            del checkpoint[k]
    print(checkpoint.keys())
    torch.save(checkpoint, "D:/mobilenet_v2.pth")
    net = MobileNetV2()
    import numpy as np

    A = np.random.rand(2, 32, 112, 112)
    A = torch.from_numpy(A)
    A = A.float()

    out = net(A)
    print(out.shape)
    '''
    net = MobileNetV2(num_classes=5)

    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = "./mobilenet_v2_small.pth"

    pre_weights = torch.load(model_weight_path, map_location='cpu')

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = False

    net = MobileNetV2(num_classes=128)
    import numpy as np

    A = np.random.rand(2, 64, 112, 112)
    A = torch.from_numpy(A)
    A = A.float()

    out = net(A)
    print(out.shape)

    '''
