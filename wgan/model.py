from torch import nn
import torch.nn.init as nninit

def avg_pool2d(x):
    return (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2]) / 4

class GeneratorBlock(nn.Module):
    def __init__(self, in_chans, out_chans, upsample=False):
        super().__init__()

        self.upsample = upsample
        self.bn1 = nn.BatchNorm2d(in_chans)
        self.conv1 = nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_chans)
        self.conv2 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, *inputs):
        x = inputs[0]

        if self.upsample:
            shortcut = nn.functional.upsample(x, scale_factor=2, mode='nearest')
        else:
            shortcut = x

        x = self.bn1(x)
        x = nn.functional.relu(x, inplace=False)
        if self.upsample:
            x = nn.functional.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.bn2(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.conv2(x)

        return x + shortcut

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        feats = 128
        self.input_linear = nn.Linear(128, 4*4*feats)
        self.block1 = GeneratorBlock(feats, feats, upsample=True)
        self.block2 = GeneratorBlock(feats, feats, upsample=True)
        self.block3 = GeneratorBlock(feats, feats, upsample=True)
        self.output_bn = nn.BatchNorm2d(feats)
        self.output_conv = nn.Conv2d(feats, 3, kernel_size=3, padding=1)

        relu_gain = nninit.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                gain = relu_gain if module != self.input_linear else 1.0
                nninit.xavier_uniform(module.weight.data, gain=gain)
                module.bias.data.zero_()

        self.last_output = None

    def forward(self, *inputs):
        x = inputs[0]

        x = self.input_linear(x)
        x = x.view(-1, 128, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output_bn(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.output_conv(x)
        x = nn.functional.tanh(x)

        self.last_output = x

        return x

class DiscriminatorFirstBlock(nn.Module):
    def __init__(self, in_chans, out_chans, downsample=False):
        super().__init__()

        self.downsample = downsample
        self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, *inputs):
        x = inputs[0]

        if self.downsample:
            shortcut = avg_pool2d(x)
        else:
            shortcut = x
        shortcut = self.shortcut_conv(shortcut)

        x = self.conv1(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.conv2(x)
        if self.downsample:
            x = avg_pool2d(x)

        return x + shortcut

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_chans, out_chans, downsample=False):
        super().__init__()

        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, *inputs):
        x = inputs[0]

        if self.downsample:
            shortcut = avg_pool2d(x)
        else:
            shortcut = x

        x = nn.functional.relu(x, inplace=False)
        x = self.conv1(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.conv2(x)
        if self.downsample:
            x = avg_pool2d(x)

        return x + shortcut

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        feats = 128
        self.block1 = DiscriminatorFirstBlock(3, feats, downsample=True)
        self.block2 = DiscriminatorBlock(feats, feats, downsample=True)
        self.block3 = DiscriminatorBlock(feats, feats, downsample=False)
        self.block4 = DiscriminatorBlock(feats, feats, downsample=False)
        self.output_linear = nn.Linear(128, 1)

        relu_gain = nninit.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                gain = relu_gain if module != self.block1.conv1 else 1.0
                nninit.xavier_uniform(module.weight.data, gain=gain)
                module.bias.data.zero_()

    def forward(self, *inputs):
        x = inputs[0]

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.functional.relu(x, inplace=False)
        x = x.mean(-1, keepdim=False).mean(-1, keepdim=False)
        x = x.view(-1, 128)
        x = self.output_linear(x)

        return x
