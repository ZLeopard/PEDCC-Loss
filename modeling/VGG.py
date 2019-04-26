import torch
import torch.nn as nn
from .metric_loss import CosineLinear_PEDCC
# Set the parameter following L-Softmax Paper
class VGG(nn.Module):
    def __init__(self, cfg, is_fn = False):
        super(VGG, self).__init__()
        self.is_fn = is_fn
        self.conv_0 = nn.Sequential(
            nn.Conv2d(3, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.PReLU()
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.PReLU(),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.PReLU(),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.PReLU(),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(96, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.PReLU(),
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.PReLU(),
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.PReLU(),
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(192, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.PReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.PReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.PReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(6144, 512, bias=False),
            nn.BatchNorm1d(512, affine=False),
        )
        if self.is_fn:
            pass
            # self.out = ArcfaceLinear(512, 100, s=7.5, m=0.35)
        else:
            self.out = CosineLinear_PEDCC(512, cfg.MODEL.NUM_CLASSES, is_pedcc=cfg.METRIC.IS_PEDCC)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.l2_norm(x)
        out = self.out(x)
        return out, x
