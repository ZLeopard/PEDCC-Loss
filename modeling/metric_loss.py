from __future__ import division
import torch
import math
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
sys.path.append('.')
from utils.utils import read_pkl




# consieLinear层 实现了norm的fea与norm weight的点积计算，服务于margin based softmax loss
# 将w替换成pedcc，固定
class CosineLinear_PEDCC(nn.Module):
    def __init__(self, in_features, out_features, is_pedcc):
        super(CosineLinear_PEDCC, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        if is_pedcc:
            self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=False)
            #self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
            map_dict = read_pkl()
            tensor_empty = torch.Tensor([]).cuda()
            for label_index in range(self.out_features):
                tensor_empty = torch.cat((tensor_empty, map_dict[label_index].float().cuda()), 0)
            label_40D_tensor = tensor_empty.view(-1, self.in_features).permute(1, 0)
            label_40D_tensor = label_40D_tensor.cuda()
            self.weight.data = label_40D_tensor
        else:
            self.weight = Parameter(torch.FloatTensor(out_features, in_features))
            nn.init.xavier_uniform_(self.weight)
        #print(self.weight.data)

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2, 1, 1e-5).mul(1e5)  # weights normed
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(ww)  # size=(B,Classnum)  x.dot(ww)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)  #
        cos_theta = cos_theta.clamp(-1, 1)
        # cos_theta = cos_theta * xlen.view(-1, 1)

        return cos_theta  # size=(B,Classnum,1)

# 原始实现caffe->https://github.com/happynear/AMSoftmax
class AMSoftmax(nn.Module):
    def __init__(self, scale, margin, is_amp=False):
        super(AMSoftmax, self).__init__()
        self.scale = scale
        self.margin = margin
        self.is_amp = is_amp
    def forward(self, input, target):
        # self.it += 1
        cos_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index1 = ~index


        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= self.margin

        if self.is_amp:
            output[index1] += self.margin
        output = output * self.scale


        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt
        loss = loss.mean()

        return loss
