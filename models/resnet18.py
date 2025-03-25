import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class ArcFaceResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ArcFaceResNet18, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改输入通道数
        # 移除模型的最后一层全连接层
        self.backbone.fc = nn.Identity()
        self.arcface = ArcFaceLoss(in_features=512, out_features=num_classes)

    def forward(self, x, label):
        x = self.backbone(x)
        x = self.arcface(x, label)
        return x


print(models.resnet18(pretrained=False))