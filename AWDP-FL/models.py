import torch
from torchvision import models
import torch.nn as nn
import math
import torch.nn.functional as F


def get_model(name="resnet18", pretrained=True):
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif name == "CIFAR_CNN":
        model = CIFAR_CNN()
    elif name == "CIFAR_CNN_P":
        model = CIFAR_CNN_P()
    elif name == "MNIST_CNN":
        model = MNIST_CNN()
    elif name == "FashionMNIST_CNN":
        model = FashionMNIST_CNN()
    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model


# MNIST_CNN
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return x

# FashionMNIST_CNN
class FashionMNIST_CNN(nn.Module):
    def __init__(self):
        super(FashionMNIST_CNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return x

# CIFAR_CNN
class CIFAR_CNN(nn.Module):
    def __init__(self, in_features=3, num_classes=10, dim=1600):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 64, kernel_size=5,
                      padding=2, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 8 * 8, 384),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(384, 192)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out

# CIFAR_CNN_P
class CIFAR_CNN_P(nn.Module):
    def __init__(self, in_features=3, num_classes=10):
        super(CIFAR_CNN_P, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 64, kernel_size=5,
                      padding=2, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3), 
            nn.Conv2d(64, 64, kernel_size=5, padding=2,
                      stride=1, bias=True), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=5, padding=2,
                      stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128 * 8 * 8, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) 
        )

        self.fc3 = nn.Linear(192, num_classes) 

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1) 
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

class LeNet(nn.Module):
    def __init__(self, in_channels=1):  # 增加 in_channels 参数
        super(LeNet, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=5//2, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(32, 32, kernel_size=5, padding=5//2, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(32, 64, kernel_size=5, padding=5//2, stride=1),
            nn.Sigmoid(),
        )
        self.fc = None  # 全连接层将在 forward 中动态定义

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)  # 展平卷积层输出
        # 如果全连接层尚未定义，则根据输入大小动态定义
        if self.fc is None:
            self.fc = nn.Linear(out.size(1), 100)
        out = self.fc(out)
        return out