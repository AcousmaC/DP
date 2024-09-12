# -*- coding: utf-8 -*-
import argparse
import math
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torchvision import models, datasets, transforms
from util import dlgAttack
from models import LeNet

# SSIM值计算
def calculate_ssim(img1, img2, window_size=11, sigma=1.5, K1=0.01, K2=0.03, L=1):
    # 生成高斯核
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    # 使用高斯核计算局部均值和方差
    def filter2d(img, window, channel):
        window = window.expand(channel, 1, window_size, window_size)  # 扩展到多个通道
        return F.conv2d(img, window, padding=window_size // 2, groups=channel)

    # 检查输入图像的维度
    if img1.dim() != 4 or img2.dim() != 4:
        raise ValueError("图像张量的维度应为 4D (N, C, H, W)")

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # 获取图像的通道数
    channel = img1.size(1)

    # 为 SSIM 计算生成高斯窗口
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    # 计算局部均值
    mu1 = filter2d(img1, window, channel)
    mu2 = filter2d(img2, window, channel)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 计算局部方差和协方差
    sigma1_sq = filter2d(img1 ** 2, window, channel) - mu1_sq
    sigma2_sq = filter2d(img2 ** 2, window, channel) - mu2_sq
    sigma12 = filter2d(img1 * img2, window, channel) - mu1_mu2

    # 计算 SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# 定义不同数据集的输入通道数
def get_in_channels(dataset_name):
    if dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
        return 1  # 灰度图像
    elif dataset_name == 'CIFAR10':
        return 3  # RGB 彩色图像
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

# 定义命令行参数解析器
parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
args = parser.parse_args()
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)
print(torch.__version__, torchvision.__version__)
# 模拟样本
index = 666
dataset_name = 'MNIST'
# 判断数据集
if dataset_name == 'MNIST':
    datasets_all = datasets.MNIST("./data", download=True)
elif dataset_name == 'CIFAR10':
    datasets_all = datasets.CIFAR10("./data", download=True)
elif dataset_name == 'FashionMNIST':
    datasets_all = datasets.FashionMNIST("./data", download=True)
# 自动选择通道
in_channels = get_in_channels(dataset_name)
local_model = LeNet(in_channels=in_channels).to(device)
img_index = index
original_data = transforms.ToTensor()(datasets_all[img_index][0]).to(device)  # 从数据集中获取图像，并转换为张量格式，然后放到指定设备上

original_data = original_data.view(1, *original_data.size())  # 调整张量形状，使其适应网络的输入要求
original_label = torch.Tensor([datasets_all[img_index][1]]).long().to(device).view(1, )  # 获取图像对应的标签，并转换为张量格式，调整标签的形状
original_lonehot_label = dlgAttack.label_to_onehot(original_label)  # 将标签转换为 one-hot 编码格式，适配损失函数
torch.manual_seed(1234)

img_pil = transforms.ToPILImage()(original_data[0].cpu() )
if img_pil.mode == 'L':
    plt.imshow(img_pil, cmap='gray')
else:
    plt.imshow(img_pil)
criterion = dlgAttack.cross_entropy_for_onehot
output = local_model(original_data)
loss = criterion(output, original_lonehot_label)
grad = torch.autograd.grad(loss, local_model.parameters())
original_grad = list((_.detach().clone() for _ in grad))  # 将梯度从计算图中分离并复制

# AWWDP-FL加噪
dlgAttack = False
lr = 0.001  # 学习率
delta = 0.001  # 差分隐私参数 delta
epsilon = 0  # 差分隐私参数 epsilon
local_epochs = 3
batch_size = 512
if dlgAttack == True:
    noised_grad = []
    for i, grad in enumerate(original_grad):
        sigma_t = (2 * lr * local_epochs * torch.norm(grad, p=2)) / (batch_size)
        sigma = sigma_t * (math.sqrt(2 * math.log(1.25 / delta)) / epsilon)
        noise = torch.normal(0, sigma, size=grad.shape, device=grad.device)
        original_grad[i] += noise

# 伪装样本
dummy_data = torch.randn(original_data.size()).to(device).requires_grad_(True)  # 生成与真实数据相同尺寸的随机噪声数据，并设置为可求导
dummy_label = torch.randn(original_lonehot_label.size()).to(device).requires_grad_(True)  # 生成与真实标签相同尺寸的随机噪声标签，并设置为可求导
plt.figure()
plt.imshow(transforms.ToPILImage()(dummy_data[0].cpu()))
op_lr =0.9
optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=op_lr)
# 定义目录和文件路径
attack_file_dir = os.path.join('adaptSave', 'attackSave')
attack_file_path = os.path.join(attack_file_dir, 'attack.txt')
# 检查目录是否存在，如果不存在则创建目录
if not os.path.exists(attack_file_dir):
    os.makedirs(attack_file_dir)
# 检查文件是否存在，如果不存在则创建文件
if not os.path.exists(attack_file_path):
    with open(attack_file_path, 'w') as f:
        f.write("init txt\n")

dummy_history_data = []  # 用于记录每次迭代后的伪造数据
k =100
all_k = 3000
for iters in range(all_k):  # 进行 3000 次迭代优化,一般1000次稳定.
    def closure():
        optimizer.zero_grad()  # 清除梯度
        dummy_output = local_model(dummy_data)  # 使用伪造数据进行前向传播
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)  # 将伪造标签通过 softmax 转换为概率分布
        dummy_loss = criterion(dummy_output, dummy_onehot_label)  # 计算伪造数据的损失
        dummy_grad = torch.autograd.grad(dummy_loss, local_model.parameters(), create_graph=True)  # 计算伪造数据的梯度
        grad_diff = 0
        for dg, og in zip(dummy_grad, original_grad):
            grad_diff += ((dg - og) ** 2).sum()
        grad_diff.backward()
        return grad_diff

    optimizer.step(closure)
    if iters % k == 0:  # 每 100 次迭代输出一次当前的损失
        current_loss = closure()
        dummy_img_np = dummy_data[0].cpu().detach().numpy()
        original_img_np = original_data[0].cpu().detach().numpy()
        if dummy_img_np.ndim == 3:
            if dummy_img_np.shape[-1] == 1:
                dummy_img_np = np.expand_dims(dummy_img_np, axis=-1)
                original_img_np = np.expand_dims(original_img_np, axis=-1)
        elif dummy_img_np.ndim == 4:
            dummy_img_np = dummy_img_np.squeeze(0)
            original_img_np = original_img_np.squeeze(0)
        else:
            raise ValueError("Unsupported number of dimensions")
        dummy_img_tensor = torch.tensor(dummy_img_np).permute(2, 0, 1).unsqueeze(0).float()
        original_img_tensor = torch.tensor(original_img_np).permute(2, 0, 1).unsqueeze(0).float()
        # 计算 SSIM
        ssim_value = calculate_ssim(dummy_img_tensor, original_img_tensor)
        print(f"{dataset_name}-{index}-{epsilon}-{lr}-{op_lr}  \tT:  {iters}\t Loss:  {current_loss.item():.4f} \t SSIM:  {ssim_value:.4f}")
        output = f"{dataset_name}-{index}-{epsilon}-{lr}-{op_lr}  \tT:  {iters}\t Loss:  {current_loss.item():.4f} \t SSIM:  {ssim_value:.4f}\n"
        with open(attack_file_path, 'a') as f:
            f.write(output)
        if dummy_data.shape[1] == 1:
            img_pil = transforms.ToPILImage()(dummy_data[0].cpu()).convert('L')
        else:
            img_pil = transforms.ToPILImage()(dummy_data[0].cpu())
        dummy_history_data.append(img_pil)

# 保存攻击后的图片
os.makedirs(attack_file_dir, exist_ok=True)
sub_dir = os.path.join(attack_file_dir, f'{dataset_name}-{index}-{epsilon}-{dlgAttack}')
os.makedirs(sub_dir, exist_ok=True)
for i, img in enumerate(dummy_history_data):
    img_path = os.path.join(sub_dir, f'image_{i * 100}.png')
    img.save(img_path)

# 保存原始图片
original_img_pil = transforms.ToPILImage()(original_data[0].cpu())
original_img_path = os.path.join(sub_dir, 'a.png')
original_img_pil.save(original_img_path)

# 汇总查看
plt.figure(figsize=(12, 8))
num_images_to_show = min(all_k // k, 30)
for i in range(num_images_to_show):
    plt.subplot(3, 10, i + 1)
    img = dummy_history_data[i]
    if img.mode == 'L':
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.title("iter = %d" % (i * k))
    plt.axis('off')
plt.show()