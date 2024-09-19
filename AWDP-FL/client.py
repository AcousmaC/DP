import math
import datetime
import math
import numpy as np
import torch.nn as nn
import os
import random
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import models
import torch
import copy
from torch.optim.optimizer import Optimizer


# 自定义动量更新
class ThirdOrderMomentumOptimizer(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamWithThirdMoment, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_cu'] = torch.zeros_like(p.data)
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # 获取当前参数的一阶和三阶矩估计
                exp_avg, exp_avg_cu = state['exp_avg'], state['exp_avg_cu']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_cu.mul_(beta2).addcmul_(
                    grad.mul_(grad), grad, value=1 - beta2)
                
                # 计算偏置校正系数
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_cu = exp_avg_cu / bias_correction2
                
                # 计算更新步长
                step_size = group['lr'] / (torch.pow(corrected_exp_avg_cu, 1/3) + group['eps'])
                p.data.add_(-step_size * corrected_exp_avg)
                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)
                    
        return loss



class Client(object):
    def __init__(self, conf, model, train_dataset, id=-1):
        self.conf = conf
        self.local_model = models.get_model(self.conf["model_name"])
        self.client_id = id
        self.train_dataset = train_dataset
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        train_indices = all_range[id * data_len: (id + 1) * data_len]
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=conf["batch_size"],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        )

    # 计算梯度均值
    def average_sample_grads(self, sample_grads_list):
        if not sample_grads_list:
            return {}
        total_grads = {}
        for sample_grads in sample_grads_list:
            for name, grad in sample_grads.items():
                if name not in total_grads:
                    total_grads[name] = grad.clone()
                else:
                    total_grads[name] += grad
        avg_grads = {}
        num_samples = len(sample_grads_list)
        for name, grad_sum in total_grads.items():
            avg_grads[name] = grad_sum / num_samples
        return avg_grads

    # 层级权重系数
    def cosine_similarity_per_layer(self, grads1, grads2):
        cos_sim_per_layer = {}
        for name in grads1.keys():
            dot_product = (grads1[name] * grads2[name]).sum()
            norm1 = torch.norm(grads1[name], 2)
            norm2 = torch.norm(grads2[name], 2)
            similarity = dot_product / (norm1 * norm2)
            cos_sim_per_layer[name] = similarity.item()
        return cos_sim_per_layer

    # 客户端训练
    def local_train(self, model):
        LTB = datetime.datetime.now()
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        optimizer = ThirdOrderMomentumOptimizer(
            self.local_model.parameters(),
            lr=self.conf['lr'],
            betas=(self.conf['beta1'], self.conf['beta2']),
            eps=self.conf['eps']
        )
        self.local_model.train()
        cut_grads_history = {name: []
                             for name, _ in self.local_model.named_parameters()}
        for el in range(self.conf["local_epochs"]):
            if torch.cuda.is_available():
                self.local_model = self.local_model.cuda()
            ETB = datetime.datetime.now()
            # 存储所有批次的加权平均梯度范数,外层LL
            grad_norms_history = {name: [] for name, _ in self.local_model.named_parameters()}
            avg_grad_batch_layer_history = []

            for batch_id, (data, target) in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                grad_batch_layer = []
                grad_norms_L2_layer = {name: [] for name, _ in self.local_model.named_parameters()}
                real_lay_grad_norm = {name: [] for name, _ in self.local_model.named_parameters()}
                # AWDP-FL
                for i, (x_single, y_single) in enumerate(zip(data, target)):
                    optimizer.zero_grad()
                    output = self.local_model( torch.unsqueeze(x_single.to(torch.float32), 0))
                    loss = torch.nn.functional.cross_entropy( output, torch.unsqueeze(y_single.to(torch.long), 0))
                    loss.backward()
                    # 存储当前样本信息
                    sample_grads = {}
                    for name, param in self.local_model.named_parameters():
                        grad = param.grad.data.clone()
                        grad_norm = torch.norm(grad, 2)
                        grad_norms_L2_layer[name].append(grad_norm.item())
                        sample_grads[name] = grad
                    grad_batch_layer.append(sample_grads)

                # 权重计算 (内层历史序列GB)
                avg_grad_batch_layer = self.average_sample_grads(grad_batch_layer)
                cos_sim_per_layer = {name: 1 for name in avg_grad_batch_layer.keys()}
                avg_grad_batch_layer_history.append(avg_grad_batch_layer)
                setNumber = 2
                if len(avg_grad_batch_layer_history) >= setNumber:
                    all_avg_sample_grads = self.average_sample_grads( avg_grad_batch_layer_history)
                    cos_sim_per_layer = self.cosine_similarity_per_layer( avg_grad_batch_layer, all_avg_sample_grads)

                # 根据权重系数和当前范数计算当前批次层级最终使用的梯度范数(加权平均梯度范数)
                for name, param in self.local_model.named_parameters():
                    avg_lay_grad_norm = np.mean(grad_norms_L2_layer[name])
                    max_lay_grad_norm = np.max(grad_norms_L2_layer[name])
                    real_lay_grad_norm[name] = (
                        max_lay_grad_norm - avg_lay_grad_norm) / 2 * cos_sim_per_layer[name] + avg_lay_grad_norm

                # 确定每层的裁剪阈值
                cut_grad = {name: None for name in grad_norms_history}
                for name in grad_norms_history:
                    grad_norms_history[name].append(real_lay_grad_norm[name])
                    P = 50
                    cut_grad[name] = np.percentile(grad_norms_history[name], P)

                # 梯度裁剪,单样本逐次裁剪
                sum_clipped_grads = {name: torch.zeros_like(
                    param) for name, param in self.local_model.named_parameters()}
                for sample_grads in grad_batch_layer:
                    for name, grad in sample_grads.items():
                        grad_norm = torch.norm(grad, 2)
                        clip_coef = min(1, cut_grad[name] / (grad_norm + 1e-6))
                        clipped_grad = grad * clip_coef
                        sum_clipped_grads[name] += clipped_grad

                # 梯度更新(基于裁剪后梯度的平均值更新模型参数)
                for name, param in self.local_model.named_parameters():
                    param.grad.data = sum_clipped_grads[name] / len(data)
                optimizer.step()

                # 记录每批次每层的裁剪阈值L2范数
                for name, param in self.local_model.named_parameters():
                    cut_grads_history[name].append(cut_grad[name])

            print(f"\tLocal iteration {el + 1} completed, time taken: {datetime.datetime.now() - ETB}")

        if torch.cuda.is_available():
            self.local_model = self.local_model.cuda()
        
        # 差分隐私加噪
        if self.conf['dp']:
            for name, param in self.local_model.named_parameters():
                sigma_t = (2 * self.conf['lr'] * self.conf["local_epochs"] * max(cut_grads_history[name])) / (self.conf['batch_size'])
                sigma = sigma_t * (math.sqrt(2 * math.log(1.25 / self.conf['delta'])) / self.conf['epsilon'])
                noise = torch.normal(0, sigma, size=param.shape, device=param.device)
                param.data.add_(noise)

        # 客户端上传模型参数
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])
        print(
            f"\t{'*' * 7}\tThis client's training is complete, time taken: {datetime.datetime.now() - LTB}")
        return diff
