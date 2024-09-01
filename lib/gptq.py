import math
import time

import torch
import torch.nn as nn
import transformers

from .quantize import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.groups = []

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
            self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
        ):
        # 复制权重数据
        W = self.layer.weight.data.clone()
        # 如果层是Conv2d，将权重展平
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        # 如果层是Conv1D，转置权重
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        # 转换权重数据类型为float
        W = W.float()

        # 如果量化器未准备好，找到量化参数
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        # 获取H矩阵并删除原始引用
        H = self.H
        del self.H
        # 标记H矩阵对角线为0的元素
        dead = torch.diag(H) == 0
        # 修正H矩阵对角线为0的元素
        H[dead, dead] = 1
        # 将W矩阵中对应的列置0
        W[:, dead] = 0

        # 如果启用静态分组
        if static_groups:
            import copy
            # 按组大小分割权重，为每组找到量化参数
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                self.groups.append(quantizer)

        # 如果启用激活顺序
        if actorder:
            # 根据H矩阵对角线排序
            perm = torch.argsort(torch.diag(H), descending=True)
            # 重新排序W和H矩阵
            W = W[:, perm]
            H = H[perm][:, perm]
            # 获取逆排序索引
            invperm = torch.argsort(perm)

        # 初始化损失和量化后的权重矩阵
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        # 计算阻尼系数
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        # 对H矩阵对角线元素增加阻尼
        H[diag, diag] += damp
        # 计算H的Cholesky分解和逆
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # 按块处理权重矩阵
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            # 复制当前块的权重
            W1 = W[:, i1:i2].clone()
            # 初始化当前块的量化权重和误差
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            # 获取当前块的H逆矩阵
            Hinv1 = Hinv[i1:i2, i1:i2]

            # 对当前块的每一列进行处理
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # 如果启用分组，为每组找到量化参数
                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = self.groups[idx // groupsize]

                # 量化当前列
                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                Q1[:, i] = q
                # 计算损失
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                # 计算误差并更新W1
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            # 更新量化权重和损失
            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            # 更新W矩阵
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            # 如果启用调试模式，打印损失
            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        # 等待CUDA操作完成
        torch.cuda.synchronize()
        # 如果启用激活顺序，重新排序量化权重
        if actorder:
            Q = Q[:, invperm]

        # 如果层是Conv1D，转置量化权重
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        # 更新层的权重
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # 如果启用调试模式，打印最终损失
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
