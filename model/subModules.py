from thop import profile
from spikingjelly.activation_based.neuron import (
    LIFNode, IFNode, ParametricLIFNode,
)
from spikingjelly.activation_based import neuron, functional, layer, surrogate
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

v_th = 0.15

alpha = 1 / (2 ** 0.5)




class ThresholdDependentBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            # Reshape only if it's 5D
            return functional.seq_to_ann_forward(x, self.bn)
        elif x.dim() == 4:
            return self.bn(x)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
            
            
            
class TemporalFusion(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(T) / T)  # initialize equal weighting

    def forward(self, x):  # x: [T, B, C, H, W]
        weights = F.softmax(self.weights, dim=0)[:, None, None, None, None]
        fused = (x * weights).sum(0)  # [B, C, H, W]
        return fused




class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=32, spike_mode="lif", LayerNorm_type='WithBias', bias=False):
        super(OverlapPatchEmbed, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        self.proj = layer.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x
        
        
        
        
class DownSampling(nn.Module):
    def __init__(self, dim):
        super(DownSampling, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        self.maxpool_conv = nn.Sequential(
            LIFNode(v_threshold=v_th, backend='torch', step_mode='m', decay_input=False),
            layer.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, step_mode='m', bias=False),
            ThresholdDependentBatchNorm2d(dim*2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class UpSampling(nn.Module):
    def __init__(self, dim):
        super(UpSampling, self).__init__()
        self.scale_factor = 2
        self.up = nn.Sequential(
            LIFNode(v_threshold=v_th, backend='torch', step_mode='m', decay_input=False),
            layer.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, step_mode='m', bias=False),
            ThresholdDependentBatchNorm2d(dim//2),
        )

    def forward(self, input):
        temp = torch.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3] * self.scale_factor,
                            input.shape[4] * self.scale_factor)).cuda()
        # print(temp.device,'-----')
        output = []
        for i in range(input.shape[0]):
            # temp[i] = self.up(input[i])
            # print(input[i].shape)
            temp[i] = F.interpolate(input[i], scale_factor=self.scale_factor, mode='bilinear')
            # print(temp.shape)
            output.append(temp[i])
        out = torch.stack(output, dim=0)
        return self.up(out)