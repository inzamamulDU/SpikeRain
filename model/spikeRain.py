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
from .subModules import (
    ThresholdDependentBatchNorm2d,
    TemporalFusion,
    OverlapPatchEmbed,
    DownSampling,
    UpSampling,
)


       
class MultiDimensionalAttention(nn.Module):
    def __init__(self, T, C, reduction_t=4, reduction_c=16, kernel_size=3):
        super().__init__()
        self.T = T
        self.C = C

        # Temporal attention (T ? 1)
        self.temporal_fc = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # Pool over (T, H, W)
            nn.Conv3d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Channel attention (C ? 1)
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(C, C // reduction_c, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(C // reduction_c, C, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1, kernel_size, kernel_size), padding=(0, kernel_size // 2, kernel_size // 2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        x_perm = x.permute(1, 2, 0, 3, 4).contiguous()  # [B, C, T, H, W]

        # Temporal attention (compress T)
        temp_att = self.temporal_fc(x_perm.mean(1, keepdim=True))  # [B, 1, T, 1, 1]
        x_temp = x_perm * temp_att  # broadcasting over C

        # Channel attention
        chn_att = self.channel_fc(x_temp)  # [B, C, 1, 1, 1]
        x_chn = x_temp * chn_att

        # Spatial attention
        spatial_pool = x_chn.mean(1, keepdim=True)  # [B, 1, T, H, W]
        spatial_att = self.spatial_conv(spatial_pool)  # [B, 1, T, H, W]
        x_spatial = x_chn * spatial_att

        out = x_spatial.permute(2, 0, 1, 3, 4).contiguous()  # back to [T, B, C, H, W]
        return out
        
        
        

        
class ARFE(nn.Module):  # Adaptive strategy; ARFE: Adaptive Residual Feature Enhancement
    def __init__(self, channel, reduction):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1),  # light projection
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 5, padding=2),
            nn.Sigmoid()
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca_out = self.ca(x) * x
        sa_out = self.sa(x) * x

        fusion = torch.cat([x, ca_out, sa_out], dim=1)
        gate = self.gate(fusion)
        out = gate * ca_out + (1 - gate) * sa_out + x  # residual connection
        return out




        

class DSRB(nn.Module):
    def __init__(self, dim, growth_rate=24, num_layers=4):
        super(DSRB, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        
        self.in_channels = dim
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        channels = dim

        for i in range(num_layers):
            layer_i = nn.Sequential(
                LIFNode(v_threshold=0.15, backend='torch', step_mode='m', decay_input=False),
                layer.Conv2d(channels, growth_rate, kernel_size=3, padding=1, bias=False, step_mode='m'),
                ThresholdDependentBatchNorm2d(growth_rate),
            )
            self.layers.append(layer_i)
            channels += growth_rate  # because of dense concat

        # Local feature fusion (1x1 conv to compress features)
        self.lff = nn.Sequential(
            LIFNode(v_threshold=0.15, backend='torch', step_mode='m', decay_input=False),
            layer.Conv2d(channels, dim, kernel_size=1, bias=False, step_mode='m'),
        )
        self.attn = MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)

    def forward(self, x):
        inputs = [x]
        for layer_i in self.layers:
            out = layer_i(torch.cat(inputs, dim=2))  # concat along channel dim
            inputs.append(out)

        dense_out = torch.cat(inputs, dim=2)
        out = self.lff(dense_out) 
        out = self.attn(out) + x  # local residual connection
        return out




class SpikeRain(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=24,
                 en_num_blocks=[4, 4, 6, 6], de_num_blocks=[4, 4, 6, 6],
                 bias=False, T=4):
        super(SpikeRain, self).__init__()

        functional.set_backend(self, backend='torch')
        functional.set_step_mode(self, step_mode='m')

        self.T = T

        # Input Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim)

        # Encoder Level 1
        self.encoder_level1 = nn.Sequential(*[
            DSRB(dim=dim) for _ in range(en_num_blocks[0])
        ])

        # Encoder Level 2
        self.downsample_1_to_2 = DownSampling(dim)
        self.encoder_level2 = nn.Sequential(*[
            DSRB(dim=dim * 2) for _ in range(en_num_blocks[1])
        ])

        # Encoder Level 3
        self.downsample_2_to_3 = DownSampling(dim * 2)
        self.encoder_level3 = nn.Sequential(*[
            DSRB(dim=dim * 4) for _ in range(en_num_blocks[2])
        ])

        # Decoder Level 3
        self.decoder_level3 = nn.Sequential(*[
            DSRB(dim=dim * 4) for _ in range(de_num_blocks[2])
        ])

        # Decoder Level 2
        self.upsample_3_to_2 = UpSampling(dim * 4)
        self.reduce_channels_2 = nn.Sequential(
            LIFNode(v_threshold=v_th, backend='torch', step_mode='m', decay_input=False),
            layer.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias, step_mode='m'),
            ThresholdDependentBatchNorm2d(dim * 2),
        )
        self.decoder_level2 = nn.Sequential(*[
            DSRB(dim=dim * 2) for _ in range(de_num_blocks[1])
        ])

        # Decoder Level 1
        self.upsample_2_to_1 = UpSampling(dim * 2)
        self.decoder_level1 = nn.Sequential(*[
            DSRB(dim=dim * 2) for _ in range(de_num_blocks[0])
        ])

        # Temporal Fusion Module
        self.temporal_fusion = TemporalFusion(T=self.T)

        # Feature Refinement and Output
        self.refinement = ARFE(channel=dim * 2, reduction=8)
        self.output_conv = nn.Conv2d(dim * 2, out_channels,
                                     kernel_size=3, stride=1, padding=1)

    def forward(self, input_image):
        residual_image = input_image.clone()

        # Expand temporal dimension if necessary
        if input_image.dim() < 5:
            input_image = input_image.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        # ----- Encoding Path -----
        enc_lvl1_inp = self.patch_embed(input_image)
        enc_lvl1_out = self.encoder_level1(enc_lvl1_inp)

        enc_lvl2_inp = self.downsample_1_to_2(enc_lvl1_out)
        enc_lvl2_out = self.encoder_level2(enc_lvl2_inp)

        enc_lvl3_inp = self.downsample_2_to_3(enc_lvl2_out)
        enc_lvl3_out = self.encoder_level3(enc_lvl3_inp)

        # ----- Decoding Path -----
        dec_lvl3_out = self.decoder_level3(enc_lvl3_out)

        dec_lvl2_inp = self.upsample_3_to_2(dec_lvl3_out)
        dec_lvl2_inp = torch.cat([dec_lvl2_inp, enc_lvl2_out], dim=2)
        dec_lvl2_inp = self.reduce_channels_2(dec_lvl2_inp)
        dec_lvl2_out = self.decoder_level2(dec_lvl2_inp)

        dec_lvl1_inp = self.upsample_2_to_1(dec_lvl2_out)
        dec_lvl1_inp = torch.cat([dec_lvl1_inp, enc_lvl1_out], dim=2)
        dec_lvl1_out = self.decoder_level1(dec_lvl1_inp)

        # ----- Temporal Fusion -----
        fused_features = self.temporal_fusion(dec_lvl1_out)

        # ----- Feature Refinement and Reconstruction -----
        refined_features = self.refinement(fused_features)
        reconstructed_image = self.output_conv(refined_features)

        # Residual connection with the original input
        final_output = reconstructed_image + residual_image

        return final_output





class SpikeRainFactory:
    """
    Factory to instantiate different SpikeRain model variants: Small (S), Medium (M), and Large (L).
    """
    def __init__(self, T=4, device='cuda'):
        self.T = T
        self.device = device

    def get_model(self, variant='M'):
        variant = variant.upper()
        if variant == 'S':
            return self.spikerain_s()
        elif variant == 'M':
            return self.spikerain_m()
        elif variant == 'L':
            return self.spikerain_l()
        else:
            raise ValueError(f"Invalid variant '{variant}'. Choose from 'S', 'M', or 'L'.")

    def spikerain_s(self):
        return SpikeRain(
            dim=32,
            en_num_blocks=[2, 2, 4, 4],
            de_num_blocks=[1, 1, 1, 1],
            T=self.T
        ).to(self.device)

    def spikerain_m(self):
        return SpikeRain(
            dim=48,
            en_num_blocks=[4, 4, 8, 8],
            de_num_blocks=[2, 2, 2, 2],
            T=self.T
        ).to(self.device)

    def spikerain_l(self):
        return SpikeRain(
            dim=64,
            en_num_blocks=[6, 6, 8, 10],
            de_num_blocks=[4, 4, 4, 4],
            T=self.T
        ).to(self.device)

