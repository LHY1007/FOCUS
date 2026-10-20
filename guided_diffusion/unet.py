"""FOCUS conditional diffusion backbone.

Methods 1.3-1.6 are implemented by the live call chain in
UNetModel._forward_manuscript, guided_diffusion.manuscript and
 guided_diffusion.morphology. See docs/reproduction.md for equation mapping.
"""

from abc import abstractmethod

import torch
import math

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from guided_diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
from guided_diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

import copy
import torch.nn.functional as F




    



        
class AttentionPool2d(nn.Module):

    def __init__(
            self,
            spacial_dim: int,
            embed_dim: int,
            num_heads_channels: int,
            output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):


    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):

    def forward(self, x, emb):

        
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class SE_Attention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False),
                                nn.Sigmoid())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Upsample(nn.Module):

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):

        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):

    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):

        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class GMSD_loss(nn.Module):
    def __init__(self, c=170, device='cuda', noise_std=0.01,num_channels=128):
        super(GMSD_loss, self).__init__()
        self.c = c
        self.device = device
        self.noise_std = noise_std
        self.num_channels = num_channels

        # Sobel算子用于梯度计算
        self.hx = torch.tensor([[1/3, 0, -1/3]]*3, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        self.hx = self.hx.expand(num_channels, 1, 3, 3).to(self.device)
        self.hy = self.hx.transpose(2, 3).to(self.device)

        # 均值滤波核
        self.ave_filter = torch.tensor([[0.25, 0.25], [0.25, 0.25]], dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)
        self.ave_filter = self.ave_filter.expand(num_channels, 1, 2, 2).to(self.device)

    def forward(self, dis_img, ref_img):
        """
        计算输入图像的GMSD loss, 并在特征映射中引入超球空间的增强。
        dis_img: 失真图像 [B, C, H, W]
        ref_img: 参考图像 [B, C, H, W]
        """
        if torch.max(dis_img) <= 1:
            dis_img = dis_img * 255
        if torch.max(ref_img) <= 1:
            ref_img = ref_img * 255

        dis_img = dis_img.float()
        ref_img = ref_img.float()

        # 均值滤波
        ave_dis = F.conv2d(dis_img, self.ave_filter, stride=1, groups=self.num_channels)
        ave_ref = F.conv2d(ref_img, self.ave_filter, stride=1, groups=self.num_channels)

        # 下采样
        down_step = 2
        ave_dis_down = ave_dis[:, :, 0::down_step, 0::down_step]
        ave_ref_down = ave_ref[:, :, 0::down_step, 0::down_step]

        # 超球空间映射
        ave_dis_down = self._map_to_hypersphere(ave_dis_down)
        ave_ref_down = self._map_to_hypersphere(ave_ref_down)

        # 梯度计算
        mr_sq = F.conv2d(ave_ref_down, self.hx, groups=self.num_channels)**2 + F.conv2d(ave_ref_down, self.hy, groups=self.num_channels)**2
        md_sq = F.conv2d(ave_dis_down, self.hx, groups=self.num_channels)**2 + F.conv2d(ave_dis_down, self.hy, groups=self.num_channels)**2
        mr = torch.sqrt(mr_sq)
        md = torch.sqrt(md_sq)

        # GMSD计算
        GMS = (2 * mr * md + self.c) / (mr_sq + md_sq + self.c)
        GMSD = torch.std(GMS.view(-1))
        return GMSD

    def _map_to_hypersphere(self, x):
        """
        将输入特征映射到超球空间并添加噪声。
        x: 输入特征 [B, C, H, W]
        """
        # 超球空间
        x_normalized = x / x.norm(dim=1, keepdim=True)  # [B, C, H, W]

        # 小范围噪声
        noise = torch.randn_like(x_normalized) * self.noise_std
        x_noisy = x_normalized + noise

        # 再次归一化
        x_noisy_normalized = x_noisy / x_noisy.norm(dim=1, keepdim=True)
        return x_noisy_normalized
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialRegionAttention(nn.Module):
    """
    Spatial Attention Module for weighting specific regions in the image.
    """
    def __init__(self):
        super(SpatialRegionAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)  # 7x7 conv
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute spatial attention weights
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)  # [batch, 2, H, W]
        attn = self.conv1(attn)  # [batch, 1, H, W]
        return self.sigmoid(attn) * x  # Apply attention weights to input


class SpatialGeneCorrelationWeighting(nn.Module):
    """
    Module to adjust ST gene features based on a gene relationship matrix.
    """
    def __init__(self, num_genes, gene_relation_matrix=None):
        super(SpatialGeneCorrelationWeighting, self).__init__()
        gene_relation_matrix = torch.tensor(gene_relation_matrix, dtype=torch.float32)

        self.gene_relation_matrix = nn.Parameter(gene_relation_matrix)

    def forward(self, gene_features):
        """
        Args:
            gene_features: [batch, num_genes, H, W] tensor
        Returns:
            Weighted gene features
        """
        # Reshape gene features for matrix multiplication
        batch, num_genes, H, W = gene_features.shape
        gene_features = gene_features.view(batch, num_genes, -1)  # [batch, num_genes, H*W]
        # Apply relationship matrix
        weighted_features = torch.matmul(self.gene_relation_matrix, gene_features)  # [batch, num_genes, H*W]
        return weighted_features.view(batch, num_genes, H, W)



class GeneNamePromptProcessor(nn.Module):
    def __init__(self, out_channels=20, spatial_size=64):
        """
        处理 gene_name_features：
         - 输入：tensor shape [batch, 20, 768]
         - 输出：tensor shape [batch, 20, spatial_size, spatial_size]
        """
        super(GeneNamePromptProcessor, self).__init__()
        self.out_channels = out_channels  # 应与输入的第二个维度一致（20）
        self.spatial_size = spatial_size
        # 将每个基因的 768 维映射到 64x64 的平面（4096 个元素）
        self.fc = nn.Linear(768, 256)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d1 = nn.Conv2d(20, 20, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d2 = nn.Conv2d(20, 20, 3, padding=1)
        

    def forward(self, gene_name_features):
        # gene_name_features: [batch, 20, 768]
        gene_name_features = gene_name_features.to(next(self.parameters()).device)
        x = self.fc(gene_name_features)  # shape: [batch, 20, 4096]
        x = x.view(-1, self.out_channels, self.spatial_size//4, self.spatial_size//4)  # shape: [batch, 20, 64, 64]
        x = self.upsample1(x)  # shape: [batch, 20, 8192]
        x = self.conv2d1(x)  # shape: [batch, 20, 8192]
        x = self.upsample2(x)
        x = self.conv2d2(x)
        return x

class MetadataPromptProcessor(nn.Module):
    def __init__(self, out_channels=20, spatial_size=64):
        super(MetadataPromptProcessor, self).__init__()
        self.out_channels = out_channels
        self.spatial_size = spatial_size

        # self.fc = nn.Linear(768, out_channels * spatial_size * spatial_size)
        self.fc = nn.Linear(768, 768)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d1 = nn.Conv2d(3, 20, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d2 = nn.Conv2d(20, 20, 3, padding=1)
    def forward(self, metadata_feature, batch_size):
        # 如果是 1D 向量，则加一维
        if metadata_feature.dim() == 1:
            metadata_feature = metadata_feature.unsqueeze(0)  # shape: [1, 768]
        # 确保输入数据在正确设备上
        device = next(self.parameters()).device
        metadata_feature = metadata_feature.to(device)
        # 映射到 [batch, out_channels*spatial_size*spatial_size]
        x = self.fc(metadata_feature)  # 假设 shape: 
        # 重塑到 [batch, out_channels, spatial_size, spatial_size]
        x = x.view(-1, 3, self.spatial_size//4, self.spatial_size//4)  # shape: [batch, 20, 16, 16]
        x = self.upsample1(x)  # shape: [batch, 20, 32, 32]
        x = self.conv2d1(x) 
        x = self.upsample2(x) # shape: [batch, 20, 64, 64]
        x = self.conv2d2(x)
        return x
class SpatialMatchingLayer(nn.Module):
    def __init__(self, in_ch, out_ch, d=128, depth=2, nheads=4, mlp_ratio=4.0):
        super().__init__()
        self.in_proj  = nn.Conv2d(in_ch, d, kernel_size=1, bias=False)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nheads, dim_feedforward=int(d*mlp_ratio),
            activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.out_proj = nn.Conv2d(d, out_ch, kernel_size=1, bias=False)

    def forward(self, x):                # x: [B, Cin, H, W]
        B, _, H, W = x.shape
        x = self.in_proj(x)              # [B, d, H, W]
        tokens = x.flatten(2).transpose(1, 2)   # [B, HW, d]
        tokens = self.encoder(tokens)           # [B, HW, d]
        x = tokens.transpose(1, 2).reshape(B, -1, H, W)  # [B, d, H, W]
        x = self.out_proj(x)             # [B, Cout, H, W]
        return x
    
class UNetModel(nn.Module):


    def __init__(
            self,
            gene_num,
            model_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,#是否使用残差块进行上采样和下采样。
            use_new_attention_order=False,#是否使用新型注意力模式
            root='',
            args=None,
            manuscript_mode=True,
            learn_sigma=True,
            he_feature_dim=1536,
            st_feature_dim=512,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if not manuscript_mode:
            raise ValueError("This model requires manuscript_mode=true and a compatible checkpoint")
        self.manuscript_mode = True
        self.gene_num = gene_num #基因数量，用于定义输入和输出通道数。
        # self.in_channels = in_channels
        self.model_channels = model_channels
        # self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions #指定在何种下采样率下使用注意力机制
        self.dropout = dropout
        self.channel_mult = channel_mult #通道数乘子，控制不同层的通道数变化
        self.conv_resample = conv_resample#是否使用卷积进行上采样和下采样。
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample


        

        time_embed_dim = model_channels * 4 #时间嵌入层用于处理时间步信息，生成一个时间嵌入向量，该向量随后会被注入到网络中的多个位置。
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.pre = nn.Sequential(
            conv_nd(dims, model_channels, self.gene_num, 3, padding=1),
            nn.SiLU()
        )
        self.post = nn.Sequential(
            conv_nd(dims, self.gene_num, model_channels, 3, padding=1),
            nn.SiLU()
        )

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, self.gene_num, ch, 3, padding=1))]
        )

        self.input_blocks_WSI5120 = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, 3, ch, 3, padding=1))]
        )

        self.input_blocks_WSI320 = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, 3, ch, 3, padding=1))]
        )


        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch,time_embed_dim,dropout,out_channels=int(mult * model_channels),dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,)]
                ch = int(mult * model_channels)
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_blocks_WSI5120.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch,time_embed_dim,dropout,out_channels=out_ch,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,down=True,)
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                )
                self.input_blocks_WSI5120.append(
                    TimestepEmbedSequential(ResBlock(ch,time_embed_dim,dropout,out_channels=out_ch,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,down=True,)
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch,time_embed_dim,dropout,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich,time_embed_dim,dropout,out_channels=int(model_channels * mult),dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,)]
                ch = int(model_channels * mult)
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResBlock(ch,time_embed_dim,dropout,out_channels=out_ch,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,up=True,)
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        conv_ch = self.channel_mult[-1] * self.model_channels

        self.input_blocks_lr = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.out = nn.Sequential(
            normalization(ch), nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, self.gene_num * (2 if learn_sigma else 1), 3, padding=1)),
        )
        self.MemoryNetwork = SpatialGeneMemoryBank(gene_dim=64, num_genes=self.gene_num, query_dim=self.gene_num)
        self.MemoryNetwork.sparse_attention.sparsity = 0.15
        self.he_attention = HEGuidedRegionalAttention(scales=[1.0, 2.0, 4.0], gamma=2.0, kappa=0.5, alpha=1.0)
        from .manuscript import ManuscriptConditioning
        self.manuscript = ManuscriptConditioning(model_channels, gene_num, he_feature_dim, st_feature_dim)
        self.meta_processor = MetadataPromptProcessor(out_channels=20, spatial_size=64)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        
    def _forward_manuscript(self, x, timesteps, low_res, he, cellular, gene_class, gene_map,
                            prompt, coexpression, masks, reference, scgpt, pre_he, complexity=None, original_st_bin_um=None):
        from .manuscript import reference_projection, cell_consistency, resolution_gate
        core = self.manuscript
        # These derived conditions depend only on the current input tensors.
        # Retain their owners to prevent pointer reuse across different images.
        owners = (low_res, he, masks, reference)
        cache = None
        if not torch.is_grad_enabled():
            if not hasattr(self, '_condition_cache') or any(
                    a is not b for a, b in zip(self._condition_cache[0], owners)):
                self._condition_cache = (owners, {})
            cache = self._condition_cache[1]
        x = x[:, self.gene_num:]
        he = he.to(x) / 255
        if reference is None:
            raise ValueError('A tissue-matched, gene-aligned scRNA reference is required')
        scores, active = core.task_scores(low_res, he, reference, masks)
        self.task_activation_scores = scores.detach()
        self.task_activation_mask = active.detach()
        self.resolution_activation = resolution_gate(original_st_bin_um)
        rectified = low_res.clone()
        idx = torch.where(active[:, 0])[0]
        if len(idx):
            key = ('ot', tuple(idx.tolist()))
            if cache is not None and key in cache:
                projected = cache[key]
            else:
                projected = reference_projection(reference[idx], low_res[idx])
                if cache is not None:
                    cache[key] = projected
            rectified[idx] = projected
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        hx, hspt, hhe, skips = x, rectified, he, []
        for denoise, spatial, histology in zip(self.input_blocks, self.input_blocks_lr, self.input_blocks_WSI5120):
            hx, hspt, hhe = denoise(hx, emb), spatial(hspt, emb), histology(hhe, emb)
            skips.append((hx + F.interpolate(hspt, size=hx.shape[-2:], mode='bilinear', align_corners=False) + hhe) / 3)
        size = hx.shape[-2:]
        hspt = F.interpolate(hspt, size=size, mode='bilinear', align_corners=False)
        if pre_he.ndim != 3 or pre_he.shape[1:] != (257, core.he_match.in_features):
            raise ValueError(f'Expected authentic Prov-GigaPath tile features [B,257,{core.he_match.in_features}]')
        hhe = hhe + core.he_match(pre_he[:, 0])[:, :, None, None]
        hx = hx + F.interpolate(core.st_match(scgpt), size=size, mode='bilinear', align_corners=False)
        features = [torch.zeros_like(hx) for _ in range(3)]
        attention = he.new_zeros((len(x), 1, *he.shape[-2:]))
        idx = torch.where(active[:, 0])[0]
        if len(idx):
            memory_parts = []
            for i in idx:
                bank = coexpression[i]
                if bank.shape[0] != self.gene_num:
                    positions = gene_class[i].long()
                    if (positions < 0).any() or (positions >= bank.shape[0]).any() or len(positions.unique()) != self.gene_num:
                        raise ValueError('Gene indices do not uniquely identify rows of the co-expression matrix')
                    bank = bank[positions]
                memory_parts.append(self.MemoryNetwork(self.pre(hspt[i:i+1]), memory_bank=bank))
            memory = torch.cat(memory_parts)
            enhanced = hspt[idx] + self.post(memory)
            key = ('attention', tuple(idx.tolist()))
            if cache is not None and key in cache:
                weights = cache[key]
                enhanced = self.he_attention.apply_feature_gating(enhanced, weights)
            else:
                enhanced, weights = self.he_attention(enhanced, he[idx], return_weights=True)
                if cache is not None:
                    cache[key] = weights
            features[0][idx], attention[idx] = enhanced, weights
        structural = hx.flatten(1).sum(1) * 0
        consistency = structural.clone()
        shared_he = shared_st = unique_he = unique_st = hx[:, :self.model_channels // 2] * 0
        idx = torch.where(active[:, 2])[0]
        fused_he = hhe.clone()
        if len(idx):
            if complexity is None or complexity.shape != (len(x), 16, 16):
                raise ValueError('Expected native morphology complexity [B,16,16] from registered H&E and native cell labels')
            grid = complexity[idx]
            tiles = cellular[idx].to(x) / 255
            if tiles.shape[1] != grid.shape[1] * grid.shape[2]:
                raise ValueError('Cellular H&E tiles must follow the row-major complexity grid')
            tile_emb = emb[idx].repeat_interleave(tiles.shape[1], dim=0)
            local = tiles.flatten(0, 1)
            for module in self.input_blocks_WSI320:
                local = module(local, tile_emb)
            local = local.reshape(len(idx), tiles.shape[1], *local.shape[1:])
            local = local + core.he_match(pre_he[idx, 1:])[:, :, :, None, None]
            fused_he[idx] = hhe[idx] + core.cellular_fusion(hhe[idx], local, grid)
            features[2][idx] = fused_he[idx]
            consistency[idx] = cell_consistency(hhe[idx], hx[idx], masks[idx], grid)
        idx = torch.where(active[:, 1])[0]
        if len(idx):
            aligned, loss, decomposed = core.alignment(hspt[idx], fused_he[idx], masks[idx])
            features[1][idx], structural[idx] = aligned, loss
            shared_he, shared_st, unique_he, unique_st = [v.clone() for v in (shared_he, shared_st, unique_he, unique_st)]
            for destination, source in zip((shared_he, shared_st, unique_he, unique_st), decomposed):
                destination[idx] = source
        prompt_map = self.meta_processor(prompt, len(x))
        prompt_map = F.interpolate(prompt_map, size=size, mode='bilinear', align_corners=False)
        gene_map = F.interpolate(gene_map, size=size, mode='nearest')
        h = core.fuse(torch.cat([hx, hhe, prompt_map, gene_map], 1))
        h = h + core.coordinator(features, active, prompt, scores)
        # Native ST bins coarser than the cellular target activate an additional
        # denoiser residual in the shared backbone. At 10 um or finer this term
        # is exactly zero, preserving the trained high-resolution path.
        h = h + 0.1 * self.resolution_activation.to(h.dtype)[:, None, None, None] * hx
        scale = h.new_ones(len(h))
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = module(torch.cat([h, skips.pop()], 1), emb)
        zero = structural * 0
        return shared_he, shared_st, unique_he, unique_st, structural, attention, consistency, zero, scale, self.out(h)

    def forward(self, x, timesteps, low_res, WSI_5120, WSI_320, gene_class,
                Gene_index_map, metadata_feature, scale_gt, co_expression, WSI_mask,
                sc, scgpt, pre_he, last_HQST=None, complexity=None, original_st_bin_um=None):
        return self._forward_manuscript(x, timesteps, low_res, WSI_5120, WSI_320,
                                        gene_class, Gene_index_map, metadata_feature, co_expression,
                                        WSI_mask, sc, scgpt, pre_he, complexity, original_st_bin_um)


class SpatialGeneMemoryBank(nn.Module):
    def __init__(self, query_dim, num_genes, gene_dim):
        """
        :param query_dim: 单基因查询向量的通道数 (例如 3)
        :param num_genes: 基因记忆库中的基因数量 (例如 200)
        :param gene_dim: 每个基因的特征维度 (例如 32)
        """
        super(SpatialGeneMemoryBank, self).__init__()
        self.query_dim = query_dim
        self.num_genes = num_genes
        self.gene_dim = gene_dim

        self.WQ = nn.Linear(gene_dim, gene_dim)
        self.WK = nn.Linear(gene_dim, gene_dim)
        self.WV = nn.Linear(gene_dim, gene_dim)

        self.meta_updater = GeneMemoryUpdater(gene_dim)
        self.sparse_attention = SparseGeneMemoryAttention(sparsity=0.3)

        self.cross_attention = nn.MultiheadAttention(embed_dim=gene_dim, num_heads=8)

        self.restore_channels = nn.Linear(gene_dim, 64 * 64)

    def forward(self, single_gene_data, memory_bank):
        """
        前向传播
        :param single_gene_data: [batch, query_dim, 32, 32]
        :param memory_bank: 全局基因共表达矩阵 [num_genes, num_genes]
        :return: [batch, query_dim, 32, 32]
        """
        batch_size, _, height, width = single_gene_data.shape


        queries = self.extract_queries(single_gene_data)  # [batch, query_dim, gene_dim]


        local_memory = self.extract_local_memory(single_gene_data)  # [batch, query_dim]


        global_memory = self.extract_gene_features(memory_bank)  # [num_genes, gene_dim]
        adapted_memory = self.meta_updater.adapt_memory(global_memory, queries)  # [num_genes, gene_dim]


        sparse_memory = self.sparse_attention(queries, adapted_memory)  # [batch, query_dim, gene_dim]


        local_memory = local_memory.unsqueeze(2).expand(-1, -1, sparse_memory.size(-1))
        combined_memory = torch.cat([local_memory, sparse_memory], dim=1)  # [batch, query_dim*2, gene_dim]

        query_proj = self.WQ(queries)
        key_proj = self.WK(combined_memory)
        value_proj = self.WV(combined_memory)

        query_proj = query_proj.permute(1, 0, 2)  # [query_dim, batch, gene_dim]
        key_proj = key_proj.permute(1, 0, 2)
        value_proj = value_proj.permute(1, 0, 2)

        attn_output, _ = self.cross_attention(query_proj, key_proj, value_proj)
        attn_output = attn_output.permute(1, 0, 2)  # [batch, query_dim, gene_dim]

        restored = self.restore_channels(attn_output)
        restored = restored.view(batch_size, self.query_dim, height, width)

        return restored

    def extract_local_memory(self, single_gene_data):
        return single_gene_data.mean(dim=[2, 3])  # [batch, query_dim]

    def extract_queries(self, single_gene_data):
        batch_size, channels, height, width = single_gene_data.shape
        queries = single_gene_data.mean(dim=[2, 3])  # [batch, query_dim]
        return queries.unsqueeze(-1).expand(-1, -1, self.gene_dim)  # [batch, query_dim, gene_dim]

    def extract_gene_features(self, memory_bank):
        """
        提取基因记忆特征
        :param memory_bank: [num_genes, num_genes]
        :return: [num_genes, gene_dim]
        """
        if not isinstance(memory_bank, torch.Tensor):
            memory_bank = torch.tensor(memory_bank, dtype=torch.float32)
        memory_bank = memory_bank.float().to(next(self.parameters()).device)

        if memory_bank.dim() > 2:
            memory_bank = memory_bank[0] 
        elif memory_bank.dim() == 1:
            memory_bank = memory_bank.unsqueeze(0)


        # Rows are selected by current ST gene names in the formal loader.
        # Pooling preserves a vector of correlations for each queried gene.
        if memory_bank.shape[-1] == self.gene_dim:
            return memory_bank
        return F.adaptive_avg_pool1d(memory_bank[None], self.gene_dim).squeeze(0)


class GeneMemoryUpdater(nn.Module):
    def __init__(self, gene_dim):
        super().__init__()
        self.adaptive_layer = nn.Linear(gene_dim, gene_dim)

    def adapt_memory(self, memory_bank, query_features):
        """
        根据查询动态调整记忆库
        :param memory_bank: [num_genes, gene_dim]
        :param query_features: [batch, query_dim, gene_dim]
        :return: [num_genes, gene_dim]
        """
        device = memory_bank.device
        task_adaptation = query_features.mean(dim=0)  # [query_dim, gene_dim]
        delta = self.adaptive_layer(task_adaptation.mean(dim=0)).to(device)
        adapted_memory = memory_bank + delta
        return adapted_memory


class SparseGeneMemoryAttention(nn.Module):
    def __init__(self, sparsity=0.1):
        super().__init__()
        self.sparsity = sparsity

    def forward(self, query, memory):
        """
        :param query: [batch, query_dim, gene_dim]
        :param memory: [num_genes, gene_dim]
        :return: [batch, query_dim, gene_dim]
        """
        scores = torch.matmul(query, memory.T)  # [batch, query_dim, num_genes]
        topk = max(1, int(memory.size(0) * self.sparsity))
        topk_scores, topk_indices = torch.topk(scores, topk, dim=-1)

        batch_size, query_dim, _ = topk_scores.shape
        gene_dim = memory.size(-1)

        sparse_memory = memory[topk_indices.view(-1)].view(batch_size, query_dim, topk, gene_dim)
        sparse_weights = torch.softmax(topk_scores, dim=-1)
        weighted_memory = torch.einsum('bqt,bqtd->bqd', sparse_weights, sparse_memory)
        return weighted_memory

class SensitivityNoiseReplacer:
    def __init__(self, replacement_prob=0.8):
        self.replacement_prob = replacement_prob
    def replace_with_noise(self, feature_map, ratio, dtype=torch.float32):
        if ratio == 2.0:
            ratio = 0.2
        elif ratio <0.5:
            ratio = 0.0
        else:
            ratio = ratio / 4.0

        B, C, H, W = feature_map.shape
        
        replaced_feature = feature_map.clone()

        for b in range(B):
            if torch.rand(1, device=feature_map.device).item() < ratio:
                half_channels = C // 2
                selected_channels = torch.randperm(C, device=feature_map.device)[:half_channels]

                noise = torch.randn((half_channels, H, W),
                                    dtype=dtype,
                                    device=feature_map.device)

                replaced_feature[b, selected_channels] = noise

        return replaced_feature

class ScalePredictionHead(nn.Module):
    """
    从低分辨率特征图预测 patch-level 放大系数 S。
    默认输入 [B, C, H, W]，C≈64~256，H,W≈32~64。
    """
    def __init__(self, in_ch: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   
            nn.Flatten(1),             
            nn.BatchNorm1d(in_ch),
            nn.Linear(in_ch, hidden),
            nn.SiLU(),                 
            nn.Linear(hidden, 1),
            nn.Softplus(beta=5.0)      
        )
        nn.init.constant_(self.net[-2].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)     # [B]


from typing import Tuple, List, Optional
from torchvision.transforms import GaussianBlur


class HEGuidedRegionalAttention(nn.Module):

    
    def __init__(
        self,
        scales: List[float] = [1.0, 2.0, 4.0],
        scale_weights: Optional[List[float]] = None,
        gamma: float = 2.0,
        kappa: float = 0.5,
        alpha: float = 1.0,
        epsilon: float = 1e-3
    ):
        super().__init__()
        
        assert gamma >= 1.0, "gamma must be >= 1.0"
        assert kappa > 0, "kappa must be > 0"
        assert alpha > 0, "alpha must be > 0"
        
        self.scales = scales
        self.scale_weights = scale_weights if scale_weights else [1.0] * len(scales)
        self.gamma = gamma
        self.kappa = kappa
        self.alpha = alpha
        self.epsilon = epsilon
        
        assert len(self.scales) == len(self.scale_weights), "scales and scale_weights must have the same length"

    def compute_gradient_magnitude(
        self, 
        image: torch.Tensor
    ) -> torch.Tensor:

        if image.shape[1] > 1:
            image = image.mean(dim=1, keepdim=True)
        
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
            dtype=image.dtype, 
            device=image.device
        ).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
            dtype=image.dtype, 
            device=image.device
        ).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)
        
        gradient_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        return gradient_mag
    
    def compute_structural_saliency(
        self, 
        he_image: torch.Tensor
    ) -> torch.Tensor:

        saliency = 0.0
        
        for scale, weight in zip(self.scales, self.scale_weights):
            # Apply Gaussian smoothing
            kernel_size = int(6 * scale + 1)  # Ensure odd kernel size
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            blurred = GaussianBlur(
                kernel_size=kernel_size, 
                sigma=scale
            )(he_image)
            
            # Compute gradient magnitude
            grad_mag = self.compute_gradient_magnitude(blurred)
            
            # Normalize by max
            grad_max = grad_mag.flatten(1).max(dim=1, keepdim=True)[0]
            grad_max = grad_max.view(-1, 1, 1, 1)
            grad_normalized = grad_mag / (grad_max + 1e-8)
            
            # Weighted accumulation
            saliency = saliency + weight * grad_normalized
        
        # Normalize to [0, 1]
        saliency = saliency / sum(self.scale_weights)
        
        return saliency
    
    def apply_spatial_transform(
        self,
        saliency_map: torch.Tensor,
        spatial_transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if spatial_transform is None:
            return saliency_map
        
        # Apply grid sampling
        transformed = F.grid_sample(
            saliency_map,
            spatial_transform,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return transformed
    
    def compute_attention_weights(
        self,
        he_image: torch.Tensor,
        spatial_transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        saliency = self.compute_structural_saliency(he_image)
        
        # Apply spatial transformation
        saliency_transformed = self.apply_spatial_transform(
            saliency, 
            spatial_transform
        )
        
        # Normalize to [0, 1]
        saliency_min = saliency_transformed.flatten(1).min(dim=1, keepdim=True)[0]
        saliency_max = saliency_transformed.flatten(1).max(dim=1, keepdim=True)[0]
        saliency_min = saliency_min.view(-1, 1, 1, 1)
        saliency_max = saliency_max.view(-1, 1, 1, 1)
        
        saliency_norm = (saliency_transformed - saliency_min) / (
            saliency_max - saliency_min + 1e-8
        )
        
        # Nonlinear enhancement
        attention_weights = torch.pow(saliency_norm, self.gamma)
        
        return attention_weights
    
    def apply_feature_gating(
        self,
        features: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:

        if attention_weights.shape[-2:] != features.shape[-2:]:
            attention_weights = F.interpolate(
                attention_weights,
                size=features.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
        
        # Apply gating: F^out = F^in ⊙ (1 + κW)
        gating_factor = 1.0 + self.kappa * attention_weights
        features_out = features * gating_factor
        
        return features_out
    

    
    def forward(
        self,
        features: torch.Tensor,
        he_image: torch.Tensor,
        spatial_transform: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        attention_weights = self.compute_attention_weights(
            he_image, 
            spatial_transform
        )
        
        # Apply feature gating
        gated_features = self.apply_feature_gating(features, attention_weights)
        
        if return_weights:
            return gated_features, attention_weights
        else:
            return gated_features, None
        








    

class ModalityDisentangler(nn.Module):
    def __init__(self, G, d):
        super().__init__()
        self.proj_x = nn.Linear(G, d, bias=False)
        self.proj_y = nn.Linear(G, d, bias=False)
        self.ln_x = nn.LayerNorm(d)
        self.ln_y = nn.LayerNorm(d)
        self.alpha = nn.Parameter(torch.zeros(d))  # learnable gate

    def forward(self, x, y):
        hx = self.ln_x(F.gelu(self.proj_x(x)))
        hy = self.ln_y(F.gelu(self.proj_y(y)))
        w = torch.sigmoid(self.alpha)
        g = w * hx + (1 - w) * hy
        return hx, hy, g
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalDisentanglement(nn.Module):

    def __init__(self, gene_dim=1000, embed_dim=64, lambda_mm=1.0):
        super().__init__()
        self.proj_x = nn.Linear(gene_dim, embed_dim, bias=False)
        self.proj_y = nn.Linear(gene_dim, embed_dim, bias=False)
        self.ln_x = nn.LayerNorm(embed_dim)
        self.ln_y = nn.LayerNorm(embed_dim)
        self.alpha = nn.Parameter(torch.zeros(embed_dim))
        self.lambda_mm = lambda_mm

    def forward(self, sc_dict):

        if not isinstance(sc_dict, dict):
            return None, torch.tensor(0.0, device=list(self.parameters())[0].device)

        sc_ot_x = sc_dict.get("ot_x", None)
        sc_ot_y = sc_dict.get("ot_y", None)

        if sc_ot_x is None or sc_ot_y is None:
            return None, torch.tensor(0.0, device=list(self.parameters())[0].device)

        # Convert to tensor if needed
        if not isinstance(sc_ot_x, torch.Tensor):
            sc_ot_x = torch.tensor(sc_ot_x, dtype=torch.float32)
        if not isinstance(sc_ot_y, torch.Tensor):
            sc_ot_y = torch.tensor(sc_ot_y, dtype=torch.float32)

        device = self.alpha.device
        sc_ot_x = sc_ot_x.to(device)
        sc_ot_y = sc_ot_y.to(device)
        hx = self.ln_x(F.gelu(self.proj_x(sc_ot_x)))
        hy = self.ln_y(F.gelu(self.proj_y(sc_ot_y)))
        w = torch.sigmoid(self.alpha)
        g = w * hx + (1 - w) * hy 
        align_loss = (1 - F.cosine_similarity(hx, hy, dim=1)).mean()

        hx_n = F.normalize(hx, dim=1)
        hy_n = F.normalize(hy, dim=1)
        logits = hx_n @ hy_n.T / 0.07
        target = torch.arange(hx.shape[0], device=device)
        nce_loss = F.cross_entropy(logits, target)

        mm_loss = self.lambda_mm * (0.5 * align_loss + 0.5 * nce_loss)

        return g, mm_loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskTokenAdapter(nn.Module):
    def __init__(self, C: int, d_token: int, mode: str = "gap"):
        super().__init__()
        self.mode = mode
        self.proj = nn.Linear(C, d_token)
        if mode == "attn":
            self.attn_conv = nn.Conv2d(C, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.mode == "gap":
            pooled = x.mean(dim=(2, 3))                # [B, C]
        else:
            A = self.attn_conv(x)                      # [B,1,H,W]
            A = torch.softmax(A.view(B, 1, -1), dim=-1).view(B, 1, H, W)
            pooled = (x * A).sum(dim=(2, 3))          # [B, C]
        token = self.proj(pooled)                      # [B, d]
        return token

import torch
import torch.nn as nn

class CrossTaskTokenAttention(nn.Module):
    def __init__(self, d_token: int, num_heads: int = 4, dropout: float = 0.0, num_modules: int = 4): 
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_token, num_heads=num_heads, dropout=dropout, batch_first=True) 
        self.ln1 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, 4 * d_token),
            nn.GELU(),
            nn.Linear(4 * d_token, d_token),
        )
        self.ln2 = nn.LayerNorm(d_token)
        self.module_embed = nn.Parameter(torch.randn(1, num_modules, d_token) * 0.02)

    def forward(self, x):

        attn_out, _ = self.mha(x, x, x)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


class FiLMModulator(nn.Module):
    def __init__(self, d_token: int, C: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_token, 2 * C),
            nn.GELU(),
            nn.Linear(2 * C, 2 * C)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, token: torch.Tensor, feat: torch.Tensor):
        B, C, H, W = feat.shape
        out = self.mlp(token)                           # [B,2C]
        gamma_raw, beta_raw = out.chunk(2, dim=-1)      # [B,C], [B,C]
        gamma = 1.0 + 0.1 * torch.tanh(gamma_raw)
        beta  = 0.0 + 0.1 * torch.tanh(beta_raw)
        gamma = gamma.view(B, C, 1, 1).expand(B, C, H, W)
        beta  = beta.view(B, C, 1, 1).expand(B, C, H, W)
        return gamma, beta
class HEMatchingLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.channel_expand = nn.Conv2d(
            in_channels=3,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, pre_he):
        """
        pre_he: [B, 3, 512]
        return: [B, 128, 64, 64]
        """
        B, C, L = pre_he.shape
        assert C == 3, "the channel of pre_he must be 3"

        H0, W0 = 16, 32 
        assert H0 * W0 == L, "the length of pre_he must be 512"
        x = pre_he.view(B, C, H0, W0)   # [B, 3, 16, 32]

        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)  # [B, 3, 64, 64]

        x = self.channel_expand(x)  # [B, 128, 64, 64]

        return x


class STMatchingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_reduce = nn.Conv2d(512, 128, kernel_size=1)

        self.upsample = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)

    def forward(self, x):
        """
        x: (b, 512, 26, 26)
        return: (b, 128, 64, 64)
        """
        x = self.channel_reduce(x)     # (b, 128, 26, 26)
        x = self.upsample(x)           # (b, 128, 64, 64)
        return x
