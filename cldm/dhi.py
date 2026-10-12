import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)


class ResidualBlock(nn.Module):
    def __init__(self, dims, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv_nd(dims, in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.silu = nn.SiLU()
        self.conv2 = conv_nd(dims, out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.residual = conv_nd(dims, in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.silu(out)
        out = self.conv2(out)
        out += residual
        out = self.silu(out)
        return out


class PatchMerging(nn.Module):
    def __init__(self, patch_dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.patch_dim = patch_dim
        self.norm = norm_layer(4 * patch_dim)
        self.reduction = nn.Linear(4 * patch_dim, 2 * patch_dim)

    def forward(self, x):
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, "Height and Width must be even."

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        x = x.view(B, H // 2, W // 2, 2 * C)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, hint_channels, dim=2):
        super(FeatureExtractor, self).__init__()
        self.initial_conv = conv_nd(dim, hint_channels, 16, kernel_size=3, stride=1, padding=1)
        self.layer1 = ResidualBlock(dim, 16, 16)
        self.conv_before_res2 = conv_nd(dim, 16, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = ResidualBlock(dim, 32, 32)
        self.patch_merge1 = PatchMerging(patch_dim=32)
        self.conv_before_res3 = conv_nd(dim, 64, 64, kernel_size=3, stride=1, padding=1)
        self.layer3 = ResidualBlock(dim, 64, 64)
        self.patch_merge2 = PatchMerging(patch_dim=64)
        self.conv_before_res4 = conv_nd(dim, 128, 128, kernel_size=3, stride=1, padding=1)
        self.layer4 = ResidualBlock(dim, 128, 128)
        self.patch_merge3 = PatchMerging(patch_dim=128)
        self.conv_before_res5 = conv_nd(dim, 256, 256, kernel_size=3, stride=1, padding=1)
        self.layer5 = ResidualBlock(dim, 256, 256)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.conv_before_res2(x)
        x = self.layer2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.patch_merge1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_before_res3(x)
        x = self.layer3(x)
        x = x.permute(0, 2, 3, 1)
        x = self.patch_merge2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_before_res4(x)
        x = self.layer4(x)
        x = x.permute(0, 2, 3, 1)
        x = self.patch_merge3(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_before_res5(x)
        x = self.layer5(x)
        return x


class TextureEnhancementEncoder(nn.Module):
    def __init__(self, in_channels, dim=2):
        super(TextureEnhancementEncoder, self).__init__()
        
        self.high_freq_conv = conv_nd(dim, in_channels, 16, kernel_size=3, stride=1, padding=1)
        
        self.layer1 = ResidualBlock(dim, 16, 16)
        self.downsample1 = conv_nd(dim, 16, 32, kernel_size=3, stride=2, padding=1)  
        
        self.layer2 = ResidualBlock(dim, 32, 32)
        self.downsample2 = conv_nd(dim, 32, 64, kernel_size=3, stride=2, padding=1)
        
        self.layer3 = ResidualBlock(dim, 64, 64)
        self.downsample3 = conv_nd(dim, 64, 128, kernel_size=3, stride=2, padding=1)
        
        self.layer4 = ResidualBlock(dim, 128, 128)
        self.downsample4 = conv_nd(dim, 128, 256, kernel_size=3, stride=2, padding=1)
        
        self.layer5 = ResidualBlock(dim, 256, 256)
        

        self.texture_attention = TextureAttentionModule(256)
        

        self.upsample1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        

        self.final_conv = conv_nd(dim, 16, 256, kernel_size=3, stride=1, padding=1)

    def extract_texture_features(self, x):

        laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                       dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(x.size(1), 1, 1, 1)
        high_freq = F.conv2d(x, laplacian_kernel, padding=1, groups=x.size(1))
        return high_freq

    def forward(self, x):
        texture_base = self.extract_texture_features(x)
        x = self.high_freq_conv(texture_base)
        
        enc1 = self.layer1(x)  # 16 channels
        x = self.downsample1(enc1)  # 32 channels
        
        enc2 = self.layer2(x)  # 32 channels
        x = self.downsample2(enc2)  # 64 channels
        
        enc3 = self.layer3(x)  # 64 channels
        x = self.downsample3(enc3)  # 128 channels
        
        enc4 = self.layer4(x)  # 128 channels
        x = self.downsample4(enc4)  # 256 channels
        
        bottleneck = self.layer5(x)
        
        attended_features = self.texture_attention(bottleneck)
        
        x = self.upsample1(attended_features)
        x = x + enc4  
        
        x = self.upsample2(x)
        x = x + enc3  
        
        x = self.upsample3(x)
        x = x + enc2  
        
        x = self.upsample4(x)
        x = x + enc1  
        

        texture_features = self.final_conv(x)
        return texture_features


class TextureAttentionModule(nn.Module):
    def __init__(self, channels):
        super(TextureAttentionModule, self).__init__()
        self.channels = channels
        

        self.spatial_attention = nn.Sequential(
            conv_nd(2, channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            conv_nd(2, channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        spatial_weights = self.spatial_attention(x)
        x_spatial = x * spatial_weights
        

        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        

        attended_x = x_spatial + x_channel
        return attended_x


class AdaptiveFeatureFusion(nn.Module):

    def __init__(self, structure_channels, texture_channels, output_channels):
        super(AdaptiveFeatureFusion, self).__init__()
        

        self.structure_conv = conv_nd(2, structure_channels, output_channels // 2, kernel_size=1)
        

        self.texture_conv = conv_nd(2, texture_channels, output_channels // 2, kernel_size=1)
        

        self.adaptive_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(output_channels, output_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels // 4, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        

        self.fusion_enhance = nn.Sequential(
            conv_nd(2, output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            conv_nd(2, output_channels, output_channels, kernel_size=3, padding=1)
        )

    def forward(self, structure_features, texture_features):

        structure_processed = self.structure_conv(structure_features)
        texture_processed = self.texture_conv(texture_features)
        

        combined = torch.cat([structure_processed, texture_processed], dim=1)
        

        weights = self.adaptive_weights(combined)
        structure_weight = weights[:, 0:1, :, :]
        texture_weight = weights[:, 1:2, :, :]
        
        weighted_fusion = (structure_processed * structure_weight + 
                          texture_processed * texture_weight)
        
        enhanced_features = self.fusion_enhance(weighted_fusion)
        
        return enhanced_features


class DualBranchFeatureExtractor(nn.Module):
    def __init__(self, hint_channels, dim=2, use_texture_branch=True):
        super(DualBranchFeatureExtractor, self).__init__()
        self.use_texture_branch = use_texture_branch
        
        self.structure_branch = FeatureExtractor(hint_channels, dim)
        
        if use_texture_branch:

            self.texture_branch = TextureEnhancementEncoder(hint_channels, dim)
            

            self.feature_fusion = AdaptiveFeatureFusion(256, 256, 256)
        else:
            self.texture_branch = None
            self.feature_fusion = None

    def forward(self, x):
        structure_features = self.structure_branch(x)
        
        if self.use_texture_branch and self.texture_branch is not None:
            texture_features = self.texture_branch(x)
            

            fused_features = self.feature_fusion(structure_features, texture_features)
            return fused_features
        else:

            return structure_features