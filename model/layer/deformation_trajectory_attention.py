#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

"""Video models."""

from ast import Global
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath
        

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            else:
                pass


class DeformationTrajectoryAttentionBlock(nn.Module):
    def __init__(
            self, dim=768, num_heads=12, x_size=16, y_size=16, frame_size=8, qkv_bias=False, mlp_ratio=4., 
            attn_drop=0., proj_drop=0., mlp_drop=0., drop_path=0., 
            act_layer=nn.GELU, norm_layer=nn.LayerNorm
        ):
        super().__init__()

        self.frame_size = frame_size
        self.img_size = [x_size, y_size]
        self.patch_embed = PatchEmbed(img_size=self.img_size, embed_dim=dim, frame_size=frame_size)
        
        self.norm1 = norm_layer(dim)
        self.attn = DeformationTrajectoryAttention(
            dim=dim, num_heads=num_heads, x_size=16, y_size=16, frame_size=frame_size, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, mlp_drop=mlp_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        initialize_weights(self.patch_embed, self.norm1, self.norm2, self.mlp)

    def forward(self, x):
        x = self.patch_embed(x)
        
        norm1_x = self.norm1(x)
        attended_x, deformation_attns = self.attn(norm1_x)
        x = x + self.drop_path(attended_x)
        # print(5, x.min().item(), x.max().item(), x.mean().item())

        norm2_x = self.norm2(x)
        x = x + self.drop_path(self.mlp(norm2_x))

        x = rearrange(x, f'b (f h w) d -> (b f) d h w', f=self.frame_size, h=16)
        x = nn.functional.interpolate(
            x, size=self.img_size, mode="bilinear", align_corners=False
        )
        # print(6, x.min().item(), x.max().item())
        
        return x, deformation_attns


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=[32, 32], embed_dim=320, frame_size=8):
        super().__init__()
        num_patches = 16 * 16
        img_size = img_size
        patch_size = [img_size[0] // 16, img_size[1] // 16]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.frame_size = frame_size

        self.proj = nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = rearrange(x, f'(b f) d x y -> b (f x y) d', f=self.frame_size)
        return x

        
class DeformationTrajectoryAttention(nn.Module):
    def __init__(self, dim, num_heads, x_size, y_size, frame_size, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.x_size = x_size
        self.y_size = y_size
        self.frame_size = frame_size

        # joint attention
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # deformation attention
        self.proj_q_defo = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv_defo = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop_defo = nn.Dropout(attn_drop)

        # for last proj
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        initialize_weights(self.qkv, self.proj_q_defo, self.proj_kv_defo, self.proj)

    def forward(self, x):
        seq_len = self.x_size * self.x_size
        num_frames = self.frame_size

        B, TS, D = x.shape

        H = self.num_heads
        S = seq_len # X * Y
        F = num_frames
        assert (S * F == TS)


        # # ------------------ Joint Attention with TS * T * S ------------------ 
        # project x to q, k, v values
        q, k, v = self.qkv(x).chunk(3, dim=-1)   
        # Reshape: 'b n (h d) -> (b h) n d'
        q_, k_, v_ = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=H), (q, k, v))
        # Using full attention
        q_dot_k = q_ @ k_.transpose(-2, -1)
        q_dot_k_defo = rearrange(q_dot_k, 'b n (f s) -> b n s f', s=S)

        # # ------------------ Deformation Attention ------------------ 
        # # ------------------ First Temproal Attention: TS * T * S -> TS * S ------------------ 
        time_attn_defo = (self.scale * q_dot_k_defo).softmax(dim=-1)
        time_attn_defo = self.attn_drop_defo(time_attn_defo)
        v_defo = rearrange(v_, 'b (f s) d -> b s f d', f=F, s=S)
        # Spatial attended results
        x_defo = torch.einsum('b n s f, b s f d -> b n s d', time_attn_defo, v_defo)

        # # ------------------ Second Spatial Attention with TS * S -> TS------------------ 
        # Spatial attention: query is the similarity-aggregated time
        x_defo = rearrange(x_defo, '(b h) n s d -> b n s (h d)', b=B)

        # Get q2, k2, v2
        x_diag_defo = rearrange(x_defo, 'b (f g) s d -> b g f s d', g=S)
        x_diag_defo = torch.diagonal(x_diag_defo, dim1=-4, dim2=-2)
        x_diag_defo = rearrange(x_diag_defo, f'b f d s -> b (f s) d', s=S)
        q2_defo = self.proj_q_defo(x_diag_defo)
        q2_defo = rearrange(q2_defo, f'b n (h d) -> b h n d', h=H)
        q2_defo *= self.scale
        k2_defo, v2_defo = self.proj_kv_defo(x_defo).chunk(2, dim=-1)
        k2_defo, v2_defo = map(lambda t: rearrange(t, f'b n s (h d) -> b h n s d', s=S, h=H), (k2_defo, v2_defo))
        # Temporal attended results
        space_attn_defo = torch.einsum('b h n d, b h n s d -> b h n s', q2_defo, k2_defo)
        space_attn_defo = space_attn_defo.softmax(dim=-1)
        x_defo = torch.einsum('b h n s, b h n s d -> b h n d', space_attn_defo, v2_defo)
        x_defo = rearrange(x_defo, f'b h n d -> b n (h d)')
        
        x = self.proj(x_defo)
        x = self.proj_drop(x)

        return x, space_attn_defo


class MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, 
        out_features=None, act_layer=nn.GELU, mlp_drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(mlp_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
