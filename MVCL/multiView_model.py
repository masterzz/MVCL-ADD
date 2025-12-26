# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
# author: zhubr17

# %%
# %load_ext autoreload
# %autoreload 2

# %%
#加入动态声纹层
import math
import random
from argparse import Namespace
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
from torchvision.transforms import v2

# %% editable=true slideshow={"slide_type": ""}
try:
    from .resnet import ResNet, convert_2d_to_1d
    from .resnet1d import ResNet1D
    from .utils import Expand, Squeeze, WavLM_1D, GatedFusionLayer
except ImportError:
    from resnet import ResNet, convert_2d_to_1d
    from resnet1d import ResNet1D
    from utils import Expand, Squeeze, WavLM_1D, GatedFusionLayer


# %%
# feature_model2D = ResNet(pretrained=True, verbose=True)
# feature_model1D = WavLM_1D()

# x = torch.randn(3, 1, 48000)
# x = feature_model2D.compute_stage1(x)
# x = feature_model2D.compute_stage2(x)
# x = feature_model2D.compute_stage3(x)
# x = feature_model2D.compute_stage4(x)
# x = feature_model2D.compute_latent_feature(x)

# %%
# squeeze_modules, expand_modules = [], []
# for dim, h, w in [[64, 65, 65], [128, 33, 33], [256, 17, 17], [512, 9, 9]]:
#     squeeze_modules.append(Squeeze(time_len=149, time_dim=768, spec_dim=dim, spec_height=h, spec_width=w))
#     expand_modules.append(Expand(time_len=149, time_dim=768, spec_dim=dim, spec_height=h, spec_width=w))

def _set_module_trainable(module, trainable: bool):
    """set `trainable` attribute for the parameters in the input module

    Args:
        module: the torch module
        trainable: True or False

    Raise:
        ValueError: raise when trainable is not True or False

    """

    if not trainable in [True, False]:
        raise ValueError(
            f"Error, please input True/False for the trainable attribute, your input is {trainable}"
        )

    for param in module.parameters():
        param.requires_grad = trainable
        
def freeze_modules(x):
    """
    freeze all the parameters of of x, i.e, set all parameter un-trainable.

    Args:
        x: a torch module or a list of modules
    """

    if isinstance(x, list):
        for _x in x:
            freeze_modules(_x)
    else:
        _set_module_trainable(x, False)





# %% editable=true slideshow={"slide_type": ""}
class MultiViewModel(nn.Module):
    def __init__(self, verbose=0, cfg=None, args=None, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.args = args

        self.feature_model2D = ResNet(pretrained=True)
        self.feature_model1D = WavLM_1D()

        if cfg is not None and cfg.only_1D:
            freeze_modules(self.feature_model2D)
        if cfg is not None and cfg.only_2D:
            freeze_modules(self.feature_model1D)

        use_PE = True
        use_fusion = cfg.use_fusion
        drop_layer = cfg.drop_layer
        squeeze_modules, expand_modules = [], []
        heads = [1, 2, 4, 4]
        for i, (dim, h, w) in enumerate([[64, 65, 65], [128, 33, 33], [256, 17, 17], [512, 9, 9]]):
            squeeze_modules.append(
                Squeeze(
                    time_len=149,
                    time_dim=768,
                    spec_dim=dim,
                    spec_height=h,
                    spec_width=w,
                    use_PE=use_PE,
                    drop_layer=drop_layer,
                    use_fusion=True if use_fusion and cfg.use_fusion1D else False,
                )
            )
            expand_modules.append(
                Expand(
                    time_len=149,
                    time_dim=768,
                    spec_dim=dim,
                    spec_height=h,
                    spec_width=w,
                    num_heads=heads[i],
                    use_PE=use_PE,
                    drop_layer=drop_layer,
                    use_fusion=True if use_fusion and cfg.use_fusion2D else False,
                )
            )
        self.squeeze_modules = nn.ModuleList(squeeze_modules)
        self.expand_modules = nn.ModuleList(expand_modules)

        # self.gated_layer = GatedFusionLayer(768, 512, 768+512)

        final_dim = 768 + 512
        # final_dim = 512
        self.cls_final = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(final_dim, 1),
        )

        feat_dim = [768, 512]
        self.cls1D, self.cls2D = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feat_dim[i], 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 1),
                )
                for i in range(2)
            ]
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.set_verbose(verbose)

    def norm_feat(self, feat):
        feat = feat / (1e-9 + torch.norm(feat, p=2, dim=-1, keepdim=True))
        return feat

    def print_shape(self, *args):
        for x in args:
            print(x.shape)

    def set_verbose(self, verbose):
        self.feature_model1D.verbose = verbose
        self.feature_model2D.verbose = verbose
        self.verbose = verbose

    def forward(self, x, stage="test", batch=None, spec_aug=None, freeze_feature_extractor=False, **kwargs):
        batch_size = x.shape[0]
        res = {}

        with torch.no_grad():
            wav1 = self.feature_model1D.compute_stage1(x)
            spec1, res["raw_spec"] = self.feature_model2D.compute_stage1(x, spec_aug=spec_aug)

        if freeze_feature_extractor:
            _ = torch.set_grad_enabled(False)

        fused_wav1 = self.squeeze_modules[0](wav1, spec1)
        fused_spec1 = self.expand_modules[0](wav1, spec1)
        wav2, position_bias = self.feature_model1D.compute_stage2(fused_wav1)
        spec2 = self.feature_model2D.compute_stage2(fused_spec1)

        wav3, position_bias = self.feature_model1D.compute_stage3(self.squeeze_modules[1](wav2, spec2), position_bias)
        spec3 = self.feature_model2D.compute_stage3(self.expand_modules[1](wav2, spec2))

        wav4, position_bias = self.feature_model1D.compute_stage4(self.squeeze_modules[2](wav3, spec3), position_bias)
        spec4 = self.feature_model2D.compute_stage4(self.expand_modules[2](wav3, spec3))

        wav5, res["raw_wav_feat"] = self.feature_model1D.compute_latent_feature(
            self.squeeze_modules[3](wav4, spec4), position_bias
        )
        spec5 = self.feature_model2D.compute_latent_feature(self.expand_modules[3](wav4, spec4))

        # if stage == "train":
        #     shuffle_id = torch.randperm(batch_size)
        #     # res["feature"] = exchange_mu_std(res["feature"], res["feature"][shuffle_id], dim=-1)
        #     wav5 = exchange_mu_std(wav5, wav5[shuffle_id], dim=-1)
        #     shuffle_id = torch.randperm(batch_size)
        #     spec5 = exchange_mu_std(spec5, spec5[shuffle_id], dim=-1)

        if freeze_feature_extractor and self.training:
            _ = torch.set_grad_enabled(True)

        res["feature1D"] = self.norm_feat(wav5)
        res["feature2D"] = self.norm_feat(spec5)
        res["feature"] = self.norm_feat(torch.concat([wav5, spec5], dim=-1))
        # res["feature"] = self.norm_feat(self.gated_layer(wav5, spec5))

        res["logit1D"] = self.cls1D(res["feature1D"]).squeeze(-1)
        res["logit2D"] = self.cls2D(res["feature2D"]).squeeze(-1)
        res["logit"] = self.cls_final(res["feature"]).squeeze(-1)
        # res["logit"] = (res["logit1D"] + res["logit2D"])/2
        return res


# %% editable=true slideshow={"slide_type": ""} tags=["style-activity", "active-ipynb"]
# # cfg = Namespace(alpha=0.5, beta=0.5, only_1D=False, only_2D=False, use_fusion=True)
# # model = MultiViewModel(verbose=1, cfg=cfg, args=cfg)
# # x = torch.randn(3, 1, 48000)
# # model(x)

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# # model = model.cuda()
# # x = x.cuda()
# # model(x)


# %%
def exchange_mu_std(x, y, dim=None):
    mu_x = torch.mean(x, dim=dim, keepdims=True)
    mu_y = torch.mean(y, dim=dim, keepdims=True)
    std_x = torch.std(x, dim=dim, keepdims=True)
    std_y = torch.std(y, dim=dim, keepdims=True)

    alpha = np.random.randint(50, 100) / 100
    target_mu = alpha * mu_x + (1 - alpha) * mu_y
    target_std = alpha * std_x + (1 - alpha) * std_y
    z = target_std * ((x - mu_x) / std_x) + target_mu

    # z = random_noise(z, noise_level=10)

    return z


def random_noise(x, noise_level=10):
    add_noise_level = np.random.randint(0, noise_level) / 100
    mult_noise_level = np.random.randint(0, noise_level) / 100
    z = _noise(x, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level)
    return z


def _noise(x, add_noise_level=0.0, mult_noise_level=0.0):
    add_noise = 0.0
    mult_noise = 1.0
    if add_noise_level > 0.0:
        add_noise = add_noise_level * np.random.beta(2, 5) * torch.FloatTensor(x.shape).normal_().to(x.device)
    if mult_noise_level > 0.0:
        mult_noise = (
            mult_noise_level * np.random.beta(2, 5) * (2 * torch.FloatTensor(x.shape).uniform_() - 1).to(x.device) + 1
        )
    return mult_noise * x + add_noise

class DynamicTrajectoryBranch(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2):
        super().__init__()
        
        # 1. 降维投影：先压缩特征，减少计算量，聚焦核心信息
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 2. 动态演化建模：使用 Bi-GRU 捕捉长时依赖
        # batch_first=True -> (Batch, Seq, Feature)
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        
        # 3. 轨迹注意力机制 (Trajectory Attention)
        # 并不是所有时间点都重要，让模型自动寻找"演化最剧烈"的片段
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)
        
        # 4. 最终映射
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        """
        Input: x (B, T, D) - 来自 WavLM 的序列特征
        Output: 
            - trajectory_feat: (B, Hidden) - 聚合后的全局轨迹特征
            - raw_trajectory: (B, T, Hidden*2) - 每一帧的动态状态
        """
        B, T, D = x.shape
        
        # --- A. 特征预处理 ---
        x_proj = self.input_proj(x)  # (B, T, hidden_dim)
        
        # --- B. 计算一阶差分 (模拟速度/演化趋势) ---
        # Deepfake 往往在帧间过渡时出现瑕疵
        # pad 第一个时间步以保持长度一致
        x_diff = x_proj - F.pad(x_proj[:, :-1, :], (0, 0, 1, 0))
        
        # 将原始特征和差分特征融合 (这里简单相加，也可以拼接)
        x_dynamic = x_proj + x_diff
        
        # --- C. 时序建模 (Bi-GRU) ---
        # output: (B, T, hidden_dim * 2)
        # hn: (num_layers*2, B, hidden_dim)
        lstm_out, _ = self.temporal_encoder(x_dynamic)
        
        # --- D. 动态注意力聚合 ---
        # 计算每个时间步的权重 score
        attn_scores = self.attention_fc(lstm_out)   # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1) # (B, T, 1)
        
        # 加权求和：得到能够代表整段语音"演化轨迹"的单个向量
        trajectory_feat = torch.sum(lstm_out * attn_weights, dim=1) # (B, hidden_dim * 2)
        
        # 最后的映射
        final_feat = self.output_proj(trajectory_feat) # (B, hidden_dim)
        
        return final_feat, lstm_out