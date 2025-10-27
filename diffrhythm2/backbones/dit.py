# Copyright 2025 ASLP Lab and Xiaomi Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import torch
import math
from torch import nn

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig
from .llama_nar import LlamaNARDecoderLayer

class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

    def forward(self, text: int["b nt"]):  # noqa: F722
        text = self.text_embed(text)  # b n -> b n d
        return text


class InputEmbedding(nn.Module):
    def __init__(self, cond_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, cond_dim)
        self.proj_2 = nn.Linear(cond_dim, out_dim)

    def forward(self, x, style_emb, time_emb):  # noqa: F722
        style_emb = style_emb.unsqueeze(1).repeat(1, x.shape[1], 1)
        x_orig = x
        x = x + style_emb + time_emb
        x = self.proj(x) + x_orig
        x = self.proj_2(x)
        return x


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=-1)

        x = self.norm(x) * (1 + scale) + shift
        return x


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def numel(self):
        return 0


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: float["b"]):  # noqa: F821
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        conv_layers=0,
        long_skip_connection=False,
        use_flex_attn=False,
        repa_depth=-1,
        repa_dims=[1024],
        **kwargs
    ):
        super().__init__()

        cond_dim = 512
        self.time_embed = TimestepEmbedding(cond_dim)
        self.text_embed = TextEmbedding(text_num_embeds, cond_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(cond_dim, dim)

        self.latent_embed = torch.nn.Sequential(
            nn.Linear(mel_dim, cond_dim),
            nn.Linear(cond_dim, cond_dim)
        )

        self.dim = dim
        self.depth = depth
        self.use_flex_attn = use_flex_attn

        llama_config = LlamaConfig(
            hidden_size=dim, 
            num_attention_heads=heads,
            intermediate_size=dim * ff_mult, 
            hidden_act='silu', 
            max_position_embeddings=4096
        )
        self.rotary_embed = LlamaRotaryEmbedding(config=llama_config)
        llama_config._attn_implementation = 'sdpa'
        self.transformer_blocks = nn.ModuleList(
            [LlamaNARDecoderLayer(llama_config, layer_idx=i, use_flex_attn=self.use_flex_attn) for i in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None


        self.norm_out = AdaLayerNormZero_Final(dim, cond_dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.repa_depth = repa_depth
        self.repa_dims = repa_dims
        self.projectors = None
        if self.repa_depth > 0:
            self.projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.dim, self.dim * 2),
                    nn.SiLU(),
                    nn.Linear(self.dim * 2, self.dim * 2),
                    nn.SiLU(),
                    nn.Linear(self.dim * 2, repa_dim),
                ) for repa_dim in self.repa_dims
            ])


    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        position_ids: torch.Tensor,
        style_prompt: torch.Tensor,
        attn_mask: torch.Tensor,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_value = None,
    ):
        """
        Args:
        x: [b, n, d]
        time: [b, n, 1]
        position_ids: [b, n]
        style_prompt: [b, 512]
        attn_mask: [b, 1, n, n]
        """
        batch, seq_len = x.shape[0], x.shape[1]
        t = self.time_embed(time)
        c = t # [B, T, dim]

        x = self.input_embed(x, style_prompt, c)

        if self.long_skip_connection is not None:
            residual = x

        position_embeddings = self.rotary_embed(x, position_ids)

        attn_weights = []
        if not use_cache:
            past_key_value = None

        repa_res = None
        for i, block in enumerate(self.transformer_blocks):
            res = block(
                x, 
                attention_mask=attn_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            x = res.pop(0)
            if output_attentions:
                attn_weights.append(res.pop(0))
            if use_cache:
                past_key_value = res.pop(0)
            if i == self.repa_depth - 1:
                repa_res = x
            
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, c)
        output = self.proj_out(x)

        return output, attn_weights, past_key_value