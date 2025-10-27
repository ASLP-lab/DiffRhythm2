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


from transformers import LlamaConfig
import torch

import torch.nn as nn
from typing import Optional, Tuple
import math

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from .llama_attention import LLAMA_ATTENTION_CLASSES

# sinusoidal positional encoding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * 1.0
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LlamaAdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size=1024, eps=1e-6, dim_cond=1024):
        super().__init__()
        self.to_weight = nn.Linear(dim_cond, hidden_size)
        nn.init.zeros_(self.to_weight.weight)
        nn.init.ones_(self.to_weight.bias)
        self.variance_epsilon = eps
        self._is_hf_initialized = True  # disable automatic init

    def forward(self, hidden_states, cond_embedding):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        weight = self.to_weight(cond_embedding)
        if len(weight.shape) == 2:
            weight = weight.unsqueeze(1)

        return (weight * hidden_states).to(input_dtype)


class LlamaNARDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, use_flex_attn: bool=False):
        """Override to adaptive layer norm"""
        super().__init__(config, layer_idx)  # init attention, mlp, etc.
        _attn_implementation = config._attn_implementation
        if use_flex_attn:
            _attn_implementation = "flex_attention"
        self.self_attn = LLAMA_ATTENTION_CLASSES[_attn_implementation](config=config, layer_idx=layer_idx)
        # self.input_layernorm = LlamaAdaptiveRMSNorm(
        #     config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        # )
        # self.post_attention_layernorm = LlamaAdaptiveRMSNorm(
        #     config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        # )

    # add `cond` in forward function
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        # print(-1, hidden_states.isnan().sum(), hidden_states.isinf().sum())
        hidden_states = self.input_layernorm(
            hidden_states
        )
        # print(0, hidden_states.isnan().sum(), hidden_states.isinf().sum())
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # print(1, hidden_states.isnan().sum(), hidden_states.isinf().sum())
        hidden_states = residual + hidden_states
        # print(2, hidden_states.isnan().sum(), hidden_states.isinf().sum())
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states
        )
        # print(3, hidden_states.isnan().sum(), hidden_states.isinf().sum())
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        # print(4, hidden_states.isnan().sum(), hidden_states.isinf().sum())
        outputs = [hidden_states,]

        if output_attentions:
            outputs += [self_attn_weights,]

        if use_cache:
            outputs += [present_key_value,]

        return outputs
