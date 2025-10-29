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
from torch import nn
from tqdm import tqdm

from torchdiffeq import odeint
from .backbones.dit import DiT
from .cache_utils import BlockFlowMatchingCache


class CFM(nn.Module):
    def __init__(
        self,
        transformer: DiT,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler" # 'midpoint'
            # method="adaptive_heun"
        ),
        odeint_options: dict = dict(
            min_step=0.05
        ),
        num_channels=None,
        block_size=None,
        num_history_block=None
    ):
        super().__init__()

        self.num_channels = num_channels

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs
        print(f"ODE SOLVER: {self.odeint_kwargs['method']}")
        
        self.odeint_options = odeint_options
        self.block_size = block_size
        self.num_history_block = num_history_block
        if self.num_history_block is not None and self.num_history_block <= 0:
            self.num_history_block = None

        print(f"block_size: {self.block_size}; num_history_block: {self.num_history_block}")

    @property
    def device(self):
        return next(self.parameters()).device
    
    @torch.no_grad()
    def sample_block_cache(
        self,
        text,
        duration,  # noqa: F821
        style_prompt,
        steps=32,
        cfg_strength=1.0,
        seed: int | None = None,
        process_bar = True
        
    ):
        self.eval()

        batch = text.shape[0]
        device = self.device
        num_blocks = duration // self.block_size + (duration % self.block_size > 0)

        text_emb = self.transformer.text_embed(text)
        cfg_text_emb = self.transformer.text_embed(torch.zeros_like(text))
        text_lens = torch.LongTensor([text_emb.shape[1]]).to(device)
        clean_emb_stream = torch.zeros(batch, 0, self.num_channels, device=device, dtype=text_emb.dtype)
        noisy_lens = torch.LongTensor([self.block_size]).to(device)
        block_iterator = range(num_blocks)
        if process_bar:
            block_iterator = tqdm(block_iterator)

        # create cache
        kv_cache = BlockFlowMatchingCache(text_lengths=text_lens, num_history_block=self.num_history_block)
        cfg_kv_cache = BlockFlowMatchingCache(text_lengths=text_lens, num_history_block=self.num_history_block)
        cache_time = torch.tensor([1], device=device)[:, None].repeat(batch, self.block_size).half()
        
        # generate text cache
        text_time = torch.tensor([-1], device=device)[:, None].repeat(batch, text_emb.shape[1]).half()
        text_position_ids = torch.arange(0, text_emb.shape[1], device=device)[None, :].repeat(batch, 1)
        text_attn_mask = torch.ones(batch, 1, text_emb.shape[1], text_emb.shape[1], device=device).bool()
        
        if text_emb.shape[1] != 0: 
            with kv_cache.cache_text():
                _, _, kv_cache = self.transformer(
                    x = text_emb,
                    time=text_time,
                    attn_mask=text_attn_mask,
                    position_ids=text_position_ids,
                    style_prompt=style_prompt, 
                    use_cache=True,
                    past_key_value = kv_cache
                )
            with cfg_kv_cache.cache_text():
                _, _, cfg_kv_cache = self.transformer(
                    x = cfg_text_emb,
                    time=text_time,
                    attn_mask=text_attn_mask,
                    position_ids=text_position_ids,
                    style_prompt=torch.zeros_like(style_prompt), 
                    use_cache=True,
                    past_key_value = cfg_kv_cache
                )

        end_pos = 0
        for bid in block_iterator:
            clean_lens = torch.LongTensor([clean_emb_stream.shape[1]]).to(device)
            #print(text_lens, clean_lens, noisy_lens, clean_emb_stream.shape, flush=True)

            # all one mask
            attn_mask = torch.ones(batch, 1, noisy_lens.max(), (text_lens + clean_lens + noisy_lens).max(), device=device).bool() # [B, 1, Q, KV]

            # generate position id
            position_ids = torch.arange(0, (clean_lens + noisy_lens).max(), device=device)[None, :].repeat(batch, 1)
            position_ids = position_ids[:, -noisy_lens.max():]

            # core sample fn
            def fn(t, x):
                noisy_embed = self.transformer.latent_embed(x)

                if t.ndim == 0:
                    t = t.repeat(batch)
                time = t[:, None].repeat(1, noisy_lens.max())

                pred, *_ = self.transformer(
                    x=noisy_embed, 
                    time=time, 
                    attn_mask=attn_mask,
                    position_ids=position_ids,
                    style_prompt=style_prompt, 
                    use_cache=True,
                    past_key_value = kv_cache
                )
                if cfg_strength < 1e-5:
                    return pred

                null_pred, *_ = self.transformer(
                    x=noisy_embed, 
                    time=time, 
                    attn_mask=attn_mask,
                    position_ids=position_ids,
                    style_prompt=torch.zeros_like(style_prompt), 
                    use_cache=True,
                    past_key_value = cfg_kv_cache
                )

                return pred + (pred - null_pred) * cfg_strength

            # generate time
            noisy_emb = torch.randn(batch, self.block_size, self.num_channels, device=device, dtype=style_prompt.dtype)
            t_start = 0
            t_set = torch.linspace(t_start, 1, steps, device=device, dtype=noisy_emb.dtype)
            
            # sampling
            outputs = odeint(fn, noisy_emb, t_set, **self.odeint_kwargs)
            sampled = outputs[-1]

            # generate next kv cache
            cache_embed = self.transformer.latent_embed(sampled)
            with kv_cache.cache_context():
                _, _, kv_cache = self.transformer(
                    x = cache_embed,
                    time=cache_time,
                    attn_mask=attn_mask,
                    position_ids=position_ids,
                    style_prompt=style_prompt, 
                    use_cache=True,
                    past_key_value = kv_cache
                )
            with cfg_kv_cache.cache_context():
                _, _, cfg_kv_cache = self.transformer(
                    x = cache_embed,
                    time=cache_time,
                    attn_mask=attn_mask,
                    position_ids=position_ids,
                    style_prompt=torch.zeros_like(style_prompt), 
                    use_cache=True,
                    past_key_value = cfg_kv_cache
                )

            # push new block
            clean_emb_stream = torch.cat([clean_emb_stream, sampled], dim=1)
            
            pos = -1
            curr_frame = clean_emb_stream[:, pos, :]
            eos = torch.ones_like(curr_frame)
            last_kl = torch.nn.functional.mse_loss(
                curr_frame, 
                eos
            )
            if last_kl.abs() <= 0.05:
                while last_kl.abs() <= 0.05 and abs(pos) < clean_emb_stream.shape[1]:
                    pos -= 1
                    curr_frame = clean_emb_stream[:, pos, :]
                    last_kl = torch.nn.functional.mse_loss(
                        curr_frame, 
                        eos
                    )
                end_pos = clean_emb_stream.shape[1] + pos
                break
            else:
                end_pos = clean_emb_stream.shape[1]
                
        clean_emb_stream = clean_emb_stream[:, :end_pos, :]

        return clean_emb_stream

