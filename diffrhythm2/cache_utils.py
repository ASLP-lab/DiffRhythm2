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


import torch

from typing import Optional, List, Tuple, Dict, Any
from transformers.cache_utils import Cache
from contextlib import contextmanager

class BlockFlowMatchingCache(Cache):
    def __init__(
            self, 
            text_lengths: Optional[torch.Tensor] = None, 
            block_size: Optional[int] = None, 
            num_history_block: Optional[int] = None
        ) -> None:
        super().__init__()
        self._seen_tokens = 0 
        self.text_key_cache: List[torch.Tensor] = []
        self.text_value_cache: List[torch.Tensor] = []
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.text_lengths = text_lengths
        self.block_size = block_size
        self.num_history_block = num_history_block
        self.is_cache_text = False
        self.is_storage_cache = False
        assert (
            (
                self.num_history_block is not None 
                and 
                self.block_size is not None
            ) or self.num_history_block is None
        ), "num_history_block and block_size must be set at the same time."

    @contextmanager
    def cache_text(self):
        self.is_cache_text = True
        try:
            yield self
        finally:
            self.is_cache_text = False

    @contextmanager
    def cache_context(self):
        self.is_storage_cache = True
        try:
            yield self
        finally:
            self.is_storage_cache = False

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # cache text
        if self.is_cache_text:
            if self.text_lengths is None:
                self.text_lengths = torch.LongTensor([key_states.shape[-2]] * key_states.shape[0])
            self.text_key_cache.append(key_states)
            self.text_value_cache.append(value_states)
            return self.text_key_cache[layer_idx], self.text_value_cache[layer_idx]

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx + 1):
                    self.key_cache.append([])
                    self.value_cache.append([])
            cached_key_state = self.key_cache[layer_idx]
            cached_value_state = self.value_cache[layer_idx]
            if len(cached_key_state) != 0:
                key_states = torch.cat([cached_key_state, key_states], dim=-2)
                value_states = torch.cat([cached_value_state, value_states], dim=-2)
            if self.num_history_block is not None:
                history_length = self.block_size * (self.num_history_block + 1)
                key_states = key_states[:, :, -history_length:, :]
                value_states = value_states[:, :, -history_length:, :]
            if self.is_storage_cache:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
        
        k_s = []
        v_s = []

        text_key_cache = (
            self.text_key_cache[layer_idx] 
            if len(self.text_key_cache) > layer_idx 
            else torch.zeros(key_states.shape[0], key_states.shape[1], 0, key_states.shape[3], device=key_states.device, dtype=key_states.dtype)
        )
        text_value_cache = (
            self.text_value_cache[layer_idx] 
            if len(self.text_value_cache) > layer_idx 
            else torch.zeros(value_states.shape[0], value_states.shape[1], 0, value_states.shape[3], device=value_states.device, dtype=value_states.dtype)
        )
        for b in range(self.text_lengths.shape[0]):
            k_s.append(torch.cat([text_key_cache[b][:, :self.text_lengths[b], :], key_states[b]], dim=-2))
            v_s.append(torch.cat([text_value_cache[b][:, :self.text_lengths[b], :], value_states[b]], dim=-2))
        k_s = torch.nn.utils.rnn.pad_sequence(k_s, batch_first=True)
        v_s = torch.nn.utils.rnn.pad_sequence(v_s, batch_first=True)

        return k_s, v_s

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

