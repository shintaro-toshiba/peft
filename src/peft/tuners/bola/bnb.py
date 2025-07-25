# Copyright 2023-present the HuggingFace Inc. team.
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

import warnings
from typing import Optional, Union

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils.integrations import dequantize_bnb_weight
from peft.tuners.tuners_utils import check_adapters_to_merge

from .layer import BolaLayer


if is_bnb_available():

	class Linear8bitLt(nn.Module, BolaLayer):
		def __init__(
			self,
			base_layer: nn.Module,
			adapter_name: str,
			num_in_blocks: int = 8,
			num_out_blocks: int = 8,
			top_k: int = 8,
			alpha: float = 1.0,
			dropout: float = 0.0,
			init_weights: Union[bool, str] = False,
			**kwargs,
		) -> None:
			super().__init__()
			BolaLayer.__init__(self, base_layer)
			self.fan_in_fan_out = False

			self._active_adapter = adapter_name
			self.update_layer(
				adapter_name,
				num_in_blocks=num_in_blocks,
				num_out_blocks=num_out_blocks,
				top_k=top_k,
				alpha=alpha,
				dropout=dropout,
				init_weights=init_weights,
			)

		def merge(
			self,
			safe_merge: bool = False,
			adapter_names: Optional[list[str]] = None,
		) -> None:
			if self.merged:
				warnings.warn(
					f"Already following adapters were merged {','.join(self.merged_adapters)}. "
					f"You are now additionally merging {','.join(self.active_adapters)}."
				)
			adapter_names = check_adapters_to_merge(self, adapter_names)
			if not adapter_names:
				# no adapter to merge
				return

			for active_adapter in adapter_names:
				if active_adapter not in self.bola_w_p.keys():
					continue
				warnings.warn(
					'Merge lora module to 8-bit linear may get different generations due to rounding errors.'
				)
				delta_weight = self.get_delta_weight(active_adapter)
				weight = self.get_base_layer().weight
				state = self.get_base_layer().state
				if state.SCB is None:
					state.SCB = weight.SCB

				# Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
				# dequantization directly
				output = dequantize_bnb_weight(weight, state=state)
				weight_data = (
					output.to(delta_weight.dtype).to(delta_weight.device) + delta_weight
				)
				if safe_merge and not torch.isfinite(weight_data).all():
					raise ValueError(
						f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken'
					)

				self.get_base_layer().weight = bnb.nn.Int8Params(
					weight_data.to('cpu'),
					requires_grad=False,
					has_fp16_weights=weight.has_fp16_weights,
				).to(weight.device)

				# TODO: use bias
				# if self.bola_bias[active_adapter] is not None:
				#     bias_data = self.get_base_layer().bias.data + self.bola_bias[active_adapter]
				#     if safe_merge and not torch.isfinite(bias_data):
				#         raise ValueError(
				#             f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
				#         )
				#     self.get_base_layer().bias.data = bias_data

				state.reset_grads()
				self.merged_adapters.append(active_adapter)

		def unmerge(self) -> None:
			if not self.merged:
				warnings.warn('Already unmerged. Nothing to do.')
				return
			while len(self.merged_adapters) > 0:
				active_adapter = self.merged_adapters.pop()
				if active_adapter not in self.bola_w_p.keys():
					continue
				warnings.warn(
					'Unmerge lora module to 8-bit linear may get different generations due to rounding errors.'
				)
				delta_w = self.get_delta_weight(active_adapter)

				weight = self.get_base_layer().weight
				state = self.get_base_layer().state
				if state.SCB is None:
					state.SCB = weight.SCB
				output = dequantize_bnb_weight(weight, state=state)

				weight_data = output.to(delta_w.dtype).to(delta_w.device) - delta_w
				self.get_base_layer().weight = bnb.nn.Int8Params(
					weight_data.to('cpu'),
					requires_grad=False,
					has_fp16_weights=weight.has_fp16_weights,
				).to(weight.device)

				# TODO: use bias
				# if self.bola_bias[active_adapter] is not None:
				#     self.get_base_layer().bias.data -= self.bola_bias[active_adapter]
				state.reset_grads()

		# def edit_base_layer()

		def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
			B, S, _ = x.shape
			x = x.view(B * S, -1)

			if self.disable_adapters:
				if self.merged:
					self.unmerge()
				result = self.base_layer(x, *args, **kwargs)
			elif self.merged:
				result = self.base_layer(x, *args, **kwargs)
			else:
				result = self.base_layer(x, *args, **kwargs)
				for active_adapter in self.active_adapters:
					if active_adapter not in self.bola_w_p.keys():
						continue
					delta_w = self.get_delta_weight(active_adapter)
					dropout = self.dropout[active_adapter]

					requires_conversion = not torch.is_autocast_enabled()
					if requires_conversion:
						compute_dtype = delta_w.dtype
						if x.dtype != compute_dtype:
							x = x.to(compute_dtype)

					hidden = F.linear(x, delta_w)
					hidden = dropout(hidden)

					if requires_conversion:
						hidden = hidden.to(result.dtype)
					result = result + hidden
			result = result.view(B, S, -1).to(x)

			return result


if is_bnb_4bit_available():

	class Linear4bit(torch.nn.Module, BolaLayer):
		def __init__(
			self,
			base_layer: torch.nn.Module,
			adapter_name: str,
			num_in_blocks: int = 8,
			num_out_blocks: int = 8,
			top_k: int = 8,
			alpha: float = 1.0,
			dropout: float = 0.0,
			init_weights: Union[bool, str] = False,
			**kwargs,
		) -> None:
			super().__init__()
			BolaLayer.__init__(self, base_layer)
			self.fan_in_fan_out = False

			self._active_adapter = adapter_name
			self.update_layer(
				adapter_name,
				num_in_blocks=num_in_blocks,
				num_out_blocks=num_out_blocks,
				top_k=top_k,
				alpha=alpha,
				dropout=dropout,
				init_weights=init_weights,
			)

		def merge(
			self,
			safe_merge: bool = False,
			adapter_names: Optional[list[str]] = None,
		) -> None:
			adapter_names = check_adapters_to_merge(self, adapter_names)
			if not adapter_names:
				# no adapter to merge
				return

			for active_adapter in adapter_names:
				if active_adapter not in self.bola_w_p.keys():
					continue
				warnings.warn(
					'Merge lora module to 8-bit linear may get different generations due to rounding errors.'
				)
				delta_weight = self.get_delta_weight(active_adapter)
				weight = self.get_base_layer().weight
				state = self.get_base_layer().state
				if state.SCB is None:
					state.SCB = weight.SCB

				# Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
				# dequantization directly
				output = dequantize_bnb_weight(weight, state=state)
				weight_data = (
					output.to(delta_weight.dtype).to(delta_weight.device) + delta_weight
				)
				if safe_merge and not torch.isfinite(weight_data).all():
					raise ValueError(
						f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken'
					)

				self.get_base_layer().weight = bnb.nn.Int8Params(
					weight_data.to('cpu'),
					requires_grad=False,
					has_fp16_weights=weight.has_fp16_weights,
				).to(weight.device)

				# if self.bola_bias[active_adapter] is not None:
				#     bias_data = self.get_base_layer().bias.data + self.bola_bias[active_adapter]
				#     if safe_merge and not torch.isfinite(bias_data):
				#         raise ValueError(
				#             f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
				#         )
				#     self.get_base_layer().bias.data = bias_data

				state.reset_grads()
				self.merged_adapters.append(active_adapter)

		def unmerge(self) -> None:
			if not self.merged:
				warnings.warn('Already unmerged. Nothing to do.')
				return

			while len(self.merged_adapters) > 0:
				active_adapter = self.merged_adapters.pop()
				if active_adapter not in self.bola_w_p.keys():
					continue
				warnings.warn(
					'Unmerge lora module to 8-bit linear may get different generations due to rounding errors.'
				)
				delta_w = self.get_delta_weight(active_adapter)

				weight = self.get_base_layer().weight
				state = self.get_base_layer().state
				if state.SCB is None:
					state.SCB = weight.SCB
				output = dequantize_bnb_weight(weight, state=state)

				weight_data = output.to(delta_w.dtype).to(delta_w.device) - delta_w
				self.get_base_layer().weight = bnb.nn.Int8Params(
					weight_data.to('cpu'),
					requires_grad=False,
					has_fp16_weights=weight.has_fp16_weights,
				).to(weight.device)

				# if self.bola_bias[active_adapter] is not None:
				#     self.get_base_layer().bias.data -= self.bola_bias[active_adapter]
				state.reset_grads()

		def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
			B, S, _ = x.shape
			x = x.view(B * S, -1)

			if self.disable_adapters:
				if self.merged:
					self.unmerge()
				result = self.base_layer(x, *args, **kwargs)
			elif self.merged:
				result = self.base_layer(x, *args, **kwargs)
			else:
				result = self.base_layer(x, *args, **kwargs)
				# As per Tim Dettmers, for 4bit, we need to defensively clone here.
				# The reason is that in some cases, an error can occur that backprop
				# does not work on a manipulated view. This issue may be solved with
				# newer PyTorch versions but this would need extensive testing to be
				# sure.
				result = result.clone()
				for active_adapter in self.active_adapters:
					if active_adapter not in self.bola_w_p.keys():
						continue
					delta_w = self.get_delta_weight(active_adapter)
					dropout = self.dropout[active_adapter]

					requires_conversion = not torch.is_autocast_enabled()
					if requires_conversion:
						compute_dtype = delta_w.dtype
						if x.dtype != compute_dtype:
							x = x.to(compute_dtype)

					hidden = F.linear(x, delta_w)
					hidden = dropout(hidden)

					if requires_conversion:
						hidden = hidden.to(result.dtype)
					result = result + hidden
			result = result.view(B, S, -1).to(x)

			return result
