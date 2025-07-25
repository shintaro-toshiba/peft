# Copyright 2024-present the HuggingFace Inc. team.
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

import re
import warnings
from dataclasses import asdict
from enum import Enum
from itertools import chain
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
	BaseTuner,
	BaseTunerLayer,
	check_target_module_exists,
)
from peft.utils import (
	TRANSFORMERS_MODELS_TO_BOLA_TARGET_MODULES_MAPPING,
	ModulesToSaveWrapper,
	_get_submodules,
)

from .config import BolaConfig
from .layer import BolaLayer, BolaLinear


class BolaModel(BaseTuner):
	prefix: str = 'bola'

	def __init__(
		self,
		model: nn.Module,
		config: BolaConfig,
		adapter_name: str,
		low_cpu_mem_usage: bool = False,
	) -> None:
		super().__init__(
			model,
			config,
			adapter_name,
			low_cpu_mem_usage=low_cpu_mem_usage,
		)

	def _check_new_adapter_config(self, config: BolaConfig) -> None:
		if (len(self.peft_config) > 1) and (config.bias != 'none'):
			raise ValueError(
				f'{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, '
				"set bias to 'none' for all adapters."
			)

	@staticmethod
	def _check_target_module_exists(bola_config, key):
		return check_target_module_exists(bola_config, key)

	def _create_and_replace(
		self,
		config,
		adapter_name,
		target,
		target_name,
		parent,
		current_key,
		**optional_kwargs,
	):
		if current_key is None:
			raise ValueError("Current Key shouldn't be `None`")
		# Regexp matching - Find key which matches current target_name in patterns provided
		pattern_keys = list(chain(config.num_in_blocks_pattern.keys()))
		target_name_key = next(
			filter(lambda key: re.match(rf'.*\.{key}$', current_key), pattern_keys),
			current_key,
		)

		num_in_blocks = config.num_in_blocks_pattern.get(
			target_name_key,
			config.num_in_blocks,
		)
		num_out_blocks = config.num_out_blocks_pattern.get(
			target_name_key,
			config.num_out_blocks,
		)
		top_k = config.top_k
		alpha = config.alpha
		dropout = config.dropout
		bias = hasattr(target, 'bias') and target.bias is not None
		kwargs = {
			'num_in_blocks': num_in_blocks,
			'num_out_blocks': num_out_blocks,
			'top_k': top_k,
			'alpha': alpha,
			'dropout': dropout,
			'fan_in_fan_out': config.fan_in_fan_out,
			'init_weights': config.init_weights,
			'loaded_in_8bit': getattr(self.model, 'is_loaded_in_8bit', False),
			'loaded_in_4bit': getattr(self.model, 'is_loaded_in_4bit', False),
		}
		kwargs['bias'] = bias
		if isinstance(target, BolaLayer):
			target.update_layer(
				adapter_name=adapter_name,
				num_in_blocks=num_in_blocks,
				num_out_blocks=num_out_blocks,
				top_k=top_k,
				alpha=alpha,
				dropout=dropout,
				init_weights=config.init_weights,
			)
		else:
			new_module = self._create_new_module(config, adapter_name, target, **kwargs)
			if adapter_name != self.active_adapter:
				# adding an additional adapter: it is not automatically trainable
				new_module.requires_grad_(False)
			self._replace_module(parent, target_name, new_module, target)

	def _replace_module(self, parent, child_name, new_module, child):
		setattr(parent, child_name, new_module)
		# It's not necessary to set requires_grad here, as that is handled by
		# _mark_only_adapters_as_trainable

		# child layer wraps the original module, unpack it
		if hasattr(child, 'base_layer'):
			child = child.base_layer

		if not hasattr(new_module, 'base_layer'):
			new_module.weight = child.weight
			if hasattr(child, 'bias'):
				new_module.bias = child.bias

		if getattr(child, 'state', None) is not None:
			if hasattr(new_module, 'base_layer'):
				new_module.base_layer.state = child.state
			else:
				new_module.state = child.state
			new_module.to(child.weight.device)

		meta = torch.device('meta')
		# dispatch to correct device
		for name, module in new_module.named_modules():
			if 'bola' in name:
				if not any(p.device == meta for p in module.parameters()):
					module.to(child.weight.device)

	def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
		for n, p in model.named_parameters():
			if self.prefix not in n:
				p.requires_grad = False

		for active_adapter in self.active_adapters:
			bias = self.peft_config[active_adapter].bias
			if bias == 'none':
				continue

			if bias == 'all':
				for n, p in model.named_parameters():
					if 'bias' in n:
						p.requires_grad = True
			elif bias == 'bola_only':
				for m in model.modules():
					if (
						isinstance(m, BolaLayer)
						and hasattr(m, 'bias')
						and m.bias is not None
					):
						m.bias.requires_grad = True
			else:
				raise NotImplementedError(
					f'Requested bias: {bias}, is not implemented.'
				)

	@staticmethod
	def _create_new_module(config, adapter_name, target, **kwargs):
		# avoid eager bnb import
		if is_bnb_available():
			import bitsandbytes as bnb

			from .bnb import Linear8bitLt

		if is_bnb_4bit_available():
			from .bnb import Linear4bit

		loaded_in_8bit = kwargs.get('loaded_in_8bit', False)
		loaded_in_4bit = kwargs.get('loaded_in_4bit', False)

		if isinstance(target, BaseTunerLayer):
			target_base_layer = target.get_base_layer()
		else:
			target_base_layer = target

		if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
			eightbit_kwargs = kwargs.copy()
			eightbit_kwargs.update(
				{
					'has_fp16_weights': target_base_layer.state.has_fp16_weights,
					'threshold': target_base_layer.state.threshold,
					'index': target_base_layer.index,
				}
			)
			return Linear8bitLt(target, adapter_name, **eightbit_kwargs)
		elif loaded_in_4bit and isinstance(target_base_layer, bnb.nn.Linear4bit):
			fourbit_kwargs = kwargs.copy()
			fourbit_kwargs.update(
				{
					'compute_dtype': target_base_layer.compute_dtype,
					'compress_statistics': target_base_layer.weight.compress_statistics,
					'quant_type': target_base_layer.weight.quant_type,
				}
			)
			return Linear4bit(target, adapter_name, **fourbit_kwargs)
		elif isinstance(target_base_layer, nn.Linear):
			if kwargs['fan_in_fan_out']:
				warnings.warn(
					'fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. '
					'Setting fan_in_fan_out to False.'
				)
				kwargs['fan_in_fan_out'] = config.fan_in_fan_out = False
			new_module = BolaLinear(target, adapter_name, **kwargs)
		else:
			raise ValueError(
				f'Target module {target} is not supported. Currently, only the following modules are supported: '
				'`nn.Linear`.'
			)
		return new_module

	def __getattr__(self, name: str):
		"""Forward missing attributes to the wrapped module."""
		try:
			return super().__getattr__(name)  # defer to nn.Module's logic
		except AttributeError:
			if name == 'model':
				raise
			return getattr(self.model, name)

	def get_peft_config_as_dict(self, inference: bool = False):
		config_dict = {}
		for key, value in self.peft_config.items():
			config = {
				k: v.value if isinstance(v, Enum) else v
				for k, v in asdict(value).items()
			}
			if inference:
				config['inference_mode'] = True
		config_dict[key] = config
		return config

	def _set_adapter_layers(self, enabled: bool = True) -> None:
		for module in self.model.modules():
			if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
				module.enable_adapters(enabled)

	def enable_adapter_layers(self) -> None:
		"""Enable all adapters.

		Call this if you have previously disabled all adapters and want to re-enable them.
		"""
		self._set_adapter_layers(enabled=True)

	def disable_adapter_layers(self) -> None:
		"""Disable all adapters.

		When disabling all adapters, the model output corresponds to the output of the base model.
		"""
		for active_adapter in self.active_adapters:
			val = self.peft_config[active_adapter].bias
			if val != 'none':
				msg = (
					f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
					'output as the the base model would without adaption.'
				)
				warnings.warn(msg)
		self._set_adapter_layers(enabled=False)

	def set_adapter(self, adapter_name: str | list[str]) -> None:
		"""Set the active adapter(s).

		Args:
		    adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
		"""
		for module in self.model.modules():
			if isinstance(module, BolaLayer):
				if module.merged:
					warnings.warn(
						'Adapter cannot be set when the model is merged. Unmerging the model first.'
					)
					module.unmerge()
				module.set_adapter(adapter_name)
		self.active_adapter = adapter_name

	@staticmethod
	def _prepare_adapter_config(peft_config, model_config):
		if peft_config.target_modules is None:
			if (
				model_config['model_type']
				not in TRANSFORMERS_MODELS_TO_BOLA_TARGET_MODULES_MAPPING
			):
				raise ValueError('Please specify `target_modules` in `peft_config`')
			peft_config.target_modules = set(
				TRANSFORMERS_MODELS_TO_BOLA_TARGET_MODULES_MAPPING[
					model_config['model_type']
				]
			)
		return peft_config

	def _unload_and_optionally_merge(
		self,
		merge=True,
		progressbar: bool = False,
		safe_merge: bool = False,
		adapter_names: Optional[list[str]] = None,
	):
		key_list = [
			key for key, _ in self.model.named_modules() if self.prefix not in key
		]
		desc = 'Unloading ' + ('and merging ' if merge else '') + 'model'
		for key in tqdm(key_list, disable=not progressbar, desc=desc):
			try:
				parent, target, target_name = _get_submodules(self.model, key)
			except AttributeError:
				continue

			if hasattr(target, 'base_layer'):
				if merge:
					target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
				self._replace_module(
					parent, target_name, target.get_base_layer(), target
				)
			elif isinstance(target, ModulesToSaveWrapper):
				# save any additional trainable modules part of `modules_to_save`
				setattr(
					parent, target_name, target.modules_to_save[target.active_adapter]
				)

		return self.model

	def delete_adapter(self, adapter_name: str):
		"""
		Deletes an existing adapter.

		Args:
		    adapter_name (str): Name of the adapter to be deleted.
		"""
		if adapter_name not in list(self.peft_config.keys()):
			raise ValueError(f'Adapter {adapter_name} does not exist')
		del self.peft_config[adapter_name]

		# we cannot use self.prefix as we want to include non-trainable fourierft parameters
		key_list = [key for key, _ in self.model.named_modules() if 'bola' not in key]
		new_adapter = None
		for key in key_list:
			_, target, _ = _get_submodules(self.model, key)
			if isinstance(target, BolaLayer):
				target.delete_adapter(adapter_name)
				if new_adapter is None:
					new_adapter = target.active_adapter[:]

		self.active_adapter = new_adapter or []

	def merge_and_unload(
		self,
		progressbar: bool = False,
		safe_merge: bool = False,
		adapter_names: Optional[list[str]] = None,
	) -> torch.nn.Module:
		r"""
		This method merges the Fourier layers into the base model. This is needed if someone wants to use the base
		model as a standalone model.

		Args:
		    progressbar (`bool`):
		        whether to show a progressbar indicating the unload and merge process
		    safe_merge (`bool`):
		        whether to activate the safe merging check to check if there is any potential Nan in the adapter
		        weights
		    adapter_names (`list[str]`, *optional*):
		        The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
		        to `None`.
		"""
		return self._unload_and_optionally_merge(
			progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
		)

	def unload(self) -> nn.Module:
		"""
		Gets back the base model by removing all the Fourier modules without merging. This gives back the original base
		model.
		"""
		return self._unload_and_optionally_merge(merge=False)
