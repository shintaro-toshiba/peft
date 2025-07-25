from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose


def merge_score(scores, indices, alpha):
    """
    tensor: [k, n x m]
    returns: [n x m]
    - Each position picks the max value among k candidates (hard selection)
    - Uses straight-through estimator to allow gradient flow to all k values
    """
    mask = F.one_hot(indices, num_classes=scores.shape[-1]).to(scores.dtype)
    mask = mask * (alpha - 1) + torch.ones_like(mask)
    out = (scores * mask.detach()).sum(dim=0)  # (nxm)
    return out

class StraightThroughEstimator(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		return x

	@staticmethod
	def backward(ctx, g):
		return g

class MultiplyWithStraightThroughEstimator(torch.autograd.Function):
	@staticmethod
	def forward(ctx, bola_weight, score, indices):
		"""Forward pass of MultiplyWithStraightThroughEstimator.
		Args:
		    bola_weight (torch.Tensor): Sparse Tensor of shape (B_out x B_in, C_out, C_in)
		    score (torch.Tensor): Score Tensor of shape (B_out x B_in, 1, 1)
		    indices (torch.Tensor): Tensor which indicate sparse locations (K)
		    scaler (float): The current scalar value used for scaling weights.
		"""
		ctx.save_for_backward(bola_weight, score, indices)
		return bola_weight * score

	@staticmethod
	def backward(ctx, g):
		bola_weight, score, indices = ctx.saved_tensors
  
		g_weight = g * score
  
		ones = torch.ones_like(bola_weight) # (B_o*B_i, D, D)
		ones[indices] = bola_weight[indices]
		g_score = g * ones
  
		return g_weight, g_score, None


class BolaLayer(BaseTunerLayer):
	def __init__(
		self,
		base_layer: nn.Module,
		**kwargs,
	):
		super().__init__()

		self.base_layer = base_layer
		self.num_in_blocks = {}
		self.num_out_blocks = {}
		self.top_k = {}
		self.alpha = {}
		self.dropout = nn.ModuleDict({})
		self.bola_w_p = nn.ParameterDict({})
		self.bola_w_v = nn.ParameterDict({})
		self._disable_adapters = False
		self.merged_adapters = []
		self.kwargs = kwargs

		base_layer = self.get_base_layer()
		if isinstance(base_layer, nn.Linear):
			self.in_features = base_layer.in_features
			self.out_features = base_layer.out_features
		else:
			raise ValueError(f'`nn.Linear` is only supported')

	def update_layer(
		self,
		adapter_name: str,
		num_in_blocks: Optional[int],
		num_out_blocks: Optional[int],
		top_k: int,
		alpha: float = 1.0,
		dropout: float = 0.0,
		init_weights: bool = False,
	):
		if num_in_blocks is None:
			num_in_blocks = self.in_features
		if num_out_blocks is None:
			num_out_blocks = self.out_features

		assert self.in_features % num_in_blocks == 0
		assert self.out_features % num_out_blocks == 0
		assert top_k <= num_in_blocks * num_out_blocks

		self.num_in_blocks[adapter_name] = num_in_blocks
		self.num_out_blocks[adapter_name] = num_out_blocks
		self.top_k[adapter_name] = top_k
		self.alpha[adapter_name] = alpha

		in_block_features = self.in_features // num_in_blocks
		out_block_features = self.out_features // num_out_blocks
		self.in_block_features = in_block_features
		self.out_block_features = out_block_features

		dtype = self.get_base_layer().weight.dtype
		device = self.get_base_layer().weight.device
		self.bola_w_p[adapter_name] = nn.Parameter(
			torch.empty((top_k, num_out_blocks * num_in_blocks), dtype=dtype, device=device), 
   			requires_grad=True,
		)
		self.bola_w_v[adapter_name] = nn.Parameter(
			torch.zeros((top_k, out_block_features, in_block_features), dtype=dtype, device=device),
			requires_grad=True,
		)

		if dropout > 0.0:
			dropout_layer = nn.Dropout(p=dropout)
		else:
			dropout_layer = nn.Identity()
		self.dropout[adapter_name] = dropout_layer

		if init_weights:
			self.reset_parameters(adapter_name)

		self._move_adapter_to_device_of_base_layer(adapter_name)
		self.set_adapter(self.active_adapters)

	@torch.no_grad()
	def reset_parameters(self, adapter_name):
		import math
		if adapter_name in self.bola_w_p.keys():
			with torch.no_grad():
				nn.init.kaiming_uniform_(
					self.bola_w_p[adapter_name],
					a=math.sqrt(5.0),
				)
				nn.init.zeros_(self.bola_w_v[adapter_name])

	def get_weight_norm(
		self,
		weight: torch.Tensor,
		dim: Optional[int | tuple[int]] = None,
		keepdim: bool = False,
		epsilon: float = 1e-8,
	):
		# calculate L2 norm of weight matrix
		return (torch.sum(weight ** 2, dim=dim, keepdim=keepdim) + epsilon) ** 0.5

	# TODO: make sparse gradients
	def _build_delta_weight(self, size, index, source, dim):
		delta_weight = torch.zeros(*size, dtype=source.dtype, device=source.device)
		delta_weight.index_add_(dim=dim, index=index, source=source)
		return delta_weight

	def get_delta_weight(self, adapter, indices = None) -> torch.Tensor:
		w_v = self.bola_w_v[adapter]
		w_p = self.bola_w_p[adapter]
		scaler = self.alpha[adapter]
		num_out_blocks = self.num_out_blocks[adapter]
		num_in_blocks = self.num_in_blocks[adapter]
  
		if indices is None:
			indices = torch.argmax(w_p.detach(), dim=-1)

		shape = (
      		num_out_blocks*num_in_blocks, 
    		self.out_block_features, 
      		self.in_block_features,
    	)
		delta_weight = self._build_delta_weight(shape, indices, w_v, dim=0)
  
		magnitude = merge_score(w_p, indices, scaler).view(-1, 1, 1)
		delta_weight = MultiplyWithStraightThroughEstimator.apply(
			delta_weight, magnitude, indices
		) # (nm, d/n, d/m)
  
		delta_weight = (
			delta_weight
   			.view(
				num_out_blocks,
				num_in_blocks,
				self.out_block_features,
				self.in_block_features,
			)
			.permute(0, 2, 1, 3)
			.contiguous()
   			.view(
				self.out_features,
				self.in_features,
			)
		)  # (D_o, D_i)

		return delta_weight


class BolaLinear(nn.Module, BolaLayer):
	def __init__(
		self,
		base_layer: nn.Module,
		adapter_name: str,
		num_in_blocks: int = 8,
		num_out_blocks: int = 8,
		alpha: float = 1.0,
		top_k: int = 8,
		dropout: float = 0.0,
		init_weights: Union[bool, str] = False,
		**kwargs,
	) -> None:
		super().__init__()
		BolaLayer.__init__(self, base_layer, **kwargs)

		self._active_adapter = adapter_name
		self.update_layer(
			adapter_name=adapter_name,
			num_in_blocks=num_in_blocks,
			num_out_blocks=num_out_blocks,
			top_k=top_k,
			alpha=alpha,
			dropout=dropout,
			init_weights=init_weights,
		)

	def __repr__(self) -> str:
		return 'bola.' + super().__repr__()

	def merge_safe(
		self,
		module: nn.Linear,
		active_adapter: str,
	):
		orig_weight = module.weight.data.clone()
		delta_weight = self.get_delta_weight(active_adapter)
		new_weight = orig_weight + delta_weight.to(orig_weight)
		return new_weight

	def merge_unsafe(
		self,
		module: nn.Linear,
		active_adapter: str,
	):
		raise NotImplementedError('not implemented ``merge_unsafe``')

	@torch.no_grad()
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
			if active_adapter in self.bola_w_p.keys():
				module = self.get_base_layer()
				if safe_merge:
					# Note that safe_merge will be slower than the normal merge
					# because of the copy operation.
					new_weight = self.merge_safe(module, active_adapter)
					if not torch.isfinite(new_weight).all():
						raise ValueError(
							f'NaNs detected in the merged weight. The adapter {active_adapter} seems to be broken'
						)
					module.weight.data = new_weight
				else:
					self.merge_unsafe(module, active_adapter)
				self.merged_adapters.append(active_adapter)

	@torch.no_grad()
	def unmerge(self) -> None:
		if not self.merged:
			warnings.warn('Already unmerged. Nothing to do.')
			return

		while len(self.merged_adapters) > 0:
			active_adapter = self.merged_adapters.pop()
			if active_adapter in self.bola_w_p.keys():
				weight = self.get_base_layer().weight
				orig_dtype = weight.dtype
				delta_weight = self.get_delta_weight(active_adapter)
				delta_weight = delta_weight.to(orig_dtype)
				weight.data -= delta_weight

	def get_delta_weight(self, adapter, indices = None) -> torch.Tensor:
		return super().get_delta_weight(adapter, indices)

	def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
		shape = x.shape
		x = x.view(-1, shape[-1])
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
				dropout = self.dropout[active_adapter]
				delta_weight = self.get_delta_weight(active_adapter)
				result = result + F.linear(dropout(x), delta_weight)
		result = result.view(*shape[:-1], -1).to(x)
		return result
