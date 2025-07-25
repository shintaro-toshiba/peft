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

import tempfile

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM

from peft import BolaConfig, BolaModel, PeftModel, TaskType, get_peft_model
from peft.tuners.bola import BolaLinear
from peft.utils import infer_device


class TestBolaLinear:
	device = infer_device()
	torch.manual_seed(0)

	def test_gradient_is_not_zero(self):
		module = nn.Linear(
			in_features=6,
			out_features=4,
		)
		module = BolaLinear(
			module,
			adapter_name='default',
			num_in_blocks=3,
			num_out_blocks=2,
			top_k=3,
			dropout=0.1,
		)

		# overwrite weights
		module.bola_w_p['default'].data = torch.randn_like(module.bola_w_p['default'])
		module.bola_w_v['default'].data = torch.randn_like(module.bola_w_v['default'])

		src = torch.randn(8, 6)  # (B, D)
		tgt = torch.randn(8, 4)  # (B, D)

		criterion = nn.MSELoss()

		outputs = module(src)
		loss = criterion(outputs, tgt)

		module.zero_grad()
		loss.backward()

		assert module.bola_w_p['default'].grad is not None
		assert module.bola_w_v['default'].grad is not None

		assert torch.all(module.bola_w_p['default'].grad != 0)
		assert torch.all(module.bola_w_v['default'].grad != 0)

	def test_get_delita_weight(self):
		module = nn.Linear(
			in_features=6,
			out_features=4,
		)
		module = BolaLinear(
			module,
			adapter_name='default',
			num_in_blocks=3,
			num_out_blocks=2,
			top_k=3,
			dropout=0.1,
		)
		with torch.no_grad():
			module.bola_w_v['default'].data = torch.tensor(
				[[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]]
			).float()
		indices = torch.tensor([0, 4, 5])
		weight = module.get_delta_weight('default', indices)
		expect = torch.tensor(
			[
				[1, 1, 0, 0, 0, 0],
				[1, 1, 0, 0, 0, 0],
				[0, 0, 2, 2, 3, 3],
				[0, 0, 2, 2, 3, 3],
			]
		).float()
		atol, rtol = 1e-5, 1e-8
		assert list(weight.shape) == list(expect.shape)
		assert torch.allclose(expect, weight, atol=atol, rtol=rtol)


class TestBolaModel:
	device = infer_device()
	torch.manual_seed(0)

	def test_config(self):
		config = BolaConfig(num_in_blocks=2, num_out_blocks=2, top_k=2, dropout=0.1)
		with tempfile.TemporaryDirectory() as tmp_dir:
			config.save_pretrained(tmp_dir)

			config_loaded = BolaConfig.from_pretrained(tmp_dir)
			assert config.to_dict() == config_loaded.to_dict()

	def test_state_dict(self):
		inputs = torch.arange(10).view(-1, 1).to(self.device)

		model_id = 'hf-internal-testing/tiny-random-OPTForCausalLM'
		model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
		with torch.no_grad():
			output_base = model(inputs).logits
		config = BolaConfig(
			task_type=TaskType.CAUSAL_LM,
			num_in_blocks=2,
			num_out_blocks=2,
			top_k=2,
		)
		model = get_peft_model(model, config)
		model.eval()
		assert isinstance(model.base_model, BolaModel)
		output_peft = model(inputs).logits
		atol, rtol = 1e-5, 1e-8
		# sanity check: init BolaModel dose not change the output
		assert list(output_base.shape) == list(output_peft.shape)
		assert torch.allclose(output_base, output_peft, atol=atol, rtol=rtol)

		def reinitialize_weights(model) -> None:
			for module in model.modules():
				if isinstance(module, BolaLinear):
					nn.init.normal_(module.bola_w_p['default'], mean=0, std=0.1)
					nn.init.normal_(module.bola_w_v['default'], mean=0, std=0.1)

		reinitialize_weights(model)

		output_peft = model(inputs).logits

		with tempfile.TemporaryDirectory() as tmp_dir:
			model.save_pretrained(tmp_dir)
			state_dict = load_file(f'{tmp_dir}/adapter_model.safetensors')
			assert any(['bola_' in key for key in state_dict])

			model_loaded = AutoModelForCausalLM.from_pretrained(model_id).to(
				self.device
			)
			model_loaded = PeftModel.from_pretrained(
				model_loaded, tmp_dir, is_trainable=True
			)
			config_loaded = model_loaded.peft_config['default']

		assert config.to_dict() == config_loaded.to_dict()

		output_loaded = model_loaded(inputs).logits
		assert torch.allclose(output_peft, output_loaded, atol=atol, rtol=rtol)

		model = model.merge_and_unload(safe_merge=True)
		output_merged = model(inputs).logits
		assert torch.allclose(output_peft, output_merged, atol=atol, rtol=rtol)


# class TestBola8bitModelS:
#     device = infer_device()
#     torch.manual_seed(0)

#     def test_state_dict(self):
#         inputs = torch.arange(10).view(-1, 1).to(self.device)

#         model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
#         bnb_config = BitsAndBytesConfig(load_in_8bit=True)
#         model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             quantization_config=bnb_config,
#             torch_dtype=torch.float32,
#         )
#         with torch.no_grad():
#             output_base = model(inputs).logits

#         config = BolaConfigS()
#         model = get_peft_model(model, config)
#         with torch.no_grad():
#             output_peft = model(inputs).logits

#         atol, rtol = 1e-5, 1e-8
#         # sanity check: init BolaModel dose not change the output
#         assert list(output_base.shape) == list(output_peft.shape)
#         assert torch.allclose(output_base, output_peft, atol=atol, rtol=rtol)
