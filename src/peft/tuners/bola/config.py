from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class BolaConfig(PeftConfig):
	# fmt: off
	num_in_blocks: Optional[int] = field(
        default=None, 
        metadata={"help": ""}
    )
	num_out_blocks: Optional[int] = field(
        default=None, 
        metadata={"help": ""}
    )
	top_k: int = field(
        default=1,
        metadata={"help": ""}
    )
	alpha: float = field(
        default=1.0, 
        metadata={"help": "Alpha"}
    )
	dropout: float = field(
        default=0.0, 
        metadata={"help": ""}
    )
	fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
	target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with FourierFT."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )
	exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from fourierft."},
    )
	bias: str = field(
        default="none",
        metadata={"help": "Bias type for FourierFT. Can be 'none', 'all' or 'fourier_only'."}
    )
	modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from FourierFT layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
	init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the BOFT layers with their default initialization. Don't change ",
                "this setting, except if you know exactly what you're doing.",
            ),
        },
    )
	layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers"
                " indexes that are specified inside this list. If a single integer is passed, PEFT will transform only"
                " the layer at this index."
            )
        },
    )
	layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer"
                " pattern is not in the common layers pattern. This should target the `nn.ModuleList` of the "
                "model, which is often called `'layers'` or `'h'`."
            )
        },
    )
	num_in_blocks_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
	num_out_blocks_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
	top_k_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
            )
        },
    )
	alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
            )
        },
    )
	# fmt: on

	def __post_init__(self):
		super().__post_init__()
		self.peft_type = PeftType.BOLA
		self.target_modules = (
			set(self.target_modules)
			if isinstance(self.target_modules, list)
			else self.target_modules
		)
		self.exclude_modules = (
			set(self.exclude_modules)
			if isinstance(self.exclude_modules, list)
			else self.exclude_modules
		)
		# if target_modules is a regex expression, then layers_to_transform should be None
		if (
			isinstance(self.target_modules, str)
			and self.layers_to_transform is not None
		):
			raise ValueError(
				'`layers_to_transform` cannot be used when `target_modules` is a str.'
			)

		# if target_modules is a regex expression, then layers_pattern should be None
		if isinstance(self.target_modules, str) and self.layers_pattern is not None:
			raise ValueError(
				'`layers_pattern` cannot be used when `target_modules` is a str.'
			)
		# check for layers_to_transform and layers_pattern
		if self.layers_pattern and not self.layers_to_transform:
			raise ValueError(
				'When `layers_pattern` is specified, `layers_to_transform` must also be specified. '
			)
