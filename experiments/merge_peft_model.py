"""
reference: https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/merge_peft_adapter.py
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser


@dataclass
class ScriptArguments:
    """
    The input names representing the Adapter and Base model fine-tuned with PEFT, and the output name representing the
    merged model.
    """

    peft_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "the adapter name"}
    )
    base_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "the base model name"}
    )
    output_path: Optional[str] = field(
        default=None, 
        metadata={"help": "the merged model name"}
    )


def main(args):
    assert args.peft_name_or_path is not None, "please set `peft_name_or_path`"
    assert args.base_name_or_path is not None, "please set `base_name_or_path`"
    assert args.output_path is not None, "please set `output_path`"

    peft_config = PeftConfig.from_pretrained(args.peft_name_or_path)
    if peft_config.task_type == "SEQ_CLS":
        # The sequence classification task is used for the reward model in PPO
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.base_name_or_path
        )
    elif peft_config.task_type == "CAUSAL_LM":
        model = AutoModelForCausalLM.from_pretrained(
            script_args.base_name_or_path
        )
    else:
        raise ValueError("Unknow task type")

    # Load the PEFT model
    tokenizer = AutoTokenizer.from_pretrained(args.base_name_or_path)
    model = PeftModel.from_pretrained(model, args.peft_name_or_path)
    with torch.no_grad():
        model = model.merge_and_unload()

    model.save_pretrained(f"{args.output_path}")
    tokenizer.save_pretrained(f"{args.output_path}")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args)
