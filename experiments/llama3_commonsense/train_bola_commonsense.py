import os
import torch
import re
import sys
from functools import partial
from logging import getLogger, basicConfig, StreamHandler
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, utils as datasets_utils
from transformers import utils as transformers_utils
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    # TrainingArguments,
    DataCollatorForSeq2Seq,
)
from trl import SFTTrainer, SFTConfig as TrainingArguments
from trl.trainer import ConstantLengthDataset
from peft import PeftModel, BolaConfig, TaskType, get_peft_model
from dataclasses import dataclass, field
import numpy as np
from typing import Optional, List


logger = getLogger(__name__)


# fmt: off
@dataclass
class DatasetArguments:
    json_data_files: Optional[str] = field(
        default=None,
        metadata={"help": ""}
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": ""}
    )
    # max_seq_length: int = field(
    #     default=512,
    #     metadata={"help": ""}
    # )
    valid_set_size: int = field(
        default=100,
        metadata={"help": ""}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": ""}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={"help": ""}
    )
    use_constat_length_dataset: Optional[bool] = field(
        default=True, 
        metadata={"help": "whether to use pack"}
    )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="llama3", 
        metadata={"help": "model path or model identifier from huggingface.co/models"}
    )
    model_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": ""}
    )
    token: bool = field(
        default=False,
        metadata={"help": ""}
    )
    use_fast: bool = field(
        default=True,
        metadata={"help": ""}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": ""}
    )
    module_pattern: Optional[str] = field(
        default=None, 
        metadata={"help": ""}
    )
    # bitandbytes
    load_in_8bit: Optional[bool] = field(
        default=False, 
        metadata={"help": "load the model in 8 bits precision"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, 
        metadata={"help": "load the model in 4 bits precision"}
    )
    # lora
    use_bola: bool = field(
        default=False,
        metadata={"help": ""}
    )
    bola_num_in_blocks: Optional[int] = field(
        default=None,
        metadata={"help": ""}
    )
    bola_num_out_blocks: Optional[int] = field(
        default=None,
        metadata={"help": ""}
    )
    bola_top_k: Optional[int] = field(
        default=None,
        metadata={"help": ""}
    )
    bola_alpha: Optional[float] = field(
        default=None,
        metadata={"help": ""}
    )
    bola_dropout: Optional[float] = field(
        default=None,
        metadata={"help": ""}
    )
    bola_target_modules: Optional[str] = field(
        default=None, # "query,value,key",
        metadata={"help": ""}
    )
    oues_path: Optional[str] = field(
        default=None,
        metadata={"help": ""}
    )
    save_module_pattern: str = field(
        default=None,
        metadata={"help": ""}
    )
# fmt: on

def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    instruction = example["instruction"]
    output = example["output"]
    match = re.search(r":\s*(.*)", output)
    if match:
        output = match.group(1).strip()
    text = text = (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n"
        f"{instruction}\n\n"
        f"### Response:\n"
        f"{output}"
    ) # noqa: E501
    return text

def format_func(example, tokenizer):
    if tokenizer.chat_template is not None:
        instruction = example["instruction"]
        output = example["output"]
        match = re.search(r":\s*(.*)", output)
        if match:
            output = match.group(1).strip()
        messages = [
            {'role': "system",'content': "You are a helpful AI assistant."},
            {'role': "user",'content': instruction},
            {'role': "assistant",'content': output},
        ]
        return tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
        )
    else:
        return prepare_sample_text(example)

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    _format_func = partial(format_func, tokenizer=tokenizer)
    total_characters, total_tokens = 0, 0
    for _, example in zip(range(nb_examples), iter(dataset)):
        text = _format_func(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))
    return total_characters / total_tokens

def main(model_config, data_config, train_config):
    # setup logger
    basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[StreamHandler(sys.stdout)],
    )
    logger_level = train_config.get_process_log_level()
    logger.setLevel(logger_level)
    # if train_config.should_log:
    #     # The default of train_config.log_level is passive, so we set log level at info here to have that default.
    #     transformers_utils.logging.set_verbosity_info()
    # datasets_utils.logging.set_verbosity(logger_level)
    # transformers_utils.logging.set_verbosity(logger_level)
    # transformers_utils.logging.enable_default_handler()
    # transformers_utils.logging.enable_explicit_format()

    logger.info(f"{model_config=}")
    logger.info(f"{data_config=}")
    logger.info(f"{train_config=}")

    # seed
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)

    quantization_config = None
    torch_dtype = torch.bfloat16
    if model_config.load_in_8bit and model_config.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif model_config.load_in_8bit or model_config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=model_config.load_in_8bit, 
            load_in_4bit=model_config.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        cache_dir=model_config.model_cache_dir,
        revision=model_config.model_revision,
        token=model_config.token,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )  # all params are trainable
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        cache_dir=model_config.model_cache_dir,
        use_fast=model_config.use_fast,
        revision=model_config.model_revision,
        token=model_config.token,
        # model_max_length=model_config.model_max_length,
    )

    if train_config.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length ({train_config.max_seq_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}). Please, use the max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(train_config.max_seq_length, tokenizer.model_max_length)

    if model_config.use_bola:
        peft_config = BolaConfig(
            task_type=TaskType.CAUSAL_LM,
            num_in_blocks=model_config.bola_num_in_blocks,
            num_out_blocks=model_config.bola_num_out_blocks,
            top_k=model_config.bola_top_k,
            alpha=model_config.bola_alpha,
            dropout=model_config.bola_dropout,
            target_modules=model_config.bola_target_modules.split(","),
        )
        model = get_peft_model(model, peft_config).to(torch_dtype)
        logger.info("bola-fintune the specific parameters of model.")

    logger.info(f"model architecture:\n{model}")
    trainable_params, all_params = model.get_nb_trainable_parameters()
    logger.info(
        f"trainable params {trainable_params:,d} || "
        f"all params: {all_params:,d} || "
        f"trainable%: {100 * trainable_params / all_params}"
    )

    padding = "max_length" if data_config.pad_to_max_length else False
        
    # data
    datasets = load_dataset(
        "json", # data_config.dataset_name, 
        data_files=data_config.json_data_files.split(","),
    )
    if data_config.valid_set_size > 0:
        train_valid = datasets["train"].train_test_split(
            test_size=data_config.valid_set_size, shuffle=True, seed=42
        )
        train_data = train_valid["train"].shuffle()
        valid_data = train_valid["test"].shuffle()
    else:
        train_data = datasets["train"].shuffle()
        valid_data = None

    chars_per_token = chars_token_ratio(datasets["train"], tokenizer)
    logger.info(f"the character to token ratio of the dataset is: {chars_per_token:.2f}")
    
    def _format_func(example):
        return format_func(example, tokenizer=tokenizer)

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=_format_func,
        infinite=True,
        seq_length=max_seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=_format_func,
        infinite=False,
        seq_length=max_seq_length,
        chars_per_token=chars_per_token,
    )
    
    trainer = SFTTrainer(
        model=model,
        # optimizers=(optimizer, lr_scheduler) # Will default to an instance of AdamW on your model and a scheduler given by get_linear_schedule_with_warmup()
        args=train_config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # tokenizer=tokenizer,
        # data_collator=DataCollatorForSeq2Seq(
        #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=False
        # ),
        packing=True,
    )

    logger.info("*-- Train --*")
    train_results = trainer.train()
    trainer.save_model()  # save model and tokenizer
    trainer.save_state()

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetArguments, TrainingArguments))
    model_config, data_config, train_config = parser.parse_args_into_dataclasses()
    main(model_config, data_config, train_config)
