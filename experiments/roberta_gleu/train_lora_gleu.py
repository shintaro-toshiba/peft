import os
import torch
import re
import sys
from logging import getLogger, basicConfig, StreamHandler
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, utils as datasets_utils
from transformers import utils as transformers_utils
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    DataCollatorWithPadding,
    set_seed,
)
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from dataclasses import dataclass, field
import numpy as np
import evaluate
from typing import Optional, List
from transformers import TrainerCallback

logger = getLogger(__name__)

GLUE_PROPERTIES = {
    # taskname: (text_column1, text_column2, label_column)
    "cola": ("sentence", None, "label"),
    "mnli": ("premise", "hypothesis", "label"),
    "mrpc": ("sentence1", "sentence2", "label"),
    "qnli": ("question", "sentence", "label"),
    "qqp": ("question1", "question2", "label"),
    "rte": ("sentence1", "sentence2", "label"),
    "sst2": ("sentence", None, "label"),
    "stsb": ("sentence1", "sentence2", "label"),
    "wnli": ("sentence1", "sentence2", "label"),
}

class GPUMemoryCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) # GB
            logger.warning(f"Step {state.global_step}: Max GPU Memory Allocated: {max_memory:.2f} GB")
            torch.cuda.reset_peak_memory_stats()  # Reset

# fmt: off
@dataclass
class DatasetArguments:
    glue_task_name: str = field(
        metadata={"help": ""}
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": ""}
    )
    max_seq_length: int = field(
        default=512,
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

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="roberta-base", 
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
    use_lora: bool = field(
        default=False,
        metadata={"help": ""}
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": ""}
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={"help": ""}
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": ""}
    )
    lora_dropout: Optional[float] = field(
        default=None,
        metadata={"help": ""}
    )
    lora_target_modules: Optional[str] = field(
        default=None, # "query,value,key",
        metadata={"help": ""}
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": ""}
    )
    save_module_pattern: str = field(
        default=None,
        metadata={"help": ""}
    )
# fmt: on


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
    
    if torch.cuda.is_available():
        logger.info("CUDA is available. Proceeding with FlashAttention2.")
    else:
        logger.info("CUDA is not available. Please install CUDA drivers.")

    logger.info(f"{model_config=}")
    logger.info(f"{data_config=}")
    logger.info(f"{train_config=}")

    # seed
    set_seed(train_config.seed)

    # data
    datasets = load_dataset(
        "glue", 
        data_config.glue_task_name,
        cache_dir=data_config.data_cache_dir,
    )
    text_column_name1, text_column_name2, label_column_name = GLUE_PROPERTIES[
        data_config.glue_task_name
    ]
    num_labels = 1
    label_to_id = None
    is_regression = data_config.glue_task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features[label_column_name].names
        num_labels = len(label_list)
        label_to_id = {k: i for i, k in enumerate(label_list)}
    logger.debug(label_to_id)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        cache_dir=model_config.model_cache_dir,
        revision=model_config.model_revision,
        token=model_config.token,
        num_labels=num_labels,
    )  # all params are trainable
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        cache_dir=model_config.model_cache_dir,
        use_fast=model_config.use_fast,
        revision=model_config.model_revision,
        token=model_config.token,
    )

    if data_config.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length ({data_config.max_seq_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}). Please, use the max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_config.max_seq_length, tokenizer.model_max_length)

    if model_config.use_lora or model_config.use_dora:
        assert not (model_config.use_lora and model_config.use_dora), "cannot use both lora and dora at the same time."
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            target_modules=model_config.lora_target_modules.split(","),
            use_dora=model_config.use_dora,
        )
        model = get_peft_model(model, peft_config)
        logger.info("lora-fintune the specific parameters of model.")
        
    def get_number_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
    
    logger.info(f"model architecture:\n{model}")
    trainable_params, all_params = get_number_trainable_parameters(model)
    logger.info(
        f"trainable params {trainable_params:,d} || "
        f"all params: {all_params:,d} || "
        f"trainable%: {100 * trainable_params / all_params}"
    )

    padding = "max_length" if data_config.pad_to_max_length else False

    def preprocess(examples):
        text = (
            (examples[text_column_name1],)
            if text_column_name2 is None
            else (examples[text_column_name1], examples[text_column_name2])
        )
        tokenizer.truncation_side = "left"
        outputs = tokenizer(
            *text,
            # return_tensor="tp",
            truncation=True,
            max_length=max_seq_length,
            padding=padding,
        )
        # map labels to ids
        if label_to_id is not None and hasattr(examples, label_column_name):
            outputs["label"] = [
                (label_to_id[name] if name != -1 else -1)
                for name in examples[label_column_name]
            ]
        return outputs

    with train_config.main_process_first(desc="dataset map pre-processing"):
        load_from_cache_file = not data_config.overwrite_cache
        non_label_column_names = [
            name for name in datasets["train"].column_names if name != "label"
        ]
        datasets = datasets.map(
            preprocess,
            batched=True,
            load_from_cache_file=load_from_cache_file,
            remove_columns=non_label_column_names,
        )
        train_dataset = datasets["train"]
        valid_dataset = datasets[
            "validation"
            if data_config.glue_task_name != "mnli"
            else "validation_matched"
        ]
    logger.debug(f"{train_dataset.features}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    metric = evaluate.load("glue", data_config.glue_task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if is_regression:
            preds = np.squeeze(preds)
            results = metric.compute(predictions=preds, references=p.label_ids)
            results["mse"] = ((preds - p.label_ids) ** 2).mean().item()
        else:
            preds = np.argmax(preds, axis=-1)
            results = metric.compute(predictions=preds, references=p.label_ids)
        if len(results) > 1:
            results["mean_score"] = np.mean(list(results.values())).item()
        return results

    trainer = Trainer(
        model=model,
        # optimizers=(optimizer, lr_scheduler) # Will default to an instance of AdamW on your model and a scheduler given by get_linear_schedule_with_warmup()
        args=train_config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[GPUMemoryCallback()],
    )

    if train_config.do_train:
        logger.info("*-- Train --*")
        train_results = trainer.train()
        trainer.save_model()  # save model and tokenizer
        trainer.save_state()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)

    if train_config.do_eval:
        logger.info("*-- Evaluate --*")
        # NOTE: To reproduce the results of orignal paper, use validation dataset for evaluation. Reference: https://github.com/microsoft/LoRA/issues/31
        metrics = trainer.evaluate(eval_dataset=valid_dataset)
        metrics["samples"] = len(valid_dataset)
        trainer.log_metrics("valid", metrics)
        trainer.save_metrics("valid", metrics)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetArguments, TrainingArguments))
    model_config, data_config, train_config = parser.parse_args_into_dataclasses()
    main(model_config, data_config, train_config)
