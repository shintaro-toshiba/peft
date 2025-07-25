from typing import List, Optional
import json
import argparse
import re
from functools import partial

import torch
from datasets import load_dataset
from peft import PeftModel, LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training, PeftConfig, PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)


def load_json_datasets(data_files):
    dataset = load_dataset(
        "json", 
        data_files=data_files, 
        split="train",
    )
    return dataset

def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    instruction = example["instruction"]
    text = (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n"
        f"{instruction}\n\n"
        f"### Response:\n" 
    ) # noqa: E501
    return text

def format_func(example, tokenizer):
    if tokenizer.chat_template is not None:
        messages = [
            {'role': "system",'content': "You are a helpful AI assistant."},
            {'role': "user",'content': example["instruction"]},
        ]
        return tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
        )
    else:
        return prepare_sample_text(example)

def update_dataset(example, tokenizer):
    text = format_func(example, tokenizer)
    return {"text": text}

REGEX_NUMBER=re.compile(r'-?\d+\.?\d*')
REGEX_LETTER=re.compile(r'A|B|C|D|E')

def extract_answer(output, regex=REGEX_NUMBER):
    output = output.strip()
    match = re.search(r"assistant\n\n([\s\S]+)|Response:\n([\s\S]+)", output)
    pred = match.group(0) if match else ""
    answers = re.findall(regex, pred)
    answer = answers[-1] if answers else "" # use the last number as an answer
    return answer
    
def evaluate_answer(output, gold, is_letter: bool):
    is_correct = False
    if not is_letter:
        answer = extract_answer(output, REGEX_NUMBER)
        try:
            gold = float(gold)
        except ValueError:
            raise ValueError("gold in not number")
        try: 
            answer = float(answer)
        except ValueError:
            answer = float("inf")
        if abs(gold - answer) <= 1e-4:
            is_correct = True
    else:
        answer = extract_answer(output, REGEX_LETTER)
        if gold == answer:
            is_correct = True
    return is_correct, answer

def get_arguments():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--peft-name-or-path', type=str, default=None)
    parser.add_argument('--base-name-or-path', type=str, required=True)
    parser.add_argument('--input-json-path', type=str, required=True)
    parser.add_argument('--output-json-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--is-letter-answer', action="store_true", default=False)
    args = parser.parse_args()
    return args

def main(args):
    print(f"{args=}")
    peft_name_or_path = args.peft_name_or_path
    base_name_or_path = args.base_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        base_name_or_path, 
        device_map="auto",
    )
    torch_dtype = model.dtype
    if peft_name_or_path is not None:
        model = PeftModel.from_pretrained(
            model,
            peft_name_or_path,
        ).to(torch_dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # NOTE: default="right" is issue of repetition: "assistant\n\nassistant\n\n"
    tokenizer.padding_side = "left"

    print(model)
    print(f"{model.dtype=}")
    
    dataset = load_json_datasets(args.input_json_path)
    dataset_func = partial(update_dataset, tokenizer=tokenizer)
    dataset = dataset.map(dataset_func, batched=False)

    results = []
    batch_size = args.batch_size
    num_batch = (len(dataset) + batch_size - 1) // batch_size
    for i in range(num_batch):    
        batch = dataset[i*batch_size:(i+1)*batch_size]
        inputs = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            do_sample=False,
        )
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(texts) 

    assert len(dataset) == len(results)
    num_correct = 0
    with open(args.output_json_path, 'w', encoding='utf-8') as file:
        for i in range(len(dataset)):
            is_correct, answer = evaluate_answer(
                output=results[i], 
                gold=dataset[i]["answer"],
                is_letter=args.is_letter_answer,
            )
            d = {
                "instruction": dataset[i]["instruction"], 
                "output": dataset[i]["output"], 
                "answer": dataset[i]["answer"], 
                "model_output": results[i],
                "model_answer": answer,
                "is_correct": int(is_correct),
            }
            json.dump(d, file, ensure_ascii=False)
            file.write('\n')
            num_correct += int(is_correct)
    print(f"accuracy: {num_correct / len(dataset)}")
            
if __name__ == "__main__":
    args = get_arguments()
    main(args)
