# BoLA: **B**l**o**ck-wise **L**ottery Ticket **A**daptation for Larage Language Models

BoLA (Block-wise Lottery Ticket Adaptation) is a sparse adaptation method that identifies and optimizes only a sparse parameters.

## QuickStart
### Setup Docker Container
Run docker container:
```sh
docker compose up -d --build
docker compose exec peft_cuda12 bash
```
If you can not use `docker compose`, please try this:
```sh
IMAGE_NAME=python-cuda12.1.1-cudnn8
CONTAINER_NAME=peft_cuda12
docker build . -f docker/peft-gpu-custom/Dockerfile -t $IMAGE_NAME
docker run $IMAGE_NAME $CONTAINER_NAME --gpus all -it --rm -d -v ./:/workspace -v /workspace/.venv 
docker attach $CONTAINER_NAME
```

### Usage of our method: BoLA
You can train a model with BoLA by wrapping the base model and PEFT configuration with `get_peft_model`:
```py
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, BolaConfig, TaskType

model_name_or_path = "elyza/Llama-3-ELYZA-JP-8B"
peft_config = BolaConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    num_in_blocks=64,
    num_out_blocks=64,
    top_k=16,
    alpha=4.0,
    dropout=0.1,
    target_modules=["q_proj","k_proj","v_proj","up_proj","down_proj"],
)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.save_pretrained("peft_path")
```

To load a PEFT model for inference:
```py
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_name_or_path = "elyza/Llama-3-ELYZA-JP-8B"
model = AutoPeftModelForCausalLM.from_pretrained("peft_apth").to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model.eval()
inputs = tokenizer("What is the highest mountain in the world?", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=64)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

## Advantages of BoLA

## Examples
- [Train RoBERTa-125M on GLEU with 4 GPUs](experiments/roberta_gleu/README.custom.md)



# Citation
```bibtex 
```