import copy
import datasets
import itertools

B_INST, E_INST = "[INST]", "[/INST]"
EOT_ID = 128009 #<|eot_id|>

def mask_target(target,seq):
    for i in range(len(seq)-len(target)):
        if seq[i:i+len(target)] == target:
            seq[i:i+len(target)] = [-100] * len(target)
    return seq


def tokenize_dialog(dialog, tokenizer):
    """
    Qwen의 포맷에 맞춰 대화(dialog)를 토큰화합니다.
    각 메시지는 "<|im_start|>{role}\n{content}<|im_end|>\n" 형식으로 변환되고,
    어시스턴트 메시지에 대해서만 labels를 실제 토큰으로, 나머지는 -100으로 설정합니다.
    마지막에 어시스턴트의 생성 프롬프트("<|im_start|>assistant\n")를 추가합니다.
    """
    input_ids = []
    labels = []

    for message in dialog:
        # 메시지 포맷: <|im_start|>{role}\n{content}<|im_end|>\n
        formatted = f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        message_ids = tokenizer.encode(formatted, add_special_tokens=False)
        input_ids.extend(message_ids)
        # 어시스턴트 메시지에 대해서만 실제 토큰 사용, 나머지는 -100 (loss 계산에서 무시)
        if message["role"] == "assistant":
            labels.extend(message_ids)
        else:
            labels.extend([-100] * len(message_ids))

    # 생성 프롬프트: 어시스턴트의 응답을 생성할 부분 (loss 계산 제외)
    gen_prompt = "<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(gen_prompt, add_special_tokens=False)
    input_ids.extend(prompt_ids)
    labels.extend([-100] * len(prompt_ids))

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids)
    }

########################

import os
from huggingface_hub import login
with open("api_key.txt", "r") as f:
    api_key = f.read().strip()  # 공백 제거
os.environ["HF_TOKEN"] = api_key
login(api_key)

import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from llama_cookbook.configs import train_config as TRAIN_CONFIG

train_config = TRAIN_CONFIG()
train_config.model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
train_config.num_epochs = 10
train_config.run_validation = False
train_config.gradient_accumulation_steps = 4
train_config.batch_size_training = 4
train_config.lr = 3e-4
train_config.use_fast_kernels = True
train_config.use_fp16 = True
train_config.context_length = 1024 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048 # T4 16GB or A10 24GB
train_config.batching_strategy = "packing"
train_config.output_dir = "Qwen2.5-Coder-7B-Instruct"
train_config.use_peft = True

from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            device_map="auto",
            quantization_config=config,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            torch_dtype=torch.float16,
        )

tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
tokenizer.pad_token = tokenizer.eos_token

##############################

from datasets import load_dataset

def get_custom_dataset_from_json(tokenizer, data_files, split="train"):
    # JSONL 파일 불러오기
    dataset = load_dataset("json", data_files=data_files, split=split)
    # 데이터셋에 필요한 속성 추가 (llama_cookbook의 get_dataloader가 요구하는 속성)
    # 각 샘플의 "conversation" 필드를 tokenize_dialog 함수를 통해 전처리합니다.
    dataset = dataset.map(
        lambda x: tokenize_dialog(x["conversation"], tokenizer),
        remove_columns=list(dataset.features)
    )
    dataset.dataset = "custom_dataset"
    dataset.train_split = split
    return dataset

# train과 validation 데이터셋 불러오기 (파일 경로 수정)
train_dataset = get_custom_dataset_from_json(tokenizer, "../dataset/train_messages.jsonl", split="train")
val_dataset = get_custom_dataset_from_json(tokenizer, "../dataset/val_messages.jsonl", split="train")


from llama_cookbook.data.concatenator import ConcatDataset
from llama_cookbook.utils.config_utils import get_dataloader_kwargs
def get_dataloader(tokenizer, dataset, train_config, split: str = "train"):
    dl_kwargs = get_dataloader_kwargs(train_config, dataset, tokenizer, split)

    if split == "train" and train_config.batching_strategy == "packing":
        dataset = ConcatDataset(dataset, chunk_size=train_config.context_length)

    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **dl_kwargs,
    )
    return dataloader

train_dataloader = get_dataloader(tokenizer, train_dataset, train_config)
eval_dataloader = get_dataloader(tokenizer, val_dataset, train_config, "val")

#############################

from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from dataclasses import asdict
from llama_cookbook.configs import lora_config as LORA_CONFIG

lora_config = LORA_CONFIG()
lora_config.r = 8
lora_config.lora_alpha = 32
lora_dropout: float=0.01

peft_config = LoraConfig(**asdict(lora_config))

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

##############################

import torch.optim as optim
from llama_cookbook.utils.train_utils import train
from torch.optim.lr_scheduler import StepLR

model.train()

optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

# Start the training process
results = train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    optimizer,
    scheduler,
    train_config.gradient_accumulation_steps,
    train_config,
    None,
    None,
    None,
    wandb_run=None,
)

model.save_pretrained(train_config.output_dir)
