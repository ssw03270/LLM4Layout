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
    if tokenizer.vocab_size >= 128000:
        dialog_tokens = tokenizer.apply_chat_template(dialog)
        eot_indices = [i for i,n in enumerate(dialog_tokens) if n == EOT_ID]
        labels = copy.copy(dialog_tokens)
        #determine token for system and user
        system_or_user = (tokenizer.encode("system")[-1], tokenizer.encode("user")[-1])
        labels[0] = -100 # bos token
        last_idx = 1
        for n, idx in enumerate(eot_indices):
            role_token = labels[last_idx+1]
            if role_token in system_or_user:
                # Set labels to -100 for system and user tokens to ignore in loss function
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
            last_idx = idx + 1
        mask_target(tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False), labels)

        dialog_tokens = [dialog_tokens]
        labels_tokens = [labels]
    else:
        prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[::2]]
        answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[1::2]]
        dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))

        #Add labels, convert prompt token to -100 in order to ignore in loss function
        labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))

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
train_config.model_name = "meta-llama/Llama-3.2-3B-Instruct"
train_config.num_epochs = 1
train_config.run_validation = False
train_config.gradient_accumulation_steps = 4
train_config.batch_size_training = 1
train_config.lr = 3e-4
train_config.use_fast_kernels = True
train_config.use_fp16 = True
train_config.context_length = 1024 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048 # T4 16GB or A10 24GB
train_config.batching_strategy = "packing"
train_config.output_dir = "meta-llama-samsum"
train_config.use_peft = True

from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            device_map="auto",
            quantization_config=config,
            use_cache=False,
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
