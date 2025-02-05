import torch
from torch import nn
import torch.nn.functional as F

from transformers import MllamaForConditionalGeneration, AutoProcessor
from accelerate import Accelerator, DeepSpeedPlugin

from visual_prompt import ExpansiveVisualPrompt

class LayoutModel(nn.Module):
    def __init__(self, model_name):
        super(LayoutModel, self).__init__()
        self.vlm = MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.vlm.tie_weights()
        for param in self.vlm.parameters():
            param.requires_grad = False

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.system_prompt = """You are an expert visual reasoning assistant. 
You have the ability to observe an image and describe it in detail. 
Then, you will answer questions about the image, step by step, to demonstrate thorough understanding and reasoning."""

        self.user_prompt = """[1] First, describe the entire scene you observe in the image. 
Include details about the space, objects, furniture, and any other notable elements.

[2] Next, explain your reasoning step by step. 
For each significant item in the scene, state what it is, where it is located, and how it relates to other objects. 
(Feel free to provide a chain-of-thought that outlines how you identify each object and interpret its position.)

[3] Answer the following specific questions:
   - What kind of room or space does this appear to be?
   - How many distinct pieces of furniture can you see?
   - Where are they positioned relative to each other?
   - Does the arrangement suggest any particular use case or activity?
   - Are there any notable design considerations, such as color scheme, user flow, or accessibility?

[4] From a designer or planner’s perspective, please provide a concise overview of how to improve the existing furniture arrangement. 
Include specific recommendations for optimizing furniture placement, enhancing traffic flow, and maximizing the space’s functionality."""

    def forward(self, real_inputs, target_inputs):
        real_outputs = self.vlm(**real_inputs, output_attentions=False, output_hidden_states=False)
        target_outputs = self.vlm(**target_inputs, output_attentions=False, output_hidden_states=False)

        real_predicted_token_ids = real_outputs.logits.argmax(dim=-1)  # Shape: (batch_size, sequence_length)

        loss = F.cross_entropy(
            target_outputs.logits.view(-1, target_outputs.logits.size(-1)),  # [batch_size * sequence_length, vocab_size]
            real_predicted_token_ids.view(-1)  # [batch_size * sequence_length]
        )

        return loss

    def get_inputs(self, real_images, target_images, text_descriptions, device):
        prompts = []
        for text_description in text_descriptions:
            prompt = f"""<|begin_of_text|>
            
<|start_header_id|>system<|end_header_id|>
{self.system_prompt}<|eot_id|>

<|start_header_id|>user<|end_header_id|>
<|image|>{text_description}

{self.user_prompt}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
            prompts.append(prompt)

        real_image_list = []
        target_image_list = []

        for real_image, target_image in zip(real_images, target_images):
            real_image_list.append([real_image])
            target_image_list.append([target_image])

        if len(prompts) != len(real_images) or len(prompts) != len(target_images):
            print("not enough prompts")
            exit()

        real_inputs = self.processor(
            images=real_image_list,
            text=prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        ).to(device)

        target_inputs = self.processor(
            images=target_image_list,
            text=prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        ).to(device)

        return real_inputs, target_inputs, prompts

    def generate(self, real_inputs, target_inputs):
        prompt_len = real_inputs.input_ids.shape[-1]

        real_outputs = self.vlm.generate(**real_inputs, max_new_tokens=1024)
        real_ids = real_outputs[:, prompt_len:]
        real_text = self.processor.batch_decode(real_ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)

        target_outputs = self.vlm.generate(**target_inputs, max_new_tokens=1024)
        target_ids = target_outputs[:, prompt_len:]
        target_text = self.processor.batch_decode(target_ids, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)

        return real_text, target_text
def build_model(args):
    vlm_model = LayoutModel(args["model_name"])
    vp_model = ExpansiveVisualPrompt(pad_size=560, target_size=500)
    return vlm_model, vp_model

def build_test_model(args, model_path):
    vlm_model = LayoutModel(args["model_name"])
    vp_model = ExpansiveVisualPrompt(pad_size=560, target_size=500)
    vp_model.load_state_dict(torch.load(model_path))
    return vlm_model, vp_model

def get_optimizer(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * args["num_epochs"]),
                                                                            int(0.72 * args["num_epochs"])], gamma=0.1)
    return scheduler

def get_accelerator(train_dataloader, val_dataloader, vlm_model, vp_model, optimizer, scheduler, ds_config_path="ds_config.json"):
    # DeepSpeedPlugin을 원하는 설정값으로 생성합니다.
    # 예: ZeRO Stage 3를 사용하고, 옵티마이저 및 파라미터를 CPU로 오프로드하며,
    # 혼합 정밀도는 bf16(혹은 fp16)으로 설정한다고 가정합니다.
    ds_plugin = DeepSpeedPlugin(
        hf_ds_config=ds_config_path,  # DeepSpeed 구성 파일 경로 또는 dict
        gradient_accumulation_steps=1,  # (필요에 따라 조정)
        gradient_clipping=1.0,  # (예: 최대 norm 1.0)
        zero_stage=3,  # ZeRO Stage 3 적용
        offload_optimizer_device="cpu",  # 옵티마이저 상태 오프로드: CPU 사용
        offload_param_device="cpu",  # 모델 파라미터 오프로드: CPU 사용
        zero3_init_flag=True,  # ZeRO‑3 초기화를 위한 flag 활성화
        zero3_save_16bit_model=True,  # 체크포인트 저장 시 16비트 가중치 수집
    )

    # Accelerator를 DeepSpeedPlugin을 포함하여 생성합니다.
    accelerator = Accelerator(deepspeed_plugin=ds_plugin)

    train_dataloader, val_dataloader, vp_model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, val_dataloader, vp_model, optimizer, scheduler
    )
    vlm_model = accelerator.prepare(vlm_model)

    return train_dataloader, val_dataloader, vlm_model, vp_model, optimizer, scheduler, accelerator

def get_test_accelerator(test_dataloader, vlm_model, vp_model, optimizer, scheduler, ds_config_path="ds_config.json"):
    # DeepSpeedPlugin을 원하는 설정값으로 생성합니다.
    # 예: ZeRO Stage 3를 사용하고, 옵티마이저 및 파라미터를 CPU로 오프로드하며,
    # 혼합 정밀도는 bf16(혹은 fp16)으로 설정한다고 가정합니다.
    ds_plugin = DeepSpeedPlugin(
        hf_ds_config=ds_config_path,  # DeepSpeed 구성 파일 경로 또는 dict
        gradient_accumulation_steps=1,  # (필요에 따라 조정)
        gradient_clipping=1.0,  # (예: 최대 norm 1.0)
        zero_stage=3,  # ZeRO Stage 3 적용
        offload_optimizer_device="cpu",  # 옵티마이저 상태 오프로드: CPU 사용
        offload_param_device="cpu",  # 모델 파라미터 오프로드: CPU 사용
        zero3_init_flag=True,  # ZeRO‑3 초기화를 위한 flag 활성화
        zero3_save_16bit_model=True,  # 체크포인트 저장 시 16비트 가중치 수집
    )

    # Accelerator를 DeepSpeedPlugin을 포함하여 생성합니다.
    accelerator = Accelerator(deepspeed_plugin=ds_plugin)

    test_dataloader, vp_model, optimizer, scheduler = accelerator.prepare(
        test_dataloader, vp_model, optimizer, scheduler
    )
    vlm_model = accelerator.prepare(vlm_model)

    return test_dataloader, vlm_model, vp_model, optimizer, scheduler, accelerator