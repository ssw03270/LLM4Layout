import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoProcessor, AutoConfig

from visual_prompt import ExpansiveVisualPrompt

class LayoutModel(nn.Module):
    def __init__(self, model_name, prompt_path):
        super(LayoutModel, self).__init__()
        if 'Llama' in model_name:
            from transformers import MllamaForConditionalGeneration
            self.vlm = MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        elif 'Qwen' in model_name:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        for param in self.vlm.parameters():
            param.requires_grad = False

        self.processor = AutoProcessor.from_pretrained(model_name)

        with open("./prompts/system_prompt.txt", "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

        if 'Llama' in model_name:
            with open(f"./prompts/Llama/{prompt_path}", "r", encoding="utf-8") as f:
                self.user_prompt = f.read()
        elif 'Qwen' in model_name:
            with open(f"./prompts/Qwen/{prompt_path}", "r", encoding="utf-8") as f:
                self.user_prompt = f.read()

        self.model_name = model_name

    def forward(self, real_inputs, target_inputs):
        with torch.no_grad():
            real_outputs = self.vlm(**real_inputs, output_attentions=False, output_hidden_states=False)
        target_outputs = self.vlm(**target_inputs, output_attentions=False, output_hidden_states=False)

        real_predicted_token_ids = real_outputs.logits.argmax(dim=-1)  # Shape: (batch_size, sequence_length)

        loss = F.cross_entropy(
            target_outputs.logits.view(-1, target_outputs.logits.size(-1)),  # [batch_size * sequence_length, vocab_size]
            real_predicted_token_ids.view(-1)  # [batch_size * sequence_length]
        )

        return loss

    def get_inputs(self, images, image_paths, text_descriptions, device):
        message_list = []
        for text_description, image_path in zip(text_descriptions, image_paths):
            user_prompt = self.user_prompt
            if 'text_description' in self.user_prompt:
                user_prompt = self.user_prompt.format(text_description=text_description)

            if 'Llama' in self.model_name:
                message = [
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
            elif 'Qwen' in self.model_name:
                message = [
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "file://" + image_path},
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]

            message_list.append(message)

        if len(message_list) != len(images):
            print("not enough prompts")
            exit()

        if 'Llama' in self.model_name:
            image_list = []
            for image in images:
                image_list.append([image])
            input_text = [
                self.processor.apply_chat_template(message, add_generation_prompt=True)
                for message in message_list
            ]
            inputs = self.processor(
                images=image_list,
                text=input_text,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt"
            ).to(device)

        elif 'Qwen' in self.model_name:
            input_text = [
                self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                for message in message_list
            ]
            from qwen_vl_utils import process_vision_info
            image_inputs, _ = process_vision_info(message_list)
            inputs = self.processor(
                images=image_inputs,
                text=input_text,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt"
            ).to(device)


        return inputs, message_list

    def generate(self, inputs):
        prompt_len = inputs.input_ids.shape[-1]

        outputs = self.vlm.generate(**inputs, max_new_tokens=1024)
        ids = outputs[:, prompt_len:]
        text = self.processor.batch_decode(ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)

        return text
def build_model(args):
    vlm_model = LayoutModel(args["model_name"], args["prompt_path"])
    vp_model = ExpansiveVisualPrompt(pad_size=560, target_size=500)
    return vlm_model, vp_model

def build_test_model(args, model_path):
    vlm_model = LayoutModel(args["model_name"], args["prompt_path"])
    vp_model = ExpansiveVisualPrompt(pad_size=560, target_size=500)
    vp_model.load_state_dict(torch.load(model_path))
    return vlm_model, vp_model

def get_optimizer(model, accelerator, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer

def get_scheduler(optimizer, accelerator, train_loader, num_epochs):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * num_epochs),
                                                                            int(0.72 * num_epochs)], gamma=0.1)
    return scheduler

def get_accelerator(train_dataloader, val_dataloader, vlm_model, vp_model, optimizer, scheduler, accelerator):
    # accelerator.state.select_deepspeed_plugin("student")
    train_dataloader, val_dataloader, vp_model, vlm_model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, val_dataloader, vp_model, vlm_model, optimizer, scheduler
    )

    # accelerator.state.select_deepspeed_plugin("teacher")
    # vlm_model = accelerator.prepare(
    #     vlm_model
    # )

    return train_dataloader, val_dataloader, vlm_model, vp_model, optimizer, scheduler

def get_test_accelerator(test_dataloader, vlm_model, vp_model, accelerator):
    vlm_model = accelerator.prepare(vlm_model)
    test_dataloader, vp_model = accelerator.prepare(test_dataloader, vp_model)

    return test_dataloader, vlm_model, vp_model