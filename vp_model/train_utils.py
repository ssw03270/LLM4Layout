import torch
from torch import nn
import torch.nn.functional as F

from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoConfig
from accelerate.utils import DummyOptim, DummyScheduler

from visual_prompt import ExpansiveVisualPrompt

class LayoutModel(nn.Module):
    def __init__(self, model_name, prompt_path):
        super(LayoutModel, self).__init__()
        self.vlm = MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        for param in self.vlm.parameters():
            param.requires_grad = False

        self.processor = AutoProcessor.from_pretrained(model_name)

        with open("./prompts/system_prompt.txt", "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

        if 'Llama' in model_name:
            with open(f"./prompts/Llama/{prompt_path}", "r", encoding="utf-8") as f:
                self.main_prompt = f.read()
        elif 'Qwen' in model_name:
            with open(f"./prompts/Qwen/{prompt_path}", "r", encoding="utf-8") as f:
                self.main_prompt = f.read()

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

    def get_inputs(self, real_images, target_images, text_descriptions, device):
        prompts = []
        for text_description in text_descriptions:
            prompt = self.main_prompt.format(text_description=text_description)
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