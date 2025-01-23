# https://github.com/jaepoong/PosterLlama/blob/main/src/model/minigpt_base.py
from torch import nn
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
from accelerate import Accelerator

import matplotlib.pyplot as plt

from visual_prompt import ExpansiveVisualPrompt

class UrbanModel(nn.Module):
    def __init__(self, args):
        super(UrbanModel, self).__init__()
        self.args = args
        model_name = args["model_name"]
        self.vlm = MllamaForConditionalGeneration.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
        self.vlm.tie_weights()
        for param in self.vlm.parameters():
            param.requires_grad = False

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.vp = ExpansiveVisualPrompt(pad_size=560, target_size=500)

        system_prompt = """You are an expert visual reasoning assistant. 
You have the ability to observe an image and describe it in detail. 
Then, you will answer questions about the image, step by step, to demonstrate thorough understanding and reasoning."""

        user_prompt = """[1] First, describe the entire scene you observe in the image. 
Include details about the space, objects, furniture, and any other notable elements.

[2] Next, explain your reasoning step by step. For each significant item in the scene, state what it is, where it is located, and how it relates to other objects. 
(Feel free to provide a chain-of-thought that outlines how you identify each object and interpret its position.)

[3] Answer the following specific questions:
   - What kind of room or space does this appear to be?
   - How many distinct pieces of furniture can you see?
   - Where are they positioned relative to each other?
   - Does the arrangement suggest any particular use case or activity?
   - Are there any notable design considerations, such as color scheme, user flow, or accessibility?

[4] Finally, summarize your observations in a concise paragraph. 
Include any important details a designer or planner might need to know about this space."""

        self.prompt = f"""
<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>

<|start_header_id|>user<|end_header_id|>
<|image|>{user_prompt}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
    def forward(self, source_image, target_image):
        prompts = [self.prompt] * source_image.size(0)

        images = []
        for i, image in enumerate(target_image):
            images.append([image])

        inputs = self.processor(
            images=images,
            text=prompts,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.args["device"])

        # inputs["pixel_values"] = self.vp(inputs["pixel_values"])

        # outputs = self.vlm(**inputs)
        output = self.vlm.generate(**inputs, max_new_tokens=1024)
        prompt_len = inputs.input_ids.shape[-1]
        # generated_ids = output[:, prompt_len:]
        # generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True,
        #                                              clean_up_tokenization_spaces=False)

        generated_text = self.processor.batch_decode(output)

        return generated_text

def build_model(args):
    model = UrbanModel(args)
    return model

def get_optimizer(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])
    return optimizer

def get_accelerator(train_dataloader, val_dataloader, model, optimizer):
    accelerator = Accelerator()
    train_dataloader, val_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, val_dataloader, model, optimizer
    )

    return train_dataloader, val_dataloader, model, optimizer, accelerator