import torch
from torch import nn
import torch.nn.functional as F

from transformers import MllamaForConditionalGeneration, AutoProcessor
from accelerate import Accelerator

from visual_prompt import ExpansiveVisualPrompt

class UrbanModel(nn.Module):
    def __init__(self, model_name):
        super(UrbanModel, self).__init__()
        self.vlm = MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
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
    def forward(self, real_inputs, target_inputs):
        # outputs = self.vlm(**inputs)
        # real_outputs = self.vlm.generate(**real_inputs, max_new_tokens=1024)
        # target_outputs = self.vlm.generate(**target_inputs, max_new_tokens=1024)

        # prompt_len = real_inputs.input_ids.shape[-1]
        # generated_ids = output[:, prompt_len:]
        # generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True,
        #                                              clean_up_tokenization_spaces=False)

        real_outputs = self.vlm(**real_inputs, output_attentions=False, output_hidden_states=False)
        target_outputs = self.vlm(**target_inputs, output_attentions=False, output_hidden_states=False)

        real_predicted_token_ids = real_outputs.logits.argmax(dim=-1)  # Shape: (batch_size, sequence_length)
        # target_predicted_token_ids = target_outputs.logits.argmax(dim=-1)

        loss = F.cross_entropy(
            target_outputs.logits.view(-1, target_outputs.logits.size(-1)),  # [batch_size * sequence_length, vocab_size]
            real_predicted_token_ids.view(-1)  # [batch_size * sequence_length]
        )

        # real_decoded_text = self.processor.batch_decode(real_predicted_token_ids)
        # target_decoded_text = self.processor.batch_decode(target_predicted_token_ids)
        # print(real_decoded_text)
        # print(target_decoded_text)

        # print(real_outputs.logits.shape)
        # print(target_outputs.logits.shape)
        # print(loss)
        # exit()

        return loss

        # real_generated_text = self.processor.batch_decode(real_output)
        # target_generated_text = self.processor.batch_decode(target_output)
        #
        # return real_generated_text, target_generated_text
    def get_inputs(self, real_images, target_images, device):
        prompts = [self.prompt] * real_images.size(0)
        real_image_list = []
        target_image_list = []
        for real_image, target_image in zip(real_images, target_images):
            real_image_list.append([real_image])
            target_image_list.append([target_image])

        real_inputs = self.processor(
            images=real_image_list,
            text=prompts,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(device)

        target_inputs = self.processor(
            images=target_image_list,
            text=prompts,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(device)

        return real_inputs, target_inputs
def build_model(args):
    vlm_model = UrbanModel(args["model_name"])
    vp_model = ExpansiveVisualPrompt(pad_size=560, target_size=500)
    return vlm_model, vp_model

def get_optimizer(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * args["num_epochs"]),
                                                                            int(0.72 * args["num_epochs"])], gamma=0.1)
    return scheduler

def get_accelerator(train_dataloader, val_dataloader, model, optimizer):
    accelerator = Accelerator()
    train_dataloader, val_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, val_dataloader, model, optimizer
    )

    return train_dataloader, val_dataloader, model, optimizer, accelerator