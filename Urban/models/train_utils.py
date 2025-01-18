from torch import nn
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
from accelerate import Accelerator

class UrbanModel(nn.Module):
    def __init__(self, args):
        super(UrbanModel, self).__init__()
        self.args = args
        model_name = args["model_name"]
        self.vlm = MllamaForConditionalGeneration.from_pretrained(model_name,
                                                                  device_map="auto",
                                                                  torch_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def forward(self, source_image, target_image):
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            source_image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.args["device"])

        return self.vlm.generate(**inputs, max_new_tokens=30)
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