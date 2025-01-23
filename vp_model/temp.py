import os

import glob
from tqdm import tqdm

from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch

from PIL import Image

vlm = MllamaForConditionalGeneration.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct", device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "what is this image?"}
    ]}
] * 3
input_texts = processor.apply_chat_template(messages, add_generation_prompt=True)

dataset_folder = "/data/ssw03270/github/LLM4Layout/vp_model/indoor_preprocessing/outputs/layouts/real_images"
image_paths = glob.glob(os.path.join(dataset_folder, ".png"))[:3]

image_list = []
for image_path in tqdm(image_paths):
    image = Image.open(image_path)
    image_list.append(image)
print(image_list)

inputs = processor(
    image_list,
    input_texts,
    add_special_tokens=False,
    return_tensors="pt"
).to(vlm.device)
output = vlm.generate(**inputs, max_new_tokens=1024)
output_text = processor.decode(output[0])

print(output_text)