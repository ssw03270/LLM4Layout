import os

import glob
from tqdm import tqdm

from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch

from PIL import Image

vlm = MllamaForConditionalGeneration.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct", device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

prompt = """
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
<|image|>what is this image?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
prompts = [prompt] * 3

dataset_folder = "./indoor_preprocessing/outputs/layouts/real_images/"
image_paths = glob.glob(os.path.join(dataset_folder, "*.png"))[:3]

image_list = []
for image_path in tqdm(image_paths):
    image = Image.open(image_path)
    image_list.append(image)
print(image_list)

inputs = processor(
    images=image_list,
    text=prompts,
    add_special_tokens=False,
    return_tensors="pt"
).to(vlm.device)
output = vlm.generate(**inputs, max_new_tokens=1024)
output_text = processor.batch_decode(output)

print(output_text)