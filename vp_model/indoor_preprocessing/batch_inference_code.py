import os

import glob
from tqdm import tqdm

from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch

from PIL import Image

vlm = MllamaForConditionalGeneration.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct", device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

prompt = """
<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>
You are an expert visual reasoning assistant. 
You have the ability to observe an image and describe it in detail. 
Then, you will answer questions about the image, step by step, to demonstrate thorough understanding and reasoning.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
<|image|>
[1] First, describe the entire scene you observe in the image. 
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
Include any important details a designer or planner might need to know about this space.
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
prompts = [prompt] * 3

dataset_folder = "/local_datasets/llm4layout/outputs/indoor_layouts/real_images/"
image_paths = glob.glob(os.path.join(dataset_folder, "*.png"))[:3]

image_list = []
for image_path in tqdm(image_paths):
    image = Image.open(image_path)
    image_list.append([image])
print(image_list)

inputs = processor(
    images=image_list,
    text=prompts,
    add_special_tokens=False,
    return_tensors="pt"
).to(vlm.device)
output = vlm.generate(**inputs, max_new_tokens=1024)
output_text = processor.batch_decode(output)

for output in output_text:
    print(output)
    print("\n\n")