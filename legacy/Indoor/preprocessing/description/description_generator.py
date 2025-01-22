import os
import glob
from tqdm import tqdm

import random
import numpy as np

from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch

from PIL import Image
import matplotlib.pyplot as plt

vlm = MllamaForConditionalGeneration.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct", device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

prompt_4plus_no_partial = f"""
You are given a rendered image of the bedroom. Your task is to write a thorough, precise, and formal description of the entire scene. Please:
1. List every piece of furniture mentioned or visible in the partial description, including their attributes (color, material, shape).
2. Describe the spatial relationships between furniture items (e.g., left, right, behind, adjacent, etc.).
3. Mention any lighting elements (lamps, windows, ceiling lights) and their materials or colors.

Do not add furniture or details that are not implied in the rendered image of the room. The final text should be suitable for academic or technical documentation, using accurate and formal language.
"""
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt_4plus_no_partial}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

seed = 327

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

file_names = ["000.png", "001.png", "002.png", "003.png", "004.png", "005.png", "006.png", "007.png"]

dataset_folder = "E:\\Resources\\IndoorSceneSynthesis\\InstructScene\\threed_front_bedroom"
image_paths = glob.glob(os.path.join(dataset_folder, "**", "blender_rendered_scene_256"))
new_image_paths = []

for image_path in image_paths:
    random_choice = random.choice(file_names)
    new_image_path = os.path.join(image_path, random_choice)
    new_image_paths.append(new_image_path)

for image_path in tqdm(new_image_paths):
    image = Image.open(image_path)
    output_file_path = [image_path.split("\\")[-3]] + [image_path.split("\\")[-1]]
    output_file_name = "_".join(output_file_path).replace(".png", ".txt")

    # 이미지 출력
    # plt.figure(figsize=(6, 6))  # 출력 창 크기 설정
    # plt.imshow(image)
    # plt.axis("off")  # 축 숨기기
    # plt.title(image_path)  # 이미지 경로를 제목으로 설정
    # plt.show()

    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(vlm.device)
    output = vlm.generate(**inputs, max_new_tokens=1024)
    output_text = processor.decode(output[0])

    with open(f"outputs/{output_file_name}", "w", encoding="utf-8") as f:
        f.write(output_text)