import shutil
import os
import glob

from ollama import chat

from tqdm import tqdm

def vlm(prompt, file_name, image_path):
    system_prompt = """
    You are an expert visual reasoning assistant.
    You have the ability to observe an image and describe it in detail.
    Then, you will answer questions about the image, step by step, to demonstrate thorough understanding and reasoning.
    """

    vlm_prompt = prompt
    vlm_response = chat(
        model='llama3.2-vision',
        messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': vlm_prompt,
                'images': [image_path],
            }
        ],
    )
    vlm_output = vlm_response.message.content

    output_file_name = image_path.split('\\')[-1].replace("_input.png", "") + f"_{file_name}.txt"
    output_file_path = os.path.join("results", output_file_name)
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(vlm_output)

datasets_folder = "../vp_model/indoor_preprocessing/outputs/indoor_layouts/"
image_paths = glob.glob(os.path.join(datasets_folder, "real_images/*.png"))[:15]
description_paths = glob.glob(os.path.join(datasets_folder, "text_description/*.txt"))[:15]

for image_path, description_path in zip(tqdm(image_paths), description_paths):
    with open(description_path, "r", encoding="utf-8") as f:
        description = f.read()

    prompt_1 = "Describe the furniture in the image."
    vlm(prompt_1, "output_1", image_path)

    prompt_2 = """
[1] First, describe the entire scene you observe in the image.
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
Include specific recommendations for optimizing furniture placement, enhancing traffic flow, and maximizing the space’s functionality.
"""
    vlm(prompt_2, "output_2", image_path)

    prompt_3 = f"""
{description}

[1] First, describe the entire scene you observe in the image.
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
Include specific recommendations for optimizing furniture placement, enhancing traffic flow, and maximizing the space’s functionality.
"""
    vlm(prompt_3, "output_3", image_path)

    output_file_path = os.path.join("results", image_path.split('\\')[-1])
    shutil.copy2(image_path, output_file_path)