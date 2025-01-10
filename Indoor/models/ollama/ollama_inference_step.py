import os
import glob

from ollama import chat
import json

from tqdm import tqdm

datasets_folder = os.path.join(os.path.dirname(__file__), "datasets")
save_folder = os.path.join(os.path.dirname(__file__), "outputs")
source_boundary_image_paths = glob.glob(os.path.join(datasets_folder, "*_source_boundary_image.png"))
retrieved_image_paths = glob.glob(os.path.join(datasets_folder, "*_retrieved_gt_image.png"))
json_paths = glob.glob(os.path.join(datasets_folder, "*_retrieved_json.json"))

for source_boundary_image_path, retrieved_image_path, json_path in zip(tqdm(source_boundary_image_paths), retrieved_image_paths, json_paths):
    with open(json_path, "rb") as file:
        json_data = str(json.load(file))

    json_data = json_data.replace("}, ", "},\n")
    json_data = json_data.replace("[", "[\n")
    json_data = json_data.replace("]", "\n]")

    vlm_prompt_first = f"""
You are an interior design expert specializing in dining room layouts. 
Use the provided 2D floor plan image and its corresponding JSON description to analyze the furniture arrangement.
JSON data: {json_data}

Identify patterns based on spacing, symmetry, alignment, and circulation, and explicitly include numerical data in the process to derive clear design policies.
Keep in mind that the units are in meters and radians.
"""
    vlm_response_first = chat(
        model='llama3.2-vision',
        messages=[
            {
                'role': 'user',
                'content': vlm_prompt_first,
                'images': [retrieved_image_path],
            }
        ],
    )
    vlm_output_first = vlm_response_first.message.content
    print(vlm_output_first, "\n\n--------\n\n")

    vlm_prompt_second = f"""
You are an interior design expert specializing in dining room layouts. 
Using the inferred design intentions and the provided floor plan, create a clear furniture arrangement plan for the new room, incorporating numerical data:
Infered desing intentions:
{vlm_output_first}

Address the following:
- Furniture placement order
- Methods to ensure symmetry or alignment
- Minimum spacing rules to avoid overlap
- Arrangement strategies adapted to the room's structure
- Additional policies (e.g., maintaining distance from walls, preserving circulation space)

Keep in mind that the units are in meters and radians.
"""
    vlm_response_first = chat(
        model='llama3.2-vision',
        messages=[
            {
                'role': 'user',
                'content': vlm_prompt_second,
                'images': [source_boundary_image_path],
            }
        ],
    )
    vlm_output_second = vlm_response_first.message.content
    print(vlm_output_second, "\n\n--------\n\n")

    llm_prompt_third = f"""
You are an interior design expert specializing in arranging furniture in dining rooms. 
Based on our previous conversation, you created a clear design plan for arranging furniture in the room:
{vlm_output_second}

Using your design plan, generate the final furniture arrangement layout.
The layout must be output in the format of a list of dictionaries, where each dictionary includes the keys: 
"furniture," "x," "y," "z," "width," "height," "depth," and "angle."

Keep in mind that the units are in meters and radians.
"""
    llm_response = chat(
        model='llama3.2',
        messages=[
            {
                'role': 'user',
                'content': llm_prompt_third,
            },
        ])
    llm_output = llm_response.message.content
    print(llm_output, "\n\n--------\n\n")

    output_file_name = source_boundary_image_path.split('\\')[-1].replace("_source_boundary_image.png", "") + "_output.txt"
    output_file_path = os.path.join(save_folder, output_file_name)
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(llm_output)