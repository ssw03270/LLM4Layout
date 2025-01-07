import os
import glob

from ollama import chat
import json

from tqdm import tqdm

datasets_folder = os.path.join(os.path.dirname(__file__), "datasets")
image_paths = glob.glob(os.path.join(datasets_folder, "*_input.png"))
json_paths = glob.glob(os.path.join(datasets_folder, "*_retrieval.json"))

for image_path, json_path in zip(tqdm(image_paths), json_paths):
    with open(json_path, "rb") as file:
        json_data = str(json.load(file))

    json_data = json_data.replace("}, ", "},\n")
    json_data = json_data.replace("[", "[\n")
    json_data = json_data.replace("]", "\n]")

    prompt = f"""
1. Example Layout Description
Look at the example layout on the left, where multiple pieces of furniture are displayed in different colors on the room’s floor plan.
The category of individual furniture in the left example layout, the colors shown in the 3D location, width, height, depth and angle, and layout are expressed in the Json format as follows.
{json_data}
Please describe how these pieces of furniture are arranged. For instance, note any patterns in spacing (e.g., minimum distance to avoid overlap), symmetry (e.g., chairs placed evenly around a table), alignment (e.g., parallel or centered), and circulation (e.g., ensuring people can move around).
Based on your observations, infer the design intentions or policies (e.g., keeping furniture from overlapping, maintaining symmetrical chair placement around the dining table, etc.). Summarize these in your own words.

2. Design Plan for the Query
We now want to apply the inferred design intentions to a new room depicted on the right. 
Please formulate a clear design plan that reflects the example’s logic and new room's structure. For instance:
Which pieces of furniture to place first?
How to ensure symmetry or parallel alignment of certain furniture items?
What minimum spacing rules or other constraints will you use to avoid overlap and maintain a functional layout?
How to arrange furniture naturally for the structure of a given new room? 
If there is any additional policy?like preserving circulation space or ensuring a certain distance from walls? please describe it.

3. Layout Element Generation
Finally, generate the actual coordinates and sizes (category, x, y, z, w, h, d, angle) for each piece of furniture, according to the design plan.
Following is a collection of categories of available furniture.
["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet"]

Be sure to explain briefly why each piece was placed at those coordinates (i.e., which part of the design policy or plan guided your decision).
Make sure no furniture items overlap, and confirm that the design intentions (e.g., symmetrical arrangement) are met.
Present the final layout in a clear format (e.g., JSON). Follow example in above:
Please follow these three steps carefully?(1) describe the example layout and infer its design intentions, (2) outline a design plan for the new space, and (3) generate the final arrangement?so we can see your reasoning at each stage.
"""
    output_file_name = image_path.split('\\')[-1].replace("_input.png", "") + "_output.txt"
    response = chat(
        model='llama3.2-vision',
        messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [image_path],
            }
        ],
    )
    output = response.message.content
    output_file_path = os.path.join(datasets_folder, output_file_name)
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(output)