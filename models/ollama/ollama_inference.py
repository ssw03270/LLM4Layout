import os
import glob

from ollama import chat
import json
import shutil

from tqdm import tqdm

datasets_folder = os.path.join(os.path.dirname(__file__), "l4")
output_folder = os.path.join(os.path.dirname(__file__), "[25_01_24]output")
image_paths = glob.glob(os.path.join(datasets_folder, "*_input.png"))
# json_paths = glob.glob(os.path.join(datasets_folder, "*15534_retrieval.json"))


# for image_path, json_path in zip(tqdm(image_paths), json_paths):
for image_path in tqdm(image_paths): #if image only
    # with open(json_path, "rb") as file:
    #     json_data = str(json.load(file))

    # json_data = json_data.replace("}, ", "},\n")
    # json_data = json_data.replace("[", "[\n")
    # json_data = json_data.replace("]", "\n]")

#     prompt = f"""
# 1. Example Layout Description
# Look at the example layout on the left, where multiple pieces of furniture are displayed in different colors on the room’s floor plan.
# The category of individual furniture in the left example layout, the colors shown in the 3D location, width, height, depth and angle, and layout are expressed in the Json format as follows.
# {json_data}
# Please describe how these pieces of furniture are arranged. For instance, note any patterns in spacing (e.g., minimum distance to avoid overlap), symmetry (e.g., chairs placed evenly around a table), alignment (e.g., parallel or centered), and circulation (e.g., ensuring people can move around).
# Based on your observations, infer the design intentions or policies (e.g., keeping furniture from overlapping, maintaining symmetrical chair placement around the dining table, etc.). Summarize these in your own words.
#
# 2. Design Plan for the Query
# We now want to apply the inferred design intentions to a new room depicted on the right.
# Please formulate a clear design plan that reflects the example’s logic and new room's structure. For instance:
# Which pieces of furniture to place first?
# How to ensure symmetry or parallel alignment of certain furniture items?
# What minimum spacing rules or other constraints will you use to avoid overlap and maintain a functional layout?
# How to arrange furniture naturally for the structure of a given new room?
# If there is any additional policy?like preserving circulation space or ensuring a certain distance from walls? please describe it.
#
# 3. Layout Element Generation
# Finally, generate the actual coordinates and sizes (category, x, y, z, w, h, d, angle) for each piece of furniture, according to the design plan.
# Following is a collection of categories of available furniture.
# ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet"]
#
# Be sure to explain briefly why each piece was placed at those coordinates (i.e., which part of the design policy or plan guided your decision).
# Make sure no furniture items overlap, and confirm that the design intentions (e.g., symmetrical arrangement) are met.
# Present the final layout in a clear format (e.g., JSON). Follow example in above:
# Please follow these three steps carefully?(1) describe the example layout and infer its design intentions, (2) outline a design plan for the new space, and (3) generate the final arrangement?so we can see your reasoning at each stage.
# """

    # prompt_simple=f"""
    # Look at the example dining room layout on the left side of the given image, where various pieces of furniture are displayed in different colors on the room's floor plan.
    # Below is the numerical representation of the example dining room layout in JSON format:
    # {json_data}
    # Now, we want to generate a dining room layout for the given room boundary, shown on the right side of the image (the empty room).
    # Since the example room and the given room have similar structures, it is reasonable to follow a similar design approach, adjusting it to fit the given room effectively.
    # Follow a step-by-step process to extract the design intention of the example room. Start by analyzing the room shape, considering the functionality of a dining room, identifying the total types and quantities of furniture, and strategically placing furniture.
    # Ensure proper alignment—using symmetry, parallel arrangements with other furniture and the room boundary—and maintain adequate spacing to avoid overlaps.
    # Finally, create your own room design plan that perfectly fits the given room boundary. Use the following list of available furniture types for the dining room:
    # ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet"]
    # Apply your design plan to the given room boundary and write the final coordinates, dimensions, and rotation (in radians) of each piece of furniture in the following JSON format like given in above.
    # """
#     prompt_simple2=f"""
#     Room Layout Description
# Boundary
# The room has an L-shaped boundary, which can be interpreted as a combination of two rectangles:
# The larger rectangle at the bottom and the smaller rectangle at the top.
# Design Decision: As this is a dining room, the primary focus is on the dining table and its surroundings. Placing the dining table in the center of the larger rectangle ensures a clear and functional layout, balancing aesthetics and accessibility.
# Dining Table
# Positioning: The dining table is located at:
# x: Center of the larger rectangle.
# z: Center of the larger rectangle.
# The central placement allows even spacing for chairs around the table.
# Dimensions:
# Width: Slightly longer than a standard dining table to accommodate four chairs comfortably.
# Depth: Slim to maintain proportion with the room and provide sufficient walking space.
# Height: Set at a comfortable height for dining.
# Orientation:
# Rotated 90° (1.57 radians) to align with the room’s longer wall.
# Color: The crimson color adds a vibrant and warm focal point to the room.
# Dining Chairs
# Quantity: Four identical chairs.
# Placement:
# Two chairs are placed on each side of the table in a symmetrical arrangement.
# Margins between the table and chairs are set to allow comfortable seating and movement:
# Chair x-position: Offset from the dining table by half the width of a chair plus a standard human seating margin (~0.4m).
# Chair z-position: Aligned to maintain symmetry on each side of the table.
# Dimensions:
# Standard dimensions to ensure compatibility with the table.
# The height is proportional to the table height.
# Orientation:
# Chairs are rotated to face the dining table (±1.57 radians, depending on the side).
# Color: Light green for a fresh and natural contrast with the crimson table.
# Wine Cabinet
# Positioning:
# Placed along the wall of the smaller rectangle (top section of the L-shape).
# x: Positioned near the wall's center for symmetry.
# z: Along the boundary to minimize interference with movement.
# Dimensions:
# Width: Spans a significant portion of the smaller rectangle’s wall for visual balance.
# Depth: Slim to prevent encroaching on walking space.
# Height: Tall to utilize vertical space efficiently.
# Color: Pink, adding a soft and elegant element to the layout.
# Pendant Lamps
# Quantity: Two identical pendant lamps.
# Positioning:
# Centered above the dining table for focused illumination.
# x: Same as the table’s x-coordinate.
# z: Aligned to the table’s length, one near each end.
# y: Height adjusted to hang above the table without obstructing view or movement.
# Dimensions:
# Slim and sleek to avoid visual clutter.
# Color: Sienna, complementing the warm tones of the dining table.
# Overall Design Intentions
# Functionality: The arrangement prioritizes clear pathways, accessible seating, and proper lighting for dining activities.
# Aesthetic Balance: Symmetry in furniture placement and harmonious color contrasts create a visually appealing layout.
# Efficient Space Usage: Slim dimensions for furniture maximize usable floor space while maintaining comfort.
#
# Above is design plan of example dining room layout on the left side of the given image, where various pieces of furniture are displayed in different colors on the room's floor plan.
#
# Implement python code to convert the above design plan into exact furniture descriptions. Function should assure that output match the conditions in layout plan (e.g. centered location, symmetry). Each furniture description should include the following attributes: furniture_name, x, y, z, h, w, d, and rotation(radian) to represent all furniture in the layout plan.
#     """
    partial_description='Grey nightstand is left of a wooden wardrobe with doors. White ceiling lamp with a wooden circle is directly hang above a block double bed.'
    prompt_simple3=f"""
    Complete the given partial scene description to describe every furniture in the scene.
    Description: Grey nightstand is left of a wooden wardrobe with doors. White ceiling lamp with a wooden circle is directly hang above a block double bed.
    """
    prompt_4plus=f"""
You are given a partial scene description and a rendered image of the room. Your task is to complete this description so that it becomes a thorough, precise, and formal depiction of the entire scene. Please:

1. List every piece of furniture mentioned or visible in the partial description, including their attributes (color, material, shape).
2. Describe the spatial relationships between furniture items (e.g., left, right, behind, adjacent, etc.).
3. Mention any lighting elements (lamps, windows, ceiling lights) and their materials or colors.


Ensure consistency with the partial description. Do not add furniture or details that are not implied in the rendered image of the room. The final text should be suitable for academic or technical documentation, using accurate and formal language.

Partial Description:
{partial_description}
"""
    prompt_simple4=f"""
You are given a partial scene description and a rendered image of the room. 
Your task is to complete the scene description so that it becomes a comprehensive and accurate depiction of all visible furniture and elements.

1. Base your description on both the partial text and the provided scene.
2. You may include any furniture or details that are visible in the scene but not explicitly mentioned in the partial description.
3. Do not introduce elements that are neither visible in the scene nor implied by the partial description.
4. If there is any discrepancy, rely on the visual content of the scene as the ground truth.
5. Use formal and precise language suitable for academic or technical documentation.

Partial Description:
{partial_description}
"""
    prompt_4plus_no_partial = f"""
    You are given a rendered image of the bedroom. Your task is to write a thorough, precise, and formal description of the entire scene. Please:

    1. List every piece of furniture mentioned or visible in the partial description, including their attributes (color, material, shape).
    2. Describe the spatial relationships between furniture items (e.g., left, right, behind, adjacent, etc.).
    3. Mention any lighting elements (lamps, windows, ceiling lights) and their materials or colors.


    Do not add furniture or details that are not implied in the rendered image of the room. The final text should be suitable for academic or technical documentation, using accurate and formal language.
    """
    prompt_city_no_partial = f"""
        You are given a satellite image of urban. Your task is to write a thorough, precise, and formal description of the entire scene. Please:

        1. List every piece of furniture mentioned or visible in the partial description, including their attributes (color, material, shape).
        2. Describe the spatial relationships between furniture items (e.g., left, right, behind, adjacent, etc.).
        3. Mention any lighting elements (lamps, windows, ceiling lights) and their materials or colors.


        Do not add furniture or details that are not implied in the rendered image of the room. The final text should be suitable for academic or technical documentation, using accurate and formal language.
        """
    prompt_VQA=f"""
    Carefully examine the attached urban block plan image. Black space means outside of urban block, which is out of our interest. White space is inside of block which is surrounded by road network. Blue bboxes are buildings and green circle is green area. Identify  urban block plan's potential weaknesses related to spatial arrangement, functionality, and Aesthetic perspective. Then, propose specific improvements for each weakness you identify, supported by detailed reasoning and references to visual elements in the image.
    """
    prompt_satellite_VQA=f"""
    Carefully examine the provided indoor layout image, evaluate the local relationships among the furniture pieces and their overall consistency, check alignment with the text prompt, and propose any necessary modifications to improve the design.
    Text condition: The bedroom where the table is in the middle and the bed and the dresser face each other
    """

    prompt_html=f"""
    Carefully examine the provided indoor layout image, evaluate the local relationships among the furniture pieces and their overall consistency, check alignment with the text prompt, and propose any necessary modifications to improve the design.
    Text condition: The bedroom where the table and ceiling lamp is in the middle and the bed and the dresser face each other.
    """


    output_file_name = image_path.split('\\')[-1].replace("_input.png", "") + "_output_code.txt"

    response = chat(
        model='llama3.2-vision',
        messages=[
            {
                'role': 'user',
                'content': prompt_html,
                'images': [image_path],
            }
        ],
    )


    output = response.message.content
    output_file_path = os.path.join(output_folder, output_file_name)
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(output)
    shutil.copy(image_path, output_folder)