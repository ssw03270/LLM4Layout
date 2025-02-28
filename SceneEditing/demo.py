# demo.py

import json
import os
import re
import string
from PIL import Image
from huggingface_hub import login

from scene_editor.editor.ccot_scene_editor import CCoTSceneEditor
# from editor.ccot_scene_editor import CCoTSceneEditor  # <== 수정: 우리가 만든 CCoT pipeline

def main():
    #token here plz
    login(token=HF_TOKEN)

    # 1) JSON 파일에서 scene graph 읽기
    json_path = r"C:\Users\SeongRae Noh\LLM4Layout\SceneEditing\data\SG_FRONT\relationships_bedroom_trainval.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    target_scan = "Bedroom-43072"
    scan_data = None

    # data["scans"] 구조 가정
    for scan in data["scans"]:
        if scan["scan"] == target_scan:
            scan_data = scan
            break

    if not scan_data:
        print(f"Scan {target_scan} not found!")
        return

    # 2) scene graph 변환
    objects_dict = scan_data["objects"]
    relationships = scan_data["relationships"]

    objects_list = []
    for obj_key, category in objects_dict.items():
        obj_id = f"{category}_{obj_key}"
        objects_list.append({
            "id": obj_id,
            "category": category,
            "updated": False
        })

    relationships_list = []
    for rel in relationships:
        subject_key = str(rel[0])
        object_key = str(rel[1])
        label = rel[3]
        subject_category = objects_dict.get(subject_key, "unknown")
        object_category = objects_dict.get(object_key, "unknown")
        subject_id = f"{subject_category}_{subject_key}"
        object_id = f"{object_category}_{object_key}"
        relationships_list.append({
            "subject": subject_id,
            "object": object_id,
            "label": label
        })

    scene_graph = {
        "scan_name": target_scan,
        "objects": objects_list,
        "relationships": relationships_list
    }

    # 3) 이미지 로드
    base_image_dir = r"D:\final_dataset\train\threed_front_bedroom"
    candidate_folder = None
    if os.path.exists(base_image_dir):
        pattern = re.compile(r".+_" + re.escape(target_scan) + r"$")
        for folder in os.listdir(base_image_dir):
            if pattern.match(folder):
                candidate_folder = folder
                break
    if candidate_folder is None:
        print("No folder found for scan:", target_scan)
        image = Image.new("RGB", (224, 224), color="white")
    else:
        image_path = os.path.join(base_image_dir, candidate_folder, "rendered_image.png")
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            print("Image not found:", image_path)
            image = Image.new("RGB", (224, 224), color="white")

    # 4) user instruction
    user_instruction = "Move the table left to the cabinet."

    # 5) 2단계 CCoT pipeline
    editor = CCoTSceneEditor(
        model_name="meta-llama/Llama-3.2-11B-Vision",
        device="cuda"
    )
    result = editor.run_ccot_pipeline(scene_graph, image, user_instruction)

    # 6) 결과 출력
    print("\n=== 1) Extracted Subgraph ===")
    print(json.dumps(result["extracted_subgraph"], indent=2))

    print("\n=== 2) Relation Changes ===")
    print(json.dumps(result["relation_changes"], indent=2))

    print("\n=== 3) Updated Scene Graph ===")
    print(json.dumps(result["updated_scene_graph"], indent=2))


if __name__ == "__main__":
    main()
