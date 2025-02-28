#!/bin/bash
# 예시: 편집 추론 스크립트
SCENE_JSON="data/processed/final/sample_scene.json"
IMAGES_DIR="data/images/sample_scene"

python -c "
import json
from src.models.inference_pipeline import SceneEditingPipeline
import os

with open('$SCENE_JSON','r') as f:
    scene_graph = json.load(f)

# 간단히 이미지 6장 로드 (여기선 placeholder)
images = [os.path.join('$IMAGES_DIR', f'view_{i}.png') for i in range(6)]

pipeline = SceneEditingPipeline()

user_instruction = 'Rotate the bed by 90 degrees, move the table near the window, keep the vase on the table.'
html_code = pipeline.edit_scene_and_export(scene_graph, images, user_instruction)

print('=== Generated HTML Code ===')
print(html_code)
"
