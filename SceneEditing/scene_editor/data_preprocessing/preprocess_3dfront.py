import os
import argparse
import json
import numpy as np


def preprocess_3dfront(raw_dir, out_dir):
    """
    3D-FRONT의 원본 데이터를 InstructScene 방식 등으로 전처리.
    - 메타데이터 추출(객체, 카테고리, 3D 정보 등)
    - 필요시 좌표계 정규화
    - 결과를 JSON 등 형태로 저장
    """
    os.makedirs(out_dir, exist_ok=True)

    # 예시 로직 (실제 구현은 InstructScene 코드 참고)
    # for scene_file in os.listdir(raw_dir):
    #     if scene_file.endswith(".json"):
    #         with open(os.path.join(raw_dir, scene_file), 'r') as f:
    #             data = json.load(f)
    #         # ... 전처리 ...
    #         output_path = os.path.join(out_dir, scene_file)
    #         with open(output_path, 'w') as of:
    #             json.dump(data, of)

    print(f"[INFO] 3D-FRONT data preprocessed and saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    preprocess_3dfront(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
