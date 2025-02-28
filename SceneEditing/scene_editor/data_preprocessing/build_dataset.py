import os
import argparse

def build_dataset(front_dir, sg_dir, img_dir, out_dir):
    """
    3D-FRONT + SG-FRONT + 다중 시점 이미지(6시점)를 하나의 샘플로 묶어서
    학습/추론에 필요한 최종 Dataset 형태로 구성.
    """
    os.makedirs(out_dir, exist_ok=True)
    # 예시 로직
    # 1) matching scene_id
    # 2) merge JSON (scene3d info + graph info)
    # 3) associate 6 viewpoint images
    # 4) save as pkl or custom format
    print(f"[INFO] Dataset built and saved to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front_dir", type=str, required=True)
    parser.add_argument("--sg_dir", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    build_dataset(args.front_dir, args.sg_dir, args.img_dir, args.out_dir)

if __name__ == "__main__":
    main()
