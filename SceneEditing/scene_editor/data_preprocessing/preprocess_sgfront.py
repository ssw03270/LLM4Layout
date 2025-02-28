import os
import argparse
import json

def preprocess_sgfront(raw_dir, out_dir):
    """
    SG-FRONT(scene graph) 전처리
    - 노드, 엣지 정보 파싱
    - 필요한 경우 category 매핑/필터링
    """
    os.makedirs(out_dir, exist_ok=True)
    # 예시
    print(f"[INFO] SG-FRONT data preprocessed and saved to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    preprocess_sgfront(args.raw_dir, args.out_dir)

if __name__ == "__main__":
    main()
