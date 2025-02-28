#!/bin/bash
# 예시: 3D-FRONT 데이터 전처리 → SG-FRONT 매칭 → 이미지 정리

python src/data_preprocessing/preprocess_3dfront.py \
    --raw_dir data/3dfront_raw \
    --out_dir data/processed/3dfront

python src/data_preprocessing/preprocess_sgfront.py \
    --raw_dir data/sgfront_raw \
    --out_dir data/processed/sgfront

python src/data_preprocessing/build_dataset.py \
    --front_dir data/processed/3dfront \
    --sg_dir data/processed/sgfront \
    --img_dir data/images \
    --out_dir data/processed/final
