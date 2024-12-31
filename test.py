import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

# models/ 폴더에 있는 모듈들 임포트
from models.LLM import Model
from models.dataloader import Dataset_3Dfront


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained Model checkpoint with a test dataloader.")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/model_checkpoint_epoch_10.pth",
                        help="Path to the saved model checkpoint .pth file")
    parser.add_argument("--root_dir", type=str, default="./datasets/threed_front_diningroom",
                        help="Root directory for test dataset")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for test dataloader")
    parser.add_argument("--save_dir", type=str, default="./test_results",
                        help="Directory to save visualization images")
    return parser.parse_args()


def visualize_layouts(gt_layout, pred_layout, batch_idx, sample_idx, save_dir):
    """
    원본 layout(gt_layout)과 모델의 output layout(pred_layout)을
    하나의 그림에 나란히 시각화하여 저장한다.

    gt_layout, pred_layout: 각각 (25, 33) 형태의 텐서 또는 넘파이 배열
    batch_idx, sample_idx: 배치 인덱스, 배치 내부 샘플 인덱스
    save_dir: 결과 이미지를 저장할 디렉토리 경로
    """

    # 텐서라면 CPU로 이동 후 넘파이 변환
    if torch.is_tensor(gt_layout):
        gt_layout = gt_layout.detach().cpu().numpy()
    if torch.is_tensor(pred_layout):
        pred_layout = pred_layout.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # 원본 layout
    for element in gt_layout:
        pos = element[:3]
        size = element[3:6]
        rot = element[6:7]
        category = element[7:]

        width, height, depth = size[0] * 2, size[1] * 2, size[2] * 2  # 원본 코드처럼 *2
        dx, dy, dz = pos

        # Shapely를 사용하여 2D Polygon(Top-View) 구성
        # (width, depth) 기준으로 XY plane 대신 XZ plane을 사용하므로,
        # Polygon의 y좌표 자리에 depth를 넣어서 "바닥에서 본 형태"를 그립니다.
        base_rect = Polygon([
            (-width / 2, -depth / 2),
            (width / 2, -depth / 2),
            (width / 2, depth / 2),
            (-width / 2, depth / 2)
        ])

        # 회전은 라디안 단위로, base_rect를 중심에서 angle[0]만큼 회전 (Y축 기준 회전 가정)
        # Shapely rotate 함수는 기본적으로 origin='center'일 때,
        # polygon의 centroid를 기준으로 회전합니다. (origin 파라미터로 조절 가능)
        rect_rotated = rotate(base_rect, rot[0], use_radians=True)

        # X방향으로 dx, Y방향(실제로는 Z축)을 위해 dz를 사용해 평행이동
        rect_translated = translate(rect_rotated, xoff=dx, yoff=dz)

        # 폴리곤 시각화
        x_coords, y_coords = rect_translated.exterior.xy
        axes[0].fill(x_coords, y_coords, alpha=0.4)

    axes[0].set_title("GT Layout")
    axes[0].axis("off")

    # 모델 output layout
    # 원본 layout
    for element in pred_layout:
        pos = element[:3]
        size = element[3:6]
        rot = element[6:7]
        category = element[7:]

        width, height, depth = size[0] * 2, size[1] * 2, size[2] * 2  # 원본 코드처럼 *2
        dx, dy, dz = pos

        # Shapely를 사용하여 2D Polygon(Top-View) 구성
        # (width, depth) 기준으로 XY plane 대신 XZ plane을 사용하므로,
        # Polygon의 y좌표 자리에 depth를 넣어서 "바닥에서 본 형태"를 그립니다.
        base_rect = Polygon([
            (-width / 2, -depth / 2),
            (width / 2, -depth / 2),
            (width / 2, depth / 2),
            (-width / 2, depth / 2)
        ])

        # 회전은 라디안 단위로, base_rect를 중심에서 angle[0]만큼 회전 (Y축 기준 회전 가정)
        # Shapely rotate 함수는 기본적으로 origin='center'일 때,
        # polygon의 centroid를 기준으로 회전합니다. (origin 파라미터로 조절 가능)
        rect_rotated = rotate(base_rect, rot[0], use_radians=True)

        # X방향으로 dx, Y방향(실제로는 Z축)을 위해 dz를 사용해 평행이동
        rect_translated = translate(rect_rotated, xoff=dx, yoff=dz)

        # 폴리곤 시각화
        x_coords, y_coords = rect_translated.exterior.xy
        axes[1].fill(x_coords, y_coords, alpha=0.4)
    axes[1].set_title("Predicted Layout")
    axes[1].axis("off")

    plt.tight_layout()

    # 저장할 디렉토리가 없으면 생성
    os.makedirs(save_dir, exist_ok=True)

    # 예: "batch0_sample0.png" 형태로 저장
    save_path = os.path.join(save_dir, f"batch{batch_idx}_sample{sample_idx}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()  # 메모리 해제


def main():
    args = parse_args()

    # ====== 예시 Config ======
    # 학습할 때 사용했던 config와 동일하게 설정해야 합니다.
    config = {
        "fix_seed": 1,
        "num_epochs": 100,
        "lr": 1e-4,
        "batch_size": 1,
        "patch_size": 16,
        "stride": 8,
        "d_llm": 3072,
        "image_size": 256,
        "pred_len": 25,
        "llm_model": "LLAMA-3.2-3B",
        "llm_layers": 28,
        "dropout": 0.1,
        "d_model": 16,
        "d_ff": 32,
        "n_heads": 8
    }
    # =========================

    # 1) 모델 초기화
    model = Model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2) 체크포인트 로드
    print(f"[INFO] Loading checkpoint from: {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3) 테스트셋 & 로더 준비
    test_dataset = Dataset_3Dfront(root_dir=args.root_dir, data_type='test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"[INFO] Test dataset size: {len(test_dataset)}")
    print(f"[INFO] Test loader batch size: {args.batch_size}")

    # 4) 모델 추론 (Inference) 및 시각화
    with torch.no_grad():
        for batch_idx, (layout, image) in enumerate(test_dataloader):
            layout = layout.to(device)  # shape: (B, 25, 33)
            image = image.to(device)    # shape: (B, H, W, 1)  (마스크)

            # 모델 추론
            output = model(image)       # shape 예: (B, 25, 33)

            # 배치 내 각 샘플별로 시각화
            batch_size = layout.size(0)
            for sample_idx in range(batch_size):
                gt_layout_sample = layout[sample_idx]  # (25, 33)
                pred_layout_sample = output[sample_idx]  # (25, 33)

                visualize_layouts(
                    gt_layout=gt_layout_sample,
                    pred_layout=pred_layout_sample,
                    batch_idx=batch_idx,
                    sample_idx=sample_idx,
                    save_dir=args.save_dir
                )

            print(f"\n[Batch {batch_idx}]")
            print(f" - Input image shape:  {image.shape}")
            print(f" - Layout shape:       {layout.shape}")
            print(f" - Model output shape: {output.shape}")

            # 필요 시, 몇 개 배치만 시각화하고 break 할 수도 있음
            # if batch_idx == 2:
            #     break

    print("[INFO] Test inference & visualization finished!")


if __name__ == "__main__":
    main()
