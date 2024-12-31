import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# models/ 폴더에 있는 모듈들 임포트
from models.LLM import Model
from models.dataloader import Dataset_3Dfront


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained Model checkpoint with a test dataloader.")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/model_checkpoint_epoch_10.pth",
                        help="Path to the saved model checkpoint .pth file")
    parser.add_argument("--root_dir", type=str, default="./datasets/threed_front_diningroom",
                        help="Root directory for test dataset")
    parser.add_argument("--batch_size", type=int, default=2,
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
    axes[0].imshow(gt_layout, aspect='auto', cmap='viridis')
    axes[0].set_title("GT Layout")
    axes[0].axis("off")

    # 모델 output layout
    axes[1].imshow(pred_layout, aspect='auto', cmap='viridis')
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
