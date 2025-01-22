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

from accelerate import Accelerator  # 추가


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained Model checkpoint with a test dataloader.")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/model_checkpoint_epoch_10.pth",
                        help="Path to the saved vp_model checkpoint .pth file")
    parser.add_argument("--root_dir", type=str, default="./datasets/threed_front_diningroom",
                        help="Root directory for test dataset")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for test dataloader")
    parser.add_argument("--save_dir", type=str, default="./test_results",
                        help="Directory to save visualization images")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        help="Mixed precision type (none, fp16, bf16)")  # 추가
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

    # Determine the number of categories based on gt_layout and pred_layout
    # Assuming categories are represented as one-hot or probability vectors starting from index 7
    num_categories_gt = gt_layout.shape[-1] - 7
    num_categories_pred = pred_layout.shape[-1] - 7
    num_categories = max(num_categories_gt, num_categories_pred)

    # Define color map
    cmap = plt.get_cmap('tab20')  # You can choose other colormaps like 'tab10', 'tab20', etc.
    colors = [cmap(i) for i in range(num_categories)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Increased figure size for better visibility

    # Helper function to plot a layout
    def plot_layout(layout, ax, is_pred=False):
        for element in layout:
            pos = element[:3]
            size = element[3:6]
            rot = element[6]
            category = element[7:]

            # For pred_layout, check if any category probability >= 0.5
            if is_pred:
                max_prob = np.max(category)
                if max_prob < 0.5:
                    continue  # Skip this object as it doesn't meet the threshold
                category_idx = np.argmax(category)
            else:
                category_idx = np.argmax(category)

            # Assign color based on category
            color = colors[category_idx] if category_idx < len(colors) else (
            0.5, 0.5, 0.5, 1)  # Default to gray if out of range

            width, height, depth = size[0] * 2, size[1] * 2, size[2] * 2  # Scale sizes as per original code
            dx, dy, dz = pos

            # Create base rectangle centered at origin
            base_rect = Polygon([
                (-width / 2, -depth / 2),
                (width / 2, -depth / 2),
                (width / 2, depth / 2),
                (-width / 2, depth / 2)
            ])

            # Rotate the rectangle around its centroid
            rect_rotated = rotate(base_rect, rot, use_radians=True)

            # Translate the rectangle to its position
            rect_translated = translate(rect_rotated, xoff=dx, yoff=dz)

            # Extract coordinates for plotting
            x_coords, y_coords = rect_translated.exterior.xy
            ax.fill(x_coords, y_coords, alpha=0.6, color=color, edgecolor='k')

    # Plot Ground Truth Layout
    plot_layout(gt_layout, axes[0], is_pred=False)
    axes[0].set_title("Ground Truth Layout", fontsize=16)
    axes[0].axis("off")
    axes[0].set_aspect('equal', adjustable='box')

    # Plot Predicted Layout
    plot_layout(pred_layout, axes[1], is_pred=True)
    axes[1].set_title("Predicted Layout", fontsize=16)
    axes[1].axis("off")
    axes[1].set_aspect('equal', adjustable='box')

    # Create a legend for categories
    # Assuming you have category names; if not, you can use category indices
    # Example:
    # category_names = ['Chair', 'Table', 'Sofa', ...]
    # Here, we'll use category indices for demonstration
    category_patches = [plt.Line2D([0], [0], marker='s', color='w', label=f'Category {i}',
                                   markerfacecolor=colors[i], markersize=10) for i in range(num_categories)]
    plt.legend(handles=category_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Categories")

    plt.tight_layout()

    # 저장할 디렉토리가 없으면 생성
    os.makedirs(save_dir, exist_ok=True)

    # 예: "batch0_sample0.png" 형태로 저장
    save_path = os.path.join(save_dir, f"batch{batch_idx}_sample{sample_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
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

    # 1) Accelerator 초기화
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    # 2) 모델 초기화
    model = Model(config)

    # 3) 체크포인트 로드
    print(f"[INFO] Loading checkpoint from: {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4) 데이터셋 & 로더 준비
    test_dataset = Dataset_3Dfront(root_dir=args.root_dir, data_type='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
    )

    print(f"[INFO] Test dataset size: {len(test_dataset)}")
    print(f"[INFO] Test loader batch size: {args.batch_size}")

    # 5) Accelerator로 모델과 데이터 로더 준비
    test_dataloader, model = accelerator.prepare(
        test_dataloader, model
    )

    # 6) 모델 추론 (Inference) 및 시각화
    with torch.no_grad():
        for batch_idx, (layout, image) in enumerate(test_dataloader):
            # 모델 추론
            output = model(image)  # shape 예: (B, 25, 33)

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
