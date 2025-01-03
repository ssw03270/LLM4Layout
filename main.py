import wandb

import argparse
import random
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs

from models.LLM import Model
from models.dataloader import Dataset_3Dfront

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

def visualize_layouts(gt_layout, pred_layout, batch_idx, sample_idx, device, save_dir):
    """
    원본 layout(gt_layout)과 모델의 output layout(pred_layout)을
    하나의 그림에 나란히 시각화하여 저장한다.

    gt_layout, pred_layout: 각각 (25, 33) 형태의 텐서 또는 넘파이 배열
    batch_idx, sample_idx: 배치 인덱스, 배치 내부 샘플 인덱스
    save_dir: 결과 이미지를 저장할 디렉토리 경로
    """

    # 텐서라면 CPU로 이동 후 넘파이 변환
    if torch.is_tensor(gt_layout):
        gt_layout = gt_layout.to(torch.float32).detach().cpu().numpy()
    if torch.is_tensor(pred_layout):
        pred_layout = pred_layout.to(torch.float32).detach().cpu().numpy()

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
    device = device.replace(":", "")
    save_path = os.path.join(save_dir, f"batch{batch_idx}_sample{sample_idx}_{device}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # 메모리 해제

def parse_args():
    """
    명령어 예시:
    $ accelerate launch train.py \
        --root_dir ./datasets/threed_front_diningroom \
        --num_epochs 10 \
        --lr 1e-4 \
        --batch_size 4 \
        --d_llm 3072 \
        --image_size 256 \
        --pred_len 25 \
        --llm_model "LLAMA-3.2-3B" \
        --llm_layers 28
    """
    parser = argparse.ArgumentParser(description="Train 3D Front LLM Model with HuggingFace Accelerate")

    parser.add_argument("--root_dir", type=str, default="./datasets/threed_front_diningroom",
                        help="Dataset root directory")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--fix_seed", type=int, default=1,
                        help="Random seed for reproducibility")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Patch size for the model")
    parser.add_argument("--stride", type=int, default=8,
                        help="Stride for the model")
    parser.add_argument("--d_llm", type=int, default=3072,
                        help="Dimension of the LLM model")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size to use for the model input")
    parser.add_argument("--pred_len", type=int, default=25,
                        help="Length of the prediction")
    parser.add_argument("--llm_model", type=str, default="LLAMA-3.2-3B",
                        help="Which LLM model to use")
    parser.add_argument("--llm_layers", type=int, default=28,
                        help="Number of layers in the LLM model")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--d_model", type=int, default=32,
                        help="Dimension of the model encoder/decoder hidden size")
    parser.add_argument("--d_ff", type=int, default=128,
                        help="Dimension of the feedforward layer")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads")

    args = parser.parse_args()
    return args


def main(args):
    # 1. Config 설정
    config = {
        "fix_seed": args.fix_seed,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "d_llm": args.d_llm,
        "image_size": args.image_size,
        "pred_len": args.pred_len,
        "llm_model": args.llm_model,
        "llm_layers": args.llm_layers,
        "dropout": args.dropout,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "n_heads": args.n_heads,
    }
    num_epochs = config["num_epochs"]

    # 2. Accelerator 설정 (멀티 GPU, TPUs 지원)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
    device = accelerator.device

    # 3. Seed 고정
    fix_seed = config["fix_seed"]
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 4. 데이터셋 및 로더 준비
    root_dir = args.root_dir
    train_dataset = Dataset_3Dfront(root_dir=root_dir, data_type='train')
    val_dataset = Dataset_3Dfront(root_dir=root_dir, data_type='val')
    test_dataset = Dataset_3Dfront(root_dir=root_dir, data_type='test')

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config["batch_size"],
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config["batch_size"],
                                shuffle=False)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=config["batch_size"],
                                shuffle=False)

    train_steps = len(train_dataloader)
    # 5. 모델 정의
    model = Model(config)

    # 6. Optimizer 정의
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

    # 7. 분산 학습 준비
    train_dataloader, val_dataloader, test_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, val_dataloader, test_dataloader, model, optimizer
    )

    # 8. Loss 정의
    criterion = nn.MSELoss()

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"run_{now_str}"
    if accelerator.is_main_process:
        os.makedirs(run_dir, exist_ok=True)

        # config.json 저장
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
    # 모든 프로세스가 run_dir 생성될 때까지 대기
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        wandb.login(key='0f272b4978c0b450c3765b24b8abd024d7799e80')
        wandb.init(project="3Dfront-LLM", config=config, name="train_3dfront_run", save_code=True)
        wandb.watch(model, log="all")

    # 9. 학습
    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            save_path = os.path.join(run_dir, f"{epoch + 1}")
            os.makedirs(save_path, exist_ok=True)

        # == Training ==
        model.train()
        train_loss_sum = 0.0
        train_steps = 0  # 평균을 내기 위해 step(배치) 수 카운트

        for layout, image in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
                                  disable=not accelerator.is_main_process):
            layout = layout.to(device)
            image = image.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, layout)

            # 역전파
            accelerator.backward(loss)
            optimizer.step()

            with torch.no_grad():
                # loss 는 shape=() 인 스칼라 텐서
                # gather 하면 모든 프로세스의 loss 값이 (world_size,) 텐서로 모임
                gathered_loss = accelerator.gather(loss.detach())
                mean_loss = gathered_loss.mean().item()

            train_loss_sum += mean_loss
            train_steps += 1

        # 모든 프로세스에서 loop가 끝난 뒤, 최종 평균(loss) 계산
        train_loss_avg = train_loss_sum / train_steps if train_steps > 0 else 0.0

        # == Validation ==
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        test_loss_sum = 0.0
        test_steps = 0

        with torch.no_grad():
            for layout, image in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]",
                                      disable=not accelerator.is_main_process):
                layout = layout.to(device)
                image = image.to(device)

                output = model(image)
                loss = criterion(output, layout)

                # 각 프로세스별 loss를 gather 후 평균
                gathered_loss = accelerator.gather(loss.detach())
                mean_loss = gathered_loss.mean().item()

                val_loss_sum += mean_loss
                val_steps += 1

            for batch_idx, (layout, image) in enumerate(tqdm(test_dataloader,
                                                             desc=f"Epoch {epoch + 1}/{num_epochs} [Test]",
                                                             disable=not accelerator.is_main_process)):
                layout = layout.to(device)
                image = image.to(device)

                output = model(image)
                loss = criterion(output, layout)

                # 각 프로세스별 loss를 gather 후 평균
                gathered_loss = accelerator.gather(loss.detach())
                mean_loss = gathered_loss.mean().item()

                test_loss_sum += mean_loss
                test_steps += 1

                # 배치 내 각 샘플별로 시각화
                batch_size = layout.size(0)
                for sample_idx in range(batch_size):
                    gt_layout_sample = layout[sample_idx]  # (25, 33)
                    pred_layout_sample = output[sample_idx]  # (25, 33)

                    save_path = os.path.join(run_dir, f"{epoch + 1}")
                    visualize_layouts(
                        gt_layout=gt_layout_sample,
                        pred_layout=pred_layout_sample,
                        batch_idx=batch_idx,
                        sample_idx=sample_idx,
                        device=str(device),
                        save_dir=save_path
                    )

        val_loss_avg = val_loss_sum / val_steps if val_steps > 0 else 0.0
        test_loss_avg = test_loss_sum / test_steps if test_steps > 0 else 0.0

        if accelerator.is_main_process:
            print(f"[Epoch {epoch + 1}/{num_epochs}] Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Test Loss: {test_loss_avg:.4f}")

            # -- wandb 로깅 --
            wandb.log({
                "train_loss": train_loss_avg,
                "val_loss": val_loss_avg,
                "test_loss": test_loss_avg,
                "epoch": epoch + 1
            })

if __name__ == "__main__":
    args = parse_args()
    main(args)
