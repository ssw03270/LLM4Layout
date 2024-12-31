import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from accelerate import Accelerator

from models.LLM import Model
from models.dataloader import Dataset_3Dfront


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
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2,
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
    parser.add_argument("--d_model", type=int, default=16,
                        help="Dimension of the model encoder/decoder hidden size")
    parser.add_argument("--d_ff", type=int, default=32,
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

    # 2. Accelerator 설정 (멀티 GPU, TPUs 지원)
    accelerator = Accelerator()
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

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config["batch_size"],
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config["batch_size"],
                                shuffle=False)

    # 5. 모델 정의
    model = Model(config)

    # 6. Optimizer 정의
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

    # 7. 분산 학습 준비
    train_dataloader, val_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, val_dataloader, model, optimizer
    )

    # 8. Loss 정의
    criterion = nn.MSELoss()

    # 9. 학습
    num_epochs = config["num_epochs"]
    for epoch in range(num_epochs):
        # == Training ==
        model.train()
        train_loss_sum = 0.0
        for layout, image in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            layout = layout.to(device)
            image = image.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, layout)

            # 역전파
            accelerator.backward(loss)
            optimizer.step()

            train_loss_sum += loss.item()

        train_loss_avg = train_loss_sum / len(train_dataloader)

        # == Validation ==
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for layout, image in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                layout = layout.to(device)
                image = image.to(device)
                output = model(image)
                loss = criterion(output, layout)
                val_loss_sum += loss.item()
        val_loss_avg = val_loss_sum / len(val_dataloader)

        # Epoch 결과 출력
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")

        # 모델 저장 (모든 프로세스가 완료될 때만 저장)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = f"model_checkpoint_epoch_{epoch+1}.pth"
        torch.save(unwrapped_model.state_dict(), save_path)
        print(f"Model saved at {save_path}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
