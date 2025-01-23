import data_utils
import pre_utils
import train_utils

import os
from datetime import datetime

from tqdm import tqdm
import wandb

import torch

if __name__ == "__main__":
    args = pre_utils.parse_args()
    pre_utils.set_seed(args["seed"])
    dataset_paths_dict = data_utils.get_dataset_paths(args["dataset_dir"])
    split_dataset_paths = data_utils.split_dataset(dataset_paths_dict)

    train_dataset = data_utils.LayoutDataset(split_dataset_paths, "train")
    val_dataset = data_utils.LayoutDataset(split_dataset_paths, "val")

    train_dataloader = data_utils.get_dataloader(train_dataset, args, shuffle=True)
    val_dataloader = data_utils.get_dataloader(val_dataset, args, shuffle=False)

    model = train_utils.build_model(args)
    optimizer = train_utils.get_optimizer(model, args)

    train_dataloader, val_dataloader, model, optimizer, accelerator = train_utils.get_accelerator(
        train_dataloader, val_dataloader, model, optimizer)
    device = accelerator.device

    if accelerator.is_main_process:
        wandb_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 형식: YYYY-MM-DD_HH-MM-SS
        wandb.login(key='0f272b4978c0b450c3765b24b8abd024d7799e80')
        wandb.init(
            project="llama_vision_vp",  # Replace with your WandB project name
            config=args,            # Logs all hyperparameters
            name=wandb_name,  # Optional: Name your run
            save_code=True                # Optional: Save your code with the run
        )
        wandb.watch(model, log="all")

    # 학습 및 검증 손실 기록을 위한 리스트
    train_losses = []
    val_losses = []

    for epoch in range(args["num_epochs"]):
        model.train()
        epoch_train_loss = 0.0
        if accelerator.is_main_process:
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args['num_epochs']}")
        else:
            progress_bar = train_dataloader

        for real_images, target_images in progress_bar:
            optimizer.zero_grad()

            # 순전파
            loss = model(real_images, target_images, device)

            # 역전파
            accelerator.backward(loss)
            optimizer.step()

            epoch_train_loss += loss.item()

            if accelerator.is_main_process:
                progress_bar.set_postfix({"loss": loss.item()})

            break

        avg_train_loss_tensor = torch.tensor(epoch_train_loss, device=device)
        gathered_train_loss = accelerator.gather(avg_train_loss_tensor).sum() / len(train_dataloader)
        avg_train_loss = gathered_train_loss.item()
        train_losses.append(avg_train_loss)

        # 검증 단계
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for real_images, target_images in val_dataloader:
                loss = model(real_images, target_images, device)
                epoch_val_loss += loss.item()

        avg_val_loss_tensor = torch.tensor(epoch_val_loss, device=device)
        gathered_val_loss = accelerator.gather(avg_val_loss_tensor).sum() / len(val_dataloader)
        avg_val_loss = gathered_val_loss.item()
        val_losses.append(avg_val_loss)

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{args['num_epochs']} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

            # W&B에 손실 값 로깅
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            })

            # 모델 저장
            save_dir = args["save_dir"]
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{args['model_name']}_vp_{epoch}.pth")
            torch.save(model.vp.state_dict(), save_path)
            print(f"Model vp saved to {save_path}")

    if accelerator.is_main_process:
        wandb.finish()