import data_utils
import pre_utils
import train_utils

import os
from tqdm import tqdm
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

        avg_train_loss = accelerator.gather(torch.tensor(epoch_train_loss)).sum() / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # 검증 단계
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for real_images, target_images in val_dataloader:
                loss = model(real_images, target_images, device)
                epoch_val_loss += loss.item()

        avg_val_loss = accelerator.gather(torch.tensor(epoch_val_loss)).sum() / len(val_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{args['num_epochs']} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        if accelerator.is_main_process:
            # 모델 저장
            save_dir = args["save_dir"]
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{args['model_name']}_vp_{epoch}.pth")
            torch.save(model.vp.state_dict(), save_path)
            print(f"Model vp saved to {save_path}")