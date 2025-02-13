import data_utils
import pre_utils
import train_utils

import os
from datetime import datetime

from tqdm import tqdm
import wandb

import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

from huggingface_hub import login

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

with open("api_key.txt", "r") as f:
    api_key = f.read().strip()  # 공백 제거
os.environ["HF_TOKEN"] = api_key
login(api_key)

if __name__ == "__main__":
    args = pre_utils.parse_args()
    pre_utils.set_seed(args["seed"])
    dataset_paths_dict = data_utils.get_dataset_paths(args["dataset_dir"])
    split_dataset_paths = data_utils.split_dataset(dataset_paths_dict)

    train_dataset = data_utils.LayoutDataset(split_dataset_paths, "train")
    val_dataset = data_utils.LayoutDataset(split_dataset_paths, "val")

    train_dataloader = data_utils.get_dataloader(train_dataset, args, shuffle=True)
    val_dataloader = data_utils.get_dataloader(val_dataset, args, shuffle=False)

    # zero2_plugin = DeepSpeedPlugin(hf_ds_config="zero2_config.json")
    # zero3_plugin = DeepSpeedPlugin(hf_ds_config="zero3_config.json")
    #
    # deepspeed_plugins = {"student": zero2_plugin, "teacher": zero3_plugin}
    # accelerator = Accelerator(deepspeed_plugins=deepspeed_plugins)
    accelerator = Accelerator()

    vlm_model, vp_model = train_utils.build_model(args)
    optimizer = train_utils.get_optimizer(vp_model, accelerator, args["learning_rate"])
    scheduler = train_utils.get_scheduler(optimizer, accelerator, train_dataloader, args["num_epochs"])

    train_dataloader, val_dataloader, vlm_model, vp_model, optimizer, scheduler = train_utils.get_accelerator(
        train_dataloader, val_dataloader, vlm_model, vp_model, optimizer, scheduler, accelerator)
    device = accelerator.device

    if accelerator.is_main_process:
        wandb_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 형식: YYYY-MM-DD_HH-MM-SS
        wandb.login(key='0f272b4978c0b450c3765b24b8abd024d7799e80')
        wandb.init(
            project="llama_vision_vp2",  # Replace with your WandB project name
            config=args,            # Logs all hyperparameters
            name=wandb_name,  # Optional: Name your run
            save_code=True                # Optional: Save your code with the run
        )
        wandb.watch(vp_model, log="all")

    # 학습 및 검증 손실 기록을 위한 리스트
    train_losses = []
    val_losses = []

    vlm_model.eval()
    for epoch in range(args["num_epochs"]):
        vp_model.train()
        epoch_train_loss = 0.0
        if accelerator.is_main_process:
            train_progress_bar = tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}/{args['num_epochs']}")
        else:
            train_progress_bar = train_dataloader

        for real_images, target_images, text_descriptions, real_image_path, target_image_path in train_progress_bar:
            # 순전파
            real_inputs, _ = accelerator.unwrap_model(vlm_model).get_inputs(real_images, real_image_path, text_descriptions, device)
            target_inputs, _ = accelerator.unwrap_model(vlm_model).get_inputs(target_images, target_image_path, text_descriptions, device)

            target_inputs["pixel_values"] = vp_model(target_inputs["pixel_values"])
            loss = vlm_model(real_inputs, target_inputs)

            # 역전파
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_train_loss += loss.item()

            if accelerator.is_main_process:
                train_progress_bar.set_postfix({"loss": loss.item()})


        avg_train_loss_tensor = torch.tensor(epoch_train_loss, device=device)
        gathered_train_loss = accelerator.gather(avg_train_loss_tensor).sum() / (len(train_dataloader) * accelerator.num_processes)
        avg_train_loss = gathered_train_loss.item()
        train_losses.append(avg_train_loss)

        if accelerator.is_main_process:
            val_progress_bar = tqdm(val_dataloader, desc=f"Val Epoch {epoch + 1}/{args['num_epochs']}")
        else:
            val_progress_bar = val_dataloader

        # 검증 단계
        vp_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for real_images, target_images, text_descriptions, real_image_path, target_image_path in val_progress_bar:
                real_inputs, _ = accelerator.unwrap_model(vlm_model).get_inputs(real_images, real_image_path, text_descriptions, device)
                target_inputs, _ = accelerator.unwrap_model(vlm_model).get_inputs(target_images, target_image_path, text_descriptions, device)

                target_inputs["pixel_values"] = vp_model(target_inputs["pixel_values"])
                loss = vlm_model(real_inputs, target_inputs)

                epoch_val_loss += loss.item()

                if accelerator.is_main_process:
                    val_progress_bar.set_postfix({"loss": loss.item()})

        avg_val_loss_tensor = torch.tensor(epoch_val_loss, device=device)
        gathered_val_loss = accelerator.gather(avg_val_loss_tensor).sum() / (len(val_dataloader) * accelerator.num_processes)
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
            model_name = args['model_name'].replace('/', '_')
            save_path = os.path.join(save_dir, f"{model_name}_vp_{epoch}.pth")
            torch.save(accelerator.unwrap_model(vp_model).state_dict(), save_path)
            print(f"Model vp saved to {save_path}")

    if accelerator.is_main_process:
        wandb.finish()