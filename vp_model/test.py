import data_utils
import pre_utils
import train_utils

import os

from tqdm import tqdm

import numpy as np
from PIL import Image

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

    test_dataset = data_utils.LayoutDataset(split_dataset_paths, "test")

    test_dataloader = data_utils.get_dataloader(test_dataset, args, shuffle=False)

    zero2_plugin = DeepSpeedPlugin(hf_ds_config="zero2_config.json")
    zero3_plugin = DeepSpeedPlugin(hf_ds_config="zero3_config.json")

    accelerator = Accelerator()

    vlm_model, vp_model = train_utils.build_test_model(args, model_path=args["model_path"])

    test_dataloader, vlm_model, vp_model = train_utils.get_test_accelerator(test_dataloader, vlm_model, vp_model, accelerator)
    device = accelerator.device

    vlm_model.eval()
    if accelerator.is_main_process:
        test_progress_bar = tqdm(test_dataloader, desc=f"Test")
    else:
        test_progress_bar = test_dataloader

    # 검증 단계
    vp_model.eval()
    with torch.no_grad():
        for idx, (real_images, target_images, text_descriptions) in enumerate(test_progress_bar):
            real_inputs, target_inputs, prompts = accelerator.unwrap_model(vlm_model).get_inputs(real_images, target_images, text_descriptions, device)
            target_inputs["pixel_values"] = vp_model(target_inputs["pixel_values"])
            real_texts, target_texts, = vlm_model.generate(real_inputs, target_inputs)

            for batch_idx in range(real_images.shape[0]):
                file_name = f"{idx}_{batch_idx}_{device}"

                real_image = Image.fromarray(real_images[batch_idx].cpu().detach().numpy())
                real_text = real_texts[batch_idx]
                target_image = Image.fromarray(target_images[batch_idx].cpu().detach().numpy())
                target_text = target_texts[batch_idx]

                prompt = prompts[batch_idx]

                save_dir = "./test_outputs"
                os.makedirs(save_dir, exist_ok=True)
                real_image.save(os.path.join(save_dir, file_name + "_real_image.png"))
                with open(os.path.join(save_dir, file_name + "_real_text.txt"), "w", encoding="utf-8") as file:
                    file.write(real_text)
                target_image.save(os.path.join(save_dir, file_name + "_target_image.png"))
                with open(os.path.join(save_dir, file_name + "_target_text.txt"), "w", encoding="utf-8") as file:
                    file.write(target_text)

                with open(os.path.join(save_dir, file_name + "_prompt.txt"), "w", encoding="utf-8") as file:
                    file.write(prompt)
