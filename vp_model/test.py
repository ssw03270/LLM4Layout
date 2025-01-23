import data_utils
import pre_utils
import train_utils

import os

from tqdm import tqdm

import torch
import numpy as np
from PIL import Image

if __name__ == "__main__":
    args = pre_utils.parse_args()
    pre_utils.set_seed(args["seed"])
    dataset_paths_dict = data_utils.get_dataset_paths(args["dataset_dir"])
    split_dataset_paths = data_utils.split_dataset(dataset_paths_dict)

    test_dataset = data_utils.LayoutDataset(split_dataset_paths, "test")

    test_dataloader = data_utils.get_dataloader(test_dataset, args, shuffle=False)

    vlm_model, vp_model = train_utils.build_test_model(args, model_path=args["model_path"])
    optimizer = train_utils.get_optimizer(vp_model, args)
    scheduler = train_utils.get_scheduler(optimizer, args)

    test_dataloader, vlm_model, vp_model, optimizer, scheduler, accelerator = train_utils.get_test_accelerator(
        test_dataloader, vlm_model, vp_model, optimizer, scheduler)
    device = accelerator.device

    vlm_model.eval()
    if accelerator.is_main_process:
        test_progress_bar = tqdm(test_dataloader, desc=f"Test")
    else:
        test_progress_bar = test_dataloader

    # 검증 단계
    vp_model.eval()
    with torch.no_grad():
        for idx, real_images, target_images in enumerate(test_progress_bar):
            real_inputs, target_inputs = accelerator.unwrap_model(vlm_model).get_inputs(real_images, target_images, device)
            target_inputs["pixel_values"] = vp_model(target_inputs["pixel_values"])
            real_texts, target_texts, = vlm_model.generate(real_inputs, target_inputs)

            for batch_idx in range(real_images.shape[0]):
                file_name = f"{idx}_{batch_idx}_{device}"

                real_image = Image.fromarray(real_images[batch_idx])
                real_text = real_texts[batch_idx].cpu().detach().numpy()
                target_image = Image.fromarray(target_images[batch_idx])
                target_text = target_texts[batch_idx].cpu().detach().numpy()

                save_dir = "./test_outputs"
                os.makedirs(save_dir, exist_ok=True)
                real_image.save(os.path.join(save_dir, file_name + "real_image.png"))
                with open(os.path.join(save_dir, file_name + "real_text.png")) as file:
                    file.write(real_text)
                target_image.save(os.path.join(save_dir, file_name + "target_image.png"))
                with open(os.path.join(save_dir, file_name + "target_text.png")) as file:
                    file.write(target_text)
