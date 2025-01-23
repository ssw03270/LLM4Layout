import data_utils
import pre_utils
import train_utils

if __name__ == "__main__":
    args = pre_utils.parse_args()
    pre_utils.set_seed(args["seed"])
    args["device"] = pre_utils.get_device()
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

    for epoch in range(args["num_epochs"]):
        for source_image, target_image in train_dataloader:
            outputs = model(source_image, target_image)
            for output in outputs:
                print(output)
            exit()