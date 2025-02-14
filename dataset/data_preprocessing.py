import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from data import get_raw_dataset, filter_function

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config

config_file = "./configs/bedroom_sg2sc_diffusion_objfeat.yaml"
config = load_config(config_file)

train_raw = get_raw_dataset(
    config["data"],
    filter_function(
        config["data"],
        split=config["training"].get("splits", ["train", "val"])
    ),
    path_to_bounds=None,
    split=config["training"].get("splits", ["train", "val"]),
)

val_raw = get_raw_dataset(
    config["data"],
    filter_function(
        config["data"],
        split=config["validation"].get("splits", ["test"])
    ),
    path_to_bounds=None,
    split=config["validation"].get("splits", ["test"])
)

for data in train_raw:
    scene_id = data.scene_id
    class_labels = data.class_labels
    translations = data.translations
    sizes = data.sizes
    angles = data.angles
    print(scene_id)