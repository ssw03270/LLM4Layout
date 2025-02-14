import numpy as np
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

def generate_text(raw_data, room_type, task_type="remaining values"):
    user_prompt = "I want to generate layout in {Domain} style. Please generate the layout according to the {Task_Condition} I provide:"
    code_template = """```
<html>
    <body>
{code}
    </body>
</html>
```"""
    assistant_prompt = """```
<html>
    <body>
{code}
    </body>
</html>
```"""
    messages = []

    text_class_labels = raw_data._class_labels
    min_translation = raw_data._centroids[0]
    for data_idx, data in enumerate(raw_data):
        scene_id = data.scene_id
        class_labels = data.class_labels
        translations = data.translations
        sizes = data.sizes
        angles = data.angles
        captions = data.captions

        element_count = len(class_labels)
        gt_layout_text = ""
        masked_layout_text = ""
        for element_idx in range(element_count):
            class_label = class_labels[element_idx]
            class_label = text_class_labels[np.argmax(class_label)]
            trans = translations[element_idx] - min_translation
            size = sizes[element_idx]
            angle = np.rad2deg(angles[element_idx][0])
            caption = captions[element_idx]

            gt_element_text = (f"<rect data-category=\"{class_label}\" caption=\"{caption}\" "
                               f"transform=\"translate3d({trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f}) "
                               f"scale3d({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}) "
                               f"rotateY({angle:.2f})\"/>")
            masked_element_text = (f"<rect data-category={class_label} caption=\"{caption}\" "
                                   f"transform=\"translate3d(<FILL_x>, <FILL_y>, <FILL_z>) "
                                   f"scale3d(<FILL_x>, <FILL_y>, <FILL_z>) "
                                   f"rotateY(<FILL_deg>)\"/>")

            gt_layout_text += f"        {gt_element_text}"
            masked_layout_text += f"        {masked_element_text}"

            if element_idx < element_count - 1:
                gt_layout_text += "\n"
                masked_layout_text += "\n"

        _user_prompt = user_prompt.format(Domain=room_type, Task_Condition=task_type)
        _code_template = code_template.format(code=masked_layout_text)
        _user_prompt += "\n" + _code_template

        _assistant_prompt = assistant_prompt.format(code=gt_layout_text)

        message = {
            {"role": "user", "content": _user_prompt},
            {"role": "assistant", "content": _assistant_prompt}
        }
        messages.append(message)

    return messages

def main():
    room_type = "bedroom"
    config_file = f"./configs/{room_type}_sg2sc_diffusion_objfeat.yaml"
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
    train_messages = generate_text(train_raw, room_type)

    val_raw = get_raw_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        path_to_bounds=None,
        split=config["validation"].get("splits", ["test"])
    )
    val_messages = generate_text(val_raw, room_type)

if __name__ == "__main__":
    main()


