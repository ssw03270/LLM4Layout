import random
from tqdm import tqdm
from shapely.geometry import Polygon

from utils import *


def process_scene(folder_path, room, data_type="test"):
    """
    Loads split data, statistics, and all relevant scene information for each subfolder.
    Constructs polygons for the room and the objects, then visualizes them.
    """
    stats = load_stats(os.path.join(folder_path, room, "dataset_stats.txt"))
    retrieval_results = load_retrieval_results(os.path.join("../datasets", room, "retrieval_results.pkl"))[data_type]

    dataset = load_dataset(os.path.join("../datasets", room, "dataset.pkl"))
    source_polygons_dataset = dataset[data_type]["polygons"]
    source_layout_dataset = dataset[data_type]["layout"]
    retrieved_polygons_dataset = dataset["train"]["polygons"]
    retrieved_layout_dataset = dataset["train"]["layout"]

    for idx, (source_file_path, _) in enumerate(tqdm(source_polygons_dataset.items())):
        source_object_polygons = [Polygon(object) for object in
                                  source_polygons_dataset[source_file_path]["object_polygons"]]
        source_room_polygon = Polygon(source_polygons_dataset[source_file_path]["room_polygon"])
        source_layout = source_layout_dataset[source_file_path]

        retrieved_file_path = retrieval_results[source_file_path][0]
        retrieved_object_polygons = [Polygon(object) for object in
                                     retrieved_polygons_dataset[retrieved_file_path]["object_polygons"]]
        retrieved_room_polygon = Polygon(retrieved_polygons_dataset[retrieved_file_path]["room_polygon"])
        retrieved_layout = retrieved_layout_dataset[retrieved_file_path]

        # Visualization
        source_boundary_image = generate_polygons_image(source_room_polygon, source_object_polygons, source_layout,
                                                        stats, draw_type="boundary")
        source_gt_image = generate_polygons_image(source_room_polygon, source_object_polygons, source_layout,
                                                  stats, draw_type="all")
        retrieved_boundary_image = generate_polygons_image(retrieved_room_polygon, retrieved_object_polygons,
                                                     retrieved_layout, stats, draw_type="boundary")
        retrieved_gt_image = generate_polygons_image(retrieved_room_polygon, retrieved_object_polygons,
                                                     retrieved_layout, stats, draw_type="all")

        combined_image = Image.fromarray(np.hstack((retrieved_gt_image, source_boundary_image)))
        source_gt_image = Image.fromarray(source_gt_image)
        source_boundary_image = Image.fromarray(source_boundary_image)
        retrieved_gt_image = Image.fromarray(retrieved_gt_image)
        retrieved_boundary_image = Image.fromarray(retrieved_boundary_image)

        retrieved_json = layout_to_json(retrieved_layout, stats)
        source_json = layout_to_json(source_layout, stats)

        save_path = "../models/ollama/datasets"
        save_image(combined_image, f"{source_file_path}_combined_image", save_path)
        save_image(source_gt_image, f"{source_file_path}_source_gt_image", save_path)
        save_image(source_boundary_image, f"{source_file_path}_source_boundary_image", save_path)
        save_image(retrieved_gt_image, f"{source_file_path}_retrieved_gt_image", save_path)
        save_image(retrieved_boundary_image, f"{source_file_path}_retrieved_boundary_image", save_path)

        save_json(retrieved_json, f"{source_file_path}_retrieved_json", save_path)
        save_json(source_json, f"{source_file_path}_source_json", save_path)


def main():
    """
    Sets up the folder paths, rooms, seeds, and initiates processing for each room type.
    """
    random.seed(42)
    np.random.seed(42)

    folder_path = 'E:/Resources/IndoorSceneSynthesis/InstructScene'
    room_lists = ["threed_front_diningroom", "threed_front_livingroom", "threed_front_bedroom"]

    for room in room_lists:
        process_scene(folder_path, room)


if __name__ == "__main__":
    main()
