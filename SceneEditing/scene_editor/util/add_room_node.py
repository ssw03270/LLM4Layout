# pseudo-code 예시: scene_editor.util.add_room_node.py

def add_room_node(scene_graph: dict, room_name: str, object_ids: list):
    """
    scene_graph에 'room' type 노드(예: room_1)를 추가하고,
    object_ids에 속하는 객체들은 해당 room 노드와 연결(relationships).
    """
    room_node_id = f"{room_name}_room"
    # room node를 objects에 추가
    scene_graph["objects"].append({
        "id": room_node_id,
        "category": "room",
        "position": [0,0,0],
        "rotation": [0,0,0],
        "updated": False
    })
    # relationships에서 room_node_id와 각 object_ids를 연결
    for obj_id in object_ids:
        scene_graph["relationships"].append({
            "subject": room_node_id,
            "object": obj_id,
            "label": "contains"
        })
    return scene_graph
