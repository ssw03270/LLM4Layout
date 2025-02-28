# scene_updater.py

from scene_editor.editor.action_definitions import Action

def apply_action(scene_graph: dict, action: Action):
    """
    - collision-free 전제
    - Modify the target object's (position, rotation) or other attributes
    """
    if action.action_type == "Translate":
        _apply_translate(scene_graph, action)
    elif action.action_type == "Rotate":
        _apply_rotate(scene_graph, action)
    elif action.action_type == "Remove":
        _apply_remove(scene_graph, action)
    # ...
    return scene_graph

def _apply_translate(scene_graph, action: Action):
    obj = _find_obj(scene_graph, action.target_object)
    if not obj:
        return
    dist = action.params.get("distance", 0.0)
    # 간단히 x축 이동
    obj["position"][0] += dist
    obj["updated"] = True

def _apply_rotate(scene_graph, action: Action):
    obj = _find_obj(scene_graph, action.target_object)
    if not obj:
        return
    angle = action.params.get("angle", 0.0)
    # z축 회전
    obj["rotation"][2] += angle
    obj["updated"] = True

def _apply_remove(scene_graph, action: Action):
    obj_id = action.target_object
    scene_graph["objects"] = [o for o in scene_graph["objects"] if o["id"] != obj_id]

def _find_obj(scene_graph, obj_id: str):
    for o in scene_graph["objects"]:
        if o["id"] == obj_id:
            return o
    return None
