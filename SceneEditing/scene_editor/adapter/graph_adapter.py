# adapter/graph_adapter.py

"""
GNN 없이, Scene Graph를 prompt-friendly 텍스트로 직렬화하여
LLM 입력에 사용.
"""

import json


class GraphAdapter:
    """
    - scene_graph(dict) -> scene_graph_text(str)
    - 예시: SG-Nav 스타일의 'hierarchical' or 'flat' 텍스트 변환
    """

    def __init__(self, use_hierarchy=True):
        """
        use_hierarchy=True 시, "Room / Group / Object" 구조를
        텍스트에 녹여서 계층적으로 표현.
        """
        self.use_hierarchy = use_hierarchy

    def scene_graph_to_text(self, scene_graph: dict) -> str:
        """
        scene_graph: {
          "objects": [...],
          "relationships": [...],
          "scan_name": ...
          ...
        }
        return: str
        """
        if self.use_hierarchy:
            return self._hierarchical_text(scene_graph)
        else:
            return self._flat_text(scene_graph)

    def _hierarchical_text(self, scene_graph: dict) -> str:
        """
        SG-Nav 식으로 room/group/object 구조를 텍스트화.
        여기선 예시로, room node나 group node가 있다 가정.
        실제 scene_graph 구조 설계에 따라 달라질 수 있음.
        """
        # 실제론 scene_graph["rooms"], scene_graph["groups"] 등이 필요.
        # 여기서는 objects + relationships로만 일단 예시
        obj_lines = []
        for obj in scene_graph.get("objects", []):
            # "Object desk_1 (category=desk) position=[2.0,0.0,0.0]"
            line = f"Object {obj['id']} (category={obj['category']})"
            obj_lines.append(line)

        rel_lines = []
        for rel in scene_graph.get("relationships", []):
            # "Relationship: subject=desk_1, object=bed_1, label='next to'"
            rel_str = f"Relationship: {rel['subject']} -> {rel['object']} = '{rel['label']}'"
            rel_lines.append(rel_str)

        text = f"# Scene Graph for {scene_graph.get('scan_name', 'unknown')}\n"
        text += "## Objects:\n"
        text += "\n".join(obj_lines) + "\n"
        text += "## Relationships:\n"
        text += "\n".join(rel_lines) + "\n"
        # print(f"scene_graph: {scene_graph}")
        # print(f"text: {text}, obj_lines: {obj_lines}, rel_lines: {rel_lines}")
        return text

    def _flat_text(self, scene_graph: dict) -> str:
        """
        객체/관계를 그냥 나열 (간단 버전)
        """
        lines = []
        lines.append(f"Scene Name: {scene_graph.get('scan_name', 'unknown')}")
        lines.append("Objects:")
        for obj in scene_graph.get("objects", []):
            lines.append(f"  - {obj['id']}({obj['category']})")

        lines.append("Relationships:")
        for rel in scene_graph.get("relationships", []):
            lines.append(f"  - {rel['subject']} {rel['label']} {rel['object']}")

        return "\n".join(lines)
