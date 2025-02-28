# parse_relation_changes.py

import re


def parse_relation_changes(output_text: str):
    """
    LLM이 다음과 같은 포맷으로 반환한다고 가정:
      "Add Relationship: table_3 -> cabinet_4 ='left'"
      "Remove Relationship: table_3 -> bed_1 ='next to'"
      "Relationship: desk_1 -> bed_1 ='behind'"

    returns: list of dict, e.g.
    [
      { "op": "Add", "subject": "table_3", "object": "cabinet_4", "relation": "left" },
      { "op": "Remove", "subject": "table_3", "object": "bed_1", "relation": "next to" },
      { "op": "AddOrUpdate", "subject": "desk_1", "object": "bed_1", "relation": "behind" }
    ]
    """
    changes = []
    lines = output_text.strip().split("\n")

    # Regex: capture group(1) = 'Add'|'Remove'|'Relationship'
    #        group(2) = subject
    #        group(3) = object
    #        group(4) = relation label
    pattern = re.compile(
        r"^(Add|Remove|Relationship)\s*(?:Relationship:)?\s*(\S+)\s*->\s*(\S+)\s*=\s*'([^']+)'"
    )

    for line in lines:
        line = line.strip()
        match = pattern.match(line)
        if match:
            cmd_type = match.group(1)  # e.g. "Add" or "Remove" or "Relationship"
            subj = match.group(2)
            obj = match.group(3)
            rel = match.group(4)

            if cmd_type == "Add":
                operation = "Add"
            elif cmd_type == "Remove":
                operation = "Remove"
            else:
                operation = "AddOrUpdate"

            changes.append({
                "op": operation,
                "subject": subj,
                "object": obj,
                "relation": rel
            })
    return changes

# apply_relation_changes.py

def apply_relation_changes(scene_graph: dict, changes: list):
    """
    changes: list of dict
      { "op": "Add"/"Remove"/"AddOrUpdate", "subject":..., "object":..., "relation":... }
    scene_graph:
      {
        "objects": [...],
        "relationships": [
          {"subject": str, "object": str, "label": str}, ...
        ]
      }

    returns: updated scene_graph
    """

    # print(changes)
    if "relationships" not in scene_graph:
        scene_graph["relationships"] = []

    for change in changes:
        op = change["op"]
        subj = change["subject"]
        obj = change["object"]
        label = change["relation"]

        if op == "Add":
            # 단순히 추가
            scene_graph["relationships"].append({
                "subject": subj,
                "object": obj,
                "label": label
            })

        elif op == "Remove":
            # subject==subj & object==obj & label==label
            scene_graph["relationships"] = [
                r for r in scene_graph["relationships"]
                if not (r["subject"] == subj and r["object"] == obj and r["label"] == label)
            ]

        elif op == "AddOrUpdate":
            # 먼저 동일 subject/object의 기존 관계 검색
            found = False
            for r in scene_graph["relationships"]:
                if r["subject"] == subj and r["object"] == obj:
                    r["label"] = label
                    found = True
                    break
            if not found:
                # 없으면 새로 추가
                scene_graph["relationships"].append({
                    "subject": subj,
                    "object": obj,
                    "label": label
                })

    return scene_graph

