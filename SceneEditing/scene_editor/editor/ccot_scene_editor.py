import os
import json
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..adapter.vision_encoder import VisionEncoder
from ..editor.parse_actions import apply_relation_changes, parse_relation_changes

possible_categories = ["armchair","bookshelf","cabinet","ceiling_lamp","chair","children_cabinet",
                       "coffee_table","desk","double_bed","dressing_chair","dressing_table","kids_bed",
                       "nightstand","pendant_lamp","shelf","single_bed","sofa","stool","table","tv_stand",
                       "wardrobe","floor"]

possible_relations = [
    "left","right","front","behind","close by","above","standing on","bigger than","smaller than",
    "taller than","shorter than","symmetrical to","same style as","same super category as",
    "same material as"
]

subgraph_extraction_prompt = f"""
You are an expert in 3D indoor scene analysis.

Task: From the given top-view indoor scene image, the existing scene graph, and the user instruction,
extract **only the relevant subgraph** necessary for fulfilling the user's request.

1. Include **only** the nodes (objects) whose categories are directly relevant or needed for a realistic arrangement.
2. Separate **spatial** edges (e.g., left, right, behind, above, etc.) from **semantic** edges (e.g., same style as, same material).
3. Return your result in **JSON** with exactly two keys:
   - "spatial_subgraph": {{ "nodes": [...], "edges": [...] }},
   - "semantic_subgraph": {{ "nodes": [...], "edges": [...] }}
   Each "edges" list should contain relations in the format:
       {{ "subject": <node_id>, "object": <node_id>, "relation": <relation_label> }}.

**Do NOT** return the entire scene graph. Provide only these two subgraphs.

You can use the following categories:
{", ".join(possible_categories)}

You can use the following relations:
{", ".join(possible_relations)}

Output **only** the JSON described above, with no additional commentary:
"""



editing_prompt = f"""
You are an expert in 3D indoor scene editing.

Task: Using the top-view image, the previously extracted subgraphs (spatial + semantic),
and the user instruction, determine what relations in the subgraph must be changed, added, or removed
to satisfy the instruction while keeping the layout realistic.

Return the changes as JSON in the format:
{{
  "changed_relations": [
    {{ "op": "Add|Remove|Update", "subject": "<node_id>", "object": "<node_id>", "relation": "<label>" }},
    ...
  ]
}}
You can use the following categories:
{", ".join(possible_categories)}

You can use the following relations:
{", ".join(possible_relations)}

Do not output anything else except this JSON.
output:
"""



################################################################################
# 3) CCoTSceneEditor: two-step pipeline
################################################################################
class CCoTSceneEditor:
    def __init__(self,
                 model_name="meta-llama/Llama-3.2-11B-Vision",
                 device="cuda"):
        print(f"[INFO] Loading MLLM from {model_name}")
        self.device = device

        # Load Model & Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mm_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        self.vision_encoder = VisionEncoder(hidden_dim=768)

    def run_ccot_pipeline(self, scene_graph: dict, image: Image.Image, user_instruction: str):
        """
        2단계:
          (1) subgraph extraction
          (2) final relation editing
        """
        # Step1
        subgraph_json_str = self._subgraph_extraction_step(scene_graph, image, user_instruction)

        # parse subgraph
        try:
            subgraph_data = json.loads(subgraph_json_str)
        except:
            print("[WARN] Subgraph JSON parse failed.")
            subgraph_data = {"spatial_subgraph":{},"semantic_subgraph":{},"raw_output": subgraph_json_str}

        # Step2
        relation_changes_str = self._relation_editing_step(image, user_instruction, subgraph_json_str)

        # parse changes
        try:
            changes_data = json.loads(relation_changes_str)
            if "changed_relations" not in changes_data:
                changes_data["changed_relations"] = []
        except:
            print("[WARN] changes JSON parse failed.")
            changes_data = {"changed_relations":[],"raw_output": relation_changes_str}

        # apply
        updated_scene_graph = apply_relation_changes(scene_graph, changes_data["changed_relations"])
        return {
            "extracted_subgraph": subgraph_data,
            "relation_changes": changes_data,
            "updated_scene_graph": updated_scene_graph
        }

    ############################################################################
    # Step1: subgraph extraction
    ############################################################################
    def _subgraph_extraction_step(self, scene_graph, image, user_instruction):
        sg_str = json.dumps(scene_graph, indent=2)
        # # fill the template with possible categories and relations
        # prompt = subgraph_extraction_prompt.format(
        #     categories="\n".join(possible_categories),
        #     relations=", ".join(possible_relations)
        # )
        # Just use + to append the relevant text
        prompt = subgraph_extraction_prompt + f"\nExisting Scene Graph:\n{sg_str}\n\nUser instruction: {user_instruction}\n"

        output_str = self._run_inference(prompt, image)
        print("=== Subgraph Extraction Output ===\n", output_str)
        return output_str

    ############################################################################
    # Step2: relation editing
    ############################################################################
    def _relation_editing_step(self, image, user_instruction, subgraph_json_str):
        # prompt = editing_prompt.format(
        #     relations=", ".join(possible_relations)
        # )
        prompt = editing_prompt+f"\nUser instruction: {user_instruction}\nExtracted Subgraphs:\n{subgraph_json_str}\n"
        output_str = self._run_inference(prompt, image)
        print("=== Relation Editing Output ===\n", output_str)
        return output_str

    ############################################################################
    # Low-level inference method
    ############################################################################
    def _run_inference(self, prompt: str, image: Image.Image):
        """
        minimal example: text prompt + single image => LLM decode
        """
        # optional vision encode if we want
        _vision_emb = self.vision_encoder([image]).to(self.device)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.mm_model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=1,
                do_sample=True,
                temperature=0.3
            )
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text.strip()
