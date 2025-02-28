# full_graph_mllm_scene_editor.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image

from ..adapter.graph_adapter import GraphAdapter
from ..adapter.vision_encoder import VisionEncoder
from ..editor.parse_actions import parse_mllm_output_to_actions
from ..editor.scene_updater import apply_action


# collision-free라서 collision 관련 로직 없음

class MLLMSceneEditor:
    """
    1) Scene Graph -> Embeddings
    2) Image -> Embeddings
    3) User Instruction -> Tokens
    4) (graph + vision + text) -> MLLM -> Action Sequence
    5) Parse & Apply
    """

    def __init__(self,
                 mllm_model_name: str = "meta-llama/Llama-3.2-11B-Vision",
                 device="cuda"):
        self.device = device

        # (A) Load LLM
        print(f"[INFO] Loading MLLM from {mllm_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(mllm_model_name)
        self.mm_model = AutoModelForCausalLM.from_pretrained(mllm_model_name)
        self.mm_model.to(self.device)

        # (B) Adapters
        self.graph_adapter = GraphAdapter(hidden_dim=768)
        self.vision_encoder = VisionEncoder(hidden_dim=768)

    def edit_scene(self, scene_graph: dict, image: Image.Image, user_instruction: str):
        # (1) Embeddings
        graph_emb = self.graph_adapter(scene_graph).to(self.device)  # (1, G, 768)
        vis_emb = self.vision_encoder([image]).to(self.device)  # (1, I, 768)

        # (2) User text
        text_enc = self.tokenizer(user_instruction, return_tensors="pt").to(self.device)

        # (3) Combine
        combined_input = self._combine_inputs(graph_emb, vis_emb, text_enc)

        # (4) Generate
        output_ids = self.mm_model.generate(
            **combined_input,
            max_length=256,
            num_beams=1,
            do_sample=True,
            temperature=0.7
        )
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print("[DEBUG] MLLM raw output:\n", output_text)

        # (5) Parse -> Action list
        actions = parse_mllm_output_to_actions(output_text)

        # (6) Apply
        updated_graph = scene_graph
        for act in actions:
            updated_graph = apply_action(updated_graph, act)

        return updated_graph

    def _combine_inputs(self, graph_emb, vis_emb, text_enc):
        """
        실제론 multi-modal attention mask/방식이 모델마다 다름.
        여기서는 pseudo-code로.
        """
        # For demonstration: treat user_instruction as normal input_ids
        # and assume graph_emb, vis_emb are fed as prefix embeddings inside the model
        # In real usage, you'd integrate with a custom multi-modal forward pass.
        combined_input = {
            "input_ids": text_enc["input_ids"],
            "attention_mask": text_enc["attention_mask"]
            # "graph_embedding": graph_emb,
            # "vision_embedding": vis_emb,
            # ...
        }
        return combined_input
