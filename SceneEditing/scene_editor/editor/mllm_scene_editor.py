# mllm_scene_editor.py

import re
import torch
import string
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image

from ..adapter.graph_adapter import GraphAdapter
from ..adapter.vision_encoder import VisionEncoder
from ..editor.parse_actions import parse_relation_changes, apply_relation_changes



class MLLMSceneEditor:
    """
    SG-Nav 아이디어를 일부 반영:
      - Subgraph prompt
      - 'Chain-of-Thought' style self-questioning (간단)
    """

    def __init__(self,
                 mllm_model_name="meta-llama/Llama-3.2-11B-Vision",
                 device="cuda"):
        self.device = device

        # Load LLM
        print(f"[INFO] Loading MLLM from {mllm_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(mllm_model_name)
        self.mm_model = AutoModelForCausalLM.from_pretrained(mllm_model_name)
        self.mm_model.to(self.device)

        # Adapters
        self.graph_adapter = GraphAdapter(use_hierarchy=True)
        self.vision_encoder = VisionEncoder(hidden_dim=768)

    def edit_scene(self, scene_graph: dict, image: Image.Image, user_instruction: str):
        """
        - Extract subgraph relevant to 'user_instruction'
        - Prepare chain-of-thought style prompt
        - Generate action sequence
        - Apply
        - Return updated scene_graph
        """
        # 1) Subgraph extraction
        subgraph = self._extract_relevant_subgraph(scene_graph, user_instruction)
        # print(f"subgraph: {subgraph}")
        # 2) scene_graph_text
        subgraph_text = self.graph_adapter.scene_graph_to_text(subgraph)

        # 3) 이미지 임베딩 (SG-Nav는 이미지→Scene Graph, 여기선 그냥 reference)
        _vision_emb = self.vision_encoder([image]).to(self.device)

        # 4) Build prompt with CoT style
        #    ex) ask the LLM to first reason about what changes are needed, then output actions
        prompt = self._build_cot_prompt(subgraph_text, user_instruction)

        # 5) LLM generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.mm_model.generate(
            **inputs,
            max_new_tokens=62,
            num_beams=1,
            do_sample=True,
            temperature=0.3
        )
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("=== LLM Output ===\n", output_text)

        # 4) Parse changes
        changes = parse_relation_changes(output_text)

        # 5) Apply changes to the FULL scene_graph
        updated_graph = apply_relation_changes(scene_graph, changes)

        return updated_graph

    def _extract_relevant_subgraph(self, scene_graph: dict, user_instruction: str):
        """
        SG-Nav처럼: user_instruction에 언급된 객체/범주/방 을 찾고,
        그 근방의 노드+관계만 모아 subgraph 구성
        (여기선 간단히 'keyword' 매칭)
        """
        keywords = self._extract_keywords(user_instruction)
        objects = scene_graph.get("objects", [])
        relationships = scene_graph.get("relationships", [])

        # 필터된 object list
        filtered_objs = []
        for obj in objects:
            for kw in keywords:
                if kw in obj["category"] or kw in obj["id"]:
                    filtered_objs.append(obj)
                    break

        # relationships: subject/obj 중 하나가 filtered_objs에 속하면 포함
        filtered_rels = []
        filtered_ids = set([o["id"] for o in filtered_objs])
        for rel in relationships:
            subj_id = self._object_id_from_graph(scene_graph, rel["subject"])
            obj_id = self._object_id_from_graph(scene_graph, rel["object"])
            if subj_id in filtered_ids or obj_id in filtered_ids:
                filtered_rels.append(rel)

        subg = {
            "scan_name": scene_graph.get("scan_name", "unknown"),
            "objects": filtered_objs,
            "relationships": filtered_rels
        }
        return subg

    def _object_id_from_graph(self, scene_graph, num_or_str):
        """
        SG-Nav 예시에선 object '1' -> 'wardrobe_1' 식.
        여기선 이미 scene_graph가 {id, category}로 dict.
        단순히 int vs string 구분.
        """
        # demo: if relationship uses an int, map to "category_id"?
        # for now, assume it's already a string.
        return str(num_or_str)

    def _extract_keywords(self, instruction: str):
        """
        매우 간단히 명사만 추출하거나,
        예: "move the wardrobe_1 closer to the bed"
        => keywords = ["wardrobe", "bed"]
        실제론 NLP parse
        """
        words = [w.strip(string.punctuation) for w in instruction.lower().split()]
        # quick hack
        # possible_objs = ["bed", "table", "wardrobe", "chair", "desk", "lamp", "floor", "nightstand", "cabinet"]
        if not hasattr(self, '_possible_objs'):
            try:
                file_path = r"C:\Users\SeongRae Noh\LLM4Layout\SceneEditing\data\SG_FRONT\classes_bedroom.txt"
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.read().splitlines()
                    # 첫 번째 행(헤더)을 제거하고, 공백 제거 후 저장
                    self._possible_objs = [line.strip() for line in lines[1:] if line.strip()]
            except Exception as e:
                print(f"객체 카테고리 파일 로드 중 에러 발생: {e}")
                self._possible_objs = []  # 에러 시 빈 리스트로 처리
        return [w for w in words if w in self._possible_objs]

    def _build_cot_prompt(self, subgraph_text: str, user_instruction: str):
        """
        SG-Nav style chain-of-thought prompt
        ex) 1) "Ask me what you need to know about this subgraph to fulfill the instruction."
            2) LLM self-answers using subgraph_text
            3) Output final action commands
        여기서는 간단히 정적 prompt 예시
        """
        relationships = ['left','right','front','behind','close by','above','standing on','bigger than','smaller than','taller than','shorter than','symmetrical to','same style as']
        prompt = f"""
You are an expert in 3D indoor scene editing.
User instruction: "{user_instruction}"
You have the following subgraph of the scene:
{subgraph_text}

Step-by-step (Chain-of-Thought):
1) Identify the objects or rooms relevant to the target objects .
2) Decide what transformations are needed (Translate? Rotate? Add? Remove?).
3) Generate the final list of actions in the format: ACTION_TYPE(obj_id, obj_id)

Now, let's reason it out:

Now, after your reasoning, please output the final list of actions in the format:
ACTION_TYPE(object_id, param=xxx)
End your answer with the action list only, and do not repeat the entire prompt.
"""
        prompt2 = f"""
        You are an expert in 3D indoor scene editing.
        You have the following subgraph of the scene:
        {subgraph_text}
        User instruction: "{user_instruction}"
        
        You have the following available list of relations:
        
        

        Step-by-step:
        1) To satisfy the user instruction, you must proceed with subgraph editing by editing one or more relations.
        2) Do not return the entire subgraph, but only add, remove, or return the changed relations
        3) The output relation lines should look like:
   "Add Relationship: <object> -> <object> ='relation'"
   "Remove Relationship: <object> -> <object> ='relation'"
        output: 
        """
        prompt3 = f"""
        You are an expert in 3D indoor scene editing.

You have the following subgraph of the scene:
{subgraph_text}

User instruction: "{user_instruction}"

Additional Constraints:
1) Even if the user instruction is brief, you must ensure the room remains coherent and realistic.  
   - If there are missing steps or changes needed to maintain a consistent layout (e.g., repositioning objects, removing overlaps), you should add them.  
   - To do this, create a detailed version of the user instruction and create an output based on it.

2) Edit the scene by adding, removing, or updating **relationships** in the subgraph.  
   - Do not repeat or return the entire subgraph; only output the lines corresponding to changed relationships.  
   - Check if extra modifications are required for a coherent, realistic room arrangement.

3) Include all additional changes that you think are necessary for a natural indoor layout, even if not explicitly stated by the user.
Write detailed version of the user instructions and corresponding changes of relation in following format
<detailed version of instruction>, <Add/Remove> <object1> -> <object2> = <relation>
output:
"""
        answerPrompt="Use the image and scene graph as context and answer the following question: "
        answerPrompt2="Use the image and context to answer the following question: "
        sgPrompt="""
        For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
        1. Objects that are relevant to answering the question
        2. Object attributes that are relevant to answering the question
        3. Object relationships that are relevant to answering the question
        
        Scene Graph:
        """
        prompt_subgraphExtraction = f"""
        You are an expert in 3D indoor scene analysis.
        Task: From the given top-view indoor scene image, the existing scene graph, and the user's instruction,
        extract the relevant subgraph (both spatial and semantic) that pertains to the user's request.
        1) Identify only the nodes and edges in the scene graph that are crucial for fulfilling the user's instruction.
        2) Distinguish spatial edges (e.g., left to, near, behind) from semantic edges (e.g., same style as, same material).
        3) Return the result in JSON format, containing:
            - "spatial_subgraph": { 'nodes': [...], 'edges': [...] }
            - "semantic_subgraph": { 'nodes': [...], 'edges': [...] }
        Do not provide the entire scene graph. Only relevant subgraphs.
        Output:
        """

        prompt_editing=f"""
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

Do not output anything else except this JSON. 
If additional edits are implicitly required (e.g., repositioning for a realistic arrangement),
please include them as well.
Now let's reason it out:
"""
        return prompt3
