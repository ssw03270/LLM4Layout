# adapter/gnn_adapter.py

import torch
import torch.nn as nn


class GraphAdapter(nn.Module):
    """
    Scene Graph -> Token Embeddings
    (Collision-free 전제이므로 bounding box, collision check 등은 불필요)

    - 가구, 위치, 회전, 스케일, 기타 속성 등
    - GNN을 쓸 수도, 단순 Embedding을 쓸 수도 있음.
    """

    def __init__(self, hidden_dim=768, category_vocab_size=1000):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 예시: Category Embedding
        self.category_embedding = nn.Embedding(category_vocab_size, hidden_dim)

        # 위치, 회전 등을 처리하는 작은 MLP
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rot_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Transformer input projection
        self.proj = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, scene_graph: dict):
        """
        scene_graph: {
          "objects": [
             {
               "id": "bed_1",
               "category": "bed",
               "position": [x, y, z],
               "rotation": [rx, ry, rz],
               ...
             }, ...
          ],
          ...
        }

        return: (batch=1, seq_len, hidden_dim) Tensor
        """
        objects = scene_graph.get("objects", [])

        token_embs = []
        for obj in objects:
            cat_id = self._category_to_id(obj["category"])
            cat_emb = self.category_embedding(torch.tensor(cat_id).long().unsqueeze(0))  # shape (1, hidden_dim)

            pos_vec = torch.tensor(obj["position"], dtype=torch.float).unsqueeze(0)  # (1,3)
            pos_emb = self.pos_encoder(pos_vec)  # (1, hidden_dim)

            rot_vec = torch.tensor(obj["rotation"], dtype=torch.float).unsqueeze(0)  # (1,3)
            rot_emb = self.rot_encoder(rot_vec)  # (1, hidden_dim)

            concat_emb = torch.cat([cat_emb, pos_emb, rot_emb], dim=-1)  # (1, 3*hidden_dim)
            final_emb = self.proj(concat_emb)  # (1, hidden_dim)

            token_embs.append(final_emb)

        if len(token_embs) == 0:
            # empty scene
            return torch.zeros(1, 1, self.hidden_dim)

        scene_tokens = torch.cat(token_embs, dim=0).unsqueeze(0)  # (1, seq_len, hidden_dim)
        return scene_tokens

    def _category_to_id(self, category_str):
        """
        실제로는 category별 사전이 필요.
        여기서는 placeholder로 hash
        """
        return abs(hash(category_str)) % 1000
