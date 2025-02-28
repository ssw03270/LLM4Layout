# adapter/vision_encoder.py

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T


class VisionEncoder(nn.Module):
    """
    Multi-view images -> (batch=1, image_token_len, hidden_dim) Tensor
    """

    def __init__(self, hidden_dim=768):
        super().__init__()
        # 예시: torchvision ResNet + projection to hidden_dim
        self.backbone = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        self.backbone.fc = nn.Identity()  # remove final classification layer
        self.proj = nn.Linear(2048, hidden_dim)

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def forward(self, images):
        """
        images: list of PIL.Image
        return: (1, token_len, hidden_dim)
                token_len = len(images) (or you can break down each image into patches)
        """
        emb_list = []
        for img in images:
            img_tensor = self.transform(img).unsqueeze(0)  # (1,3,224,224)
            feat = self.backbone(img_tensor)  # (1, 2048)
            feat_proj = self.proj(feat)  # (1, hidden_dim)
            emb_list.append(feat_proj)

        if not emb_list:
            print("NO IMAGE ON VISION ENCODER")
            # no images
            # return torch.zeros(1, 1, hidden_dim)

        vision_tokens = torch.cat(emb_list, dim=0).unsqueeze(0)  # (1, num_views, hidden_dim)
        return vision_tokens
