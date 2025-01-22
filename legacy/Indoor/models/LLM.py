from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig

from models.layers.Embed import PatchEmbedding2D

class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.target_window = target_window

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window * 34)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], self.target_window, -1)
        return x

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.pred_len = config["pred_len"]
        self.image_size = config["image_size"]
        self.d_llm = config["d_llm"]
        self.d_ff = config["d_ff"]
        self.patch_size = config["patch_size"]
        self.stride = config["stride"]

        if config["llm_model"] == 'LLAMA-3.2-3B':
            self.llama_config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-3B')
            self.llama_config.num_hidden_layers = config["llm_layers"]
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True

            self.llm_model = AutoModel.from_pretrained(
                'meta-llama/Llama-3.2-3B',
                config=self.llama_config,
                # load_in_4bit=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                'meta-llama/Llama-3.2-3B'
            )
        elif config["llm_model"] == 'LLAMA-3.2-1B':
            self.llama_config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B')
            self.llama_config.num_hidden_layers = config["llm_layers"]
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True

            self.llm_model = AutoModel.from_pretrained(
                'meta-llama/Llama-3.2-1B',
                config=self.llama_config,
                # load_in_4bit=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                'meta-llama/Llama-3.2-1B'
            )
        else:
            raise Exception('LLM vp_model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(config["dropout"])

        self.patch_embedding = PatchEmbedding2D(
            config["d_model"], self.patch_size, self.stride, config["dropout"])

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(config["d_model"], config["n_heads"], self.d_ff, self.d_llm)

        self.patch_nums = int((config["image_size"] - self.patch_size) / self.stride + 2) ** 2
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(self.head_nf, self.pred_len,
                                             head_dropout=config["dropout"])

    def forward(self, floor_plan):
        B, W, H, N = floor_plan.size()
        floor_plan = floor_plan.permute(0, 3, 1, 2).contiguous().reshape(B * N, W, H, 1)

        prompt = []
        for b in range(floor_plan.shape[0]):
            prompt_ = (
                f"<|start_prompt|>3D Front Indoor is a crucial resource for exploring the design and usage of indoor spaces. "
                f"This dataset contains indoor layouts from two distinct urban areas in China. It includes three types of room categories—living room, dining room, and bedroom. "
                f"Each data point provides an empty room image and layout elements, described by {{object category, 3D location (x, y, z), axis-aligned size (h, w, d), orientation (radians)}}. "
                f"The train/validation/test split is 7:2:1.\n\n"
                f"Task description:\n"
                f"Given an empty dining room image, the goal is to generate up to 25 furniture elements. "
                f"Each element should be specified as {{object category, 3D location (x, y, z), axis-aligned size (w, h, d), theta (radians)}}. "
                f"The object category must be one of the following: [\"armchair\", \"bookshelf\", \"cabinet\", \"ceiling_lamp\", \"chaise_longue_sofa\", \"chinese_chair\", "
                f"\"coffee_table\", \"console_table\", \"corner_side_table\", \"desk\", \"dining_chair\", \"dining_table\", \"l_shaped_sofa\", \"lazy_sofa\", "
                f"\"lounge_chair\", \"loveseat_sofa\", \"multi_seat_sofa\", \"pendant_lamp\", \"round_end_table\", \"shelf\", \"stool\", \"tv_stand\", \"wardrobe\", \"wine_cabinet\"].\n\n"
                # f"Statistics\n"
                # f"In this dataset, the frequency of each object category ranges from a minimum of 0.00064 to a maximum of 0.34746, with a median of 0.02061. "
                # f"Examining the distribution of 3D locations for each object category reveals that, for the x-coordinate, the smallest mean belongs to chinese_chair (mean = -0.72692, std = 2.25003), "
                # f"while chaise_longue_sofa holds the largest mean (mean = 0.64562, std = 1.2007). armchair takes the median position, with a mean of -0.02204 and a standard deviation of 1.73172.\n\n"
                # f"Turning to the y-coordinate, coffee_table has the lowest mean (0.22034, std = 0.09243), whereas ceiling_lamp has the highest (2.5118, std = 0.20129). "
                # f"Situated at the median is multi_seat_sofa (0.42958, std = 0.04607). Meanwhile, the z-coordinate shows chaise_longue_sofa at the lowest mean (-0.61940, std = 2.26890), "
                # f"lazy_sofa at the highest (0.99755, std = 2.17601), and wine_cabinet in the median range (-0.01169, std = 2.05231).\n\n"
                # f"When looking at axis-aligned sizes, the smallest average width (w) appears in round_end_table (mean = 0.19482, std = 0.05178), "
                # f"whereas l_shaped_sofa has the largest (mean = 1.61473, std = 0.27087). cabinet sits at the median (mean = 0.49284, std = 0.17989). "
                # f"For height (h), ceiling_lamp is the shortest (mean = 0.15274, std = 0.12888), wine_cabinet is the tallest (mean = 1.11078, std = 0.22275), "
                # f"and dining_chair lands in the middle (mean = 0.42613, std = 0.07340). As for depth (d), shelf records the smallest mean (0.16912, std = 0.03879), "
                # f"whereas l_shaped_sofa has the highest (mean = 0.90699, std = 0.16590). Situated at the median is ceiling_lamp (mean = 0.29098, std = 0.09426).\n\n"
                # f"Lastly, analyzing the rotation angles in radians reveals that chaise_longue_sofa has the lowest mean (-1.12199, std = 1.38329), "
                # f"chinese_chair holds the highest (0.21154, std = 1.78975), and stool occupies the median position (-0.24491, std = 1.42962). "
                f"These figures together illustrate how the dataset captures a wide variety of configurations for different categories, providing comprehensive statistics on their 3D positions, dimensions, and orientations.<|end_prompt|>"
            )
            prompt.append(prompt_)

        floor_plan = floor_plan.permute(0, 3, 1, 2).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(floor_plan.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        enc_out, n_vars = self.patch_embedding(floor_plan.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])

        return dec_out

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding