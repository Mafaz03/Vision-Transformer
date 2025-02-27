import torch
from torch import nn

class SigLipVisionConfig:
    def __init__(
            self,
            hidden_size = 768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels: int = 3,
            patch_size= 16,
            layer_norm_eps=1e-6,
            attention_driopout=0.0, # not using it right now
            num_image_tokens: int = None,
            *kwargs
            ):
            super().__init__()      
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_channels = num_channels
            self.patch_siz, = patch_size
            self.layer_norm_eps = layer_norm_eps
            self.attention_driopout = attention_driopout
            self.num_image_tokens = num_image_tokens

class SiglipVisionTransformer(nn.Module):
    super().__init__()    
    def __init__(self, config: SigLipVisionConfig):
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoderf(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)