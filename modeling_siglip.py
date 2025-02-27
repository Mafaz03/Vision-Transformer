import torch
from torch import nn
import numpy as np
class SigLipVisionConfig:
    def __init__(
            self,
            hidden_size = 768,
            image_size = 224,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels: int = 3,
            patch_size= 16,
            layer_norm_eps=1e-6,
            attention_dropout=0.0, # not using it right now
            num_image_tokens: int = None,
            *kwargs
            ):

            super().__init__()      
            self.hidden_size = hidden_size
            self.image_size = image_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_channels = num_channels
            self.patch_siz, = patch_size
            self.layer_norm_eps = layer_norm_eps
            self.attention_driopout = attention_dropout
            self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Cov2d(
            in_channels = config.num_channels,
            out_channels = self.embed_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            padding = "valid", # No padding
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.positional_embedding = nn.Embedding(self.num_position, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)), 
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.tensor:
        _, _, height, width = pixel_values.shape
        
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Embed_Dim, Num_Patches_Height, Num_Patches_Width]
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2) # [Batch_Size, Num_Patches, Embed_Dim], where Num_Patches = Num_Patches_Height * Num_Patches_Width
        return embeddings + self.positional_embedding(self.position_ids) # [Batch_Size, Num_Patches, Embed_Dim]

class SiglipMLP(nn.Module):
    def __init__(self, config) :
        super()._init_()
        self.config = config
        self.fc1 = nn. Linear (config.hidden_size, config. intermediate_size)
        self.fc2 = nn. Linear (config. intermediate_size, config.hidden_size)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states) 
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SiglipEncoderf(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states #........................................residual: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states) #...........................[Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states) #............[Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states) #...........................[Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states) #...................................[Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = hidden_states + residual
        return hidden_states

class SiglipVisionTransformer(nn.Module):
    super().__init__()    
    def __init__(self, config: SigLipVisionConfig):
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoderf(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.FloatTensor):
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(input_embeds = hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state

class SigLipVisionModel(nn.Module):
    super().__init__()    
    def __init__(self, config: SigLipVisionConfig):
        self.vision_model = SiglipVisionTransformer(config)
        self.config = config
    
    def forward(self, pixel_values: np.ndarray) -> tuple: 
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values = pixel_values)
         