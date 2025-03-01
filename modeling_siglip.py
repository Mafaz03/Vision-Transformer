import torch
from torch import nn
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Iterable

class SiglipVisionConfig:
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
            self.patch_size = patch_size
            self.layer_norm_eps = layer_norm_eps
            self.attention_dripout = attention_dropout
            self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.embed_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            padding = "valid", # No padding
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.positional_embedding = nn.Embedding(self.num_positions, self.embed_dim)
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
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear (config.hidden_size, config. intermediate_size)
        self.fc2 = nn.Linear (config. intermediate_size, config.hidden_size)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states) 
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 # 1/root(head_dim)
        self.dropout = config.attention_dripout
        
        self.k_proj = nn.Linear(in_features = self.embed_dim, out_features = self.embed_dim)
        self.v_proj = nn.Linear(in_features = self.embed_dim, out_features = self.embed_dim)
        self.q_proj = nn.Linear(in_features = self.embed_dim, out_features = self.embed_dim)

        self.out_proj = nn.Linear(in_features = self.embed_dim, out_features = self.embed_dim)


    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch. Tensor, torch. Tensor]:
        # [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, embed_dim = hidden_states.size() # seq_len is same as Num_Patches

        query_states = self.q_proj(hidden_states) # [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)   # [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states) # [Batch_Size, Num_Patches, Embed_Dim]

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)     # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]

        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] * [Batch_Size, Num_Heads, Head_Dim, Num_Patches] -> [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)} but is {attn_weights.size()}")

        # [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states) # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)} but is {attn_weights.size()}")
        
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.embed_dim) # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output) # [Batch_Size, Num_Patches, Embed_Dim]

        return attn_output, attn_weights



class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
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

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config)  for _ in range(config.num_hidden_layers)]
        )
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states

class SiglipVisionTransformer(nn.Module):    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.FloatTensor):
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state

class SiglipVisionModel(nn.Module):    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(config)
        self.config = config
    
    def forward(self, pixel_values: np.ndarray) -> Tuple: 
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values = pixel_values)
    
if __name__ == "__main__":
    config = SiglipVisionConfig()
    config.num_image_tokens = 1024
    model = SiglipVisionModel(config)
    output = model(torch.randn(1, 3, 224, 224))
    print(output.shape)
    parameters = sum(p.numel() for p in model.parameters())

    print("Total parameters: ", "{:,}".format(parameters))
    
         