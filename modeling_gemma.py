import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math 
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class GemmaConfig():
    def __init__(
            self,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim=256,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_bias=False, attention_dropout=0.0,
            pad_token_id=None,
            **kwargs,):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.pad_token_id = pad_token_id        


class PaliGemmaConfig:
    def __init__(
            self,
            vision_config = None,
            text_config = None,
            ignore_index = -100, # wont use            
            image_token_index = 256 * 1000,
            vocab_size = 257152,
            projection_dim = 2048, # Output after brining image tokens in 'same range'
            hidden_size = 2048,
            pad_token_id = None,
            **kwargs,
    ):
        super().__init__()
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_token = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # 1 / sqrt(...)
    def forward (self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to (float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: KVCache = None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # [Batch_Size, Seq_Len, Hidden_Size]
        residual = hidden_states
        
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            kv_cache=kv_cache
        )
        
        # [Batch_Size, Seq_Len, Hidden_Size]
        residual = hidden_states
        hidden_states = residual + hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

class GemmaAttention(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0            

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # Equivalent to:
        # y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # y = torch.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMS(config.hidden_size, eps=config.rms_norm_eps)
    
    def get_input_embeddings(self): return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: KVCache = None
    ) -> Tuple:
        # Batch Size, Seq_Len, Hidden Size
        hidden_states = inputs_embeds
        # [Batch_Size, Seq_Len, Hidden_Size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                kv_cache=kv_cache,
            )
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: KVCache = None
    ) -> Tuple:

        # Input_embeds = [Batch_Size, Seq_Len, Hidden_Size]
        # Outputs = [Batch_Size, Seq_Len, Hideen_Size]
        outputs = self.model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            input_embeds =inputs_embeds,
            kv_cache = kv_cache,
        ) # This gives emebedding, we need logits, so we can do softmax stuff

        hidden_states = outputs
        logits = self.lm_head(hidden_states) # [Batch_Size, Seq_Len, Vocab_Size]
        logits = logits.float()

        return_data = {
            "logits": logits
        }

        if kv_cache is not None:
            # Return updated cache
            return_data["kv_cache"] = kv_cache
        return return_data


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)
    def forward(self, image_features):
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        return self.linear(image_features)

class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_model_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config. text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        

    def tie_weights(self):
        self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(self,
                                         image_features: torch.Tensor,  # comes from modelling_siglip.py, the image embeddings only
                                         inputs_embeds: torch.Tensor,   # language model embedding, including <image><image><image>.....
                                         input_ids: torch.Tensor, 
                                         attention_mask: torch.Tensor, 
                                         kv_cache: torch.Tensor):
        
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=dtype, device=device)

        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id) # [Batch_Size, Seq_Len]. True for text tokens
        image_mask = input_ids == self.config.image_token_index                                     # [Batch_Size, Seq_Len]. True for image tokens
        pad_mask = input_ids == self.pad_token_id                                                   # [Batch_Size, Seq_Len]. True for padding tokens

        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        final_embedding = torch.where(text_mask_expanded, input_ids, final_embedding)
        final_embedding = torch.masked_scatter(image_mask_expanded, scaled_image_features)
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)
        

        ### KV cache implementation ###
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0: # Prefilling 
            casual_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device # Not actually masking out anything
            )
        else:
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len

            casual_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device,
            )
        
        # Adding head dimention
        # [Batch_size, Q_len, KV_len] -> [Batch_size, Num_Head_Q, Q_len, KV_len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None or kv_cache.num_items() > 0: 
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim == 0:
                position_ids = position_ids.unsqueeze(0)
            else:
                # Create a position ids based on the size of the attention_mask
                # For masked tokens, use the number 1 as
                position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask==0), 1).to(device)

        return final_embedding, causal_mask, position_ids

    
    def forward(self, 
                input_ids:torch. LongTensor = None, 
                pixel_vales: torch.FloatTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None):
        
        assert torch.all(attention_mask == 1), "Input cannot be padded, this code is not optimised :("

        # 1. Extract the input Embeddings 
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids) # (Batch_Size, Seq_Len, Hidden_Size)

        # 2. Merge text and image
        selected_image_feature = self.vision_tower(pixel_vales.to(inputs_embeds.dtype))
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.multi_model_projector(selected_image_feature) # Bringing it into same range
        inputs_embeds, attention_mask, position_ids = _merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds, 
            kv_cache = kv_cache
        )