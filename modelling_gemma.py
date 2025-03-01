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
        self.text_config =text_config
        self.text_config = Gemmaconfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_token = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id

def _merge_input_ids_with_image_features(self,
                                         image_features: torch.Tensor,  # comes from modelling_siglip.py, the image embeddings only
                                         inputs_embeds: torch.Tensor,   # language model embedding, including <image><image><image>.....
                                         input_ids: torch.Tensor, 
                                         attention_mask: torch.Tensor, 
                                         kv_cache: torch.Tensor):
    _, _, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape




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