import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math 
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

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
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds, 
            kv_cache = kv_cache
        )