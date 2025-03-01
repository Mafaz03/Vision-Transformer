import torch
from torch import nn
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Iterable
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def resize(image: Image.Image, size: Tuple[int, int], resample: Image.Resampling = None) -> Image.Image:
    return image.resize(size, resample = resample)


def rescale(image: np.ndarray, rescale_factor: float) -> np.ndarray:
    return (image * rescale_factor).astype(np.float32)

def normalize(image: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    return (image - np.array(mean, dtype=image.dtype)) / np.array(std, dtype=image.dtype)

def process_image(images: List[Image.Image], size: Tuple[int, int], resameple: Image.Resampling = None, rescale_factor: float = None, image_mean: List[float] = None, image_std: List[float] = None) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [np.ndarray(resize(image, size = (height, width))) for image in images] # Resize the image and convert to ndarray
    images = [rescale(image) for image in images] # Rescale the image
    images = [normalize(image = image, mean = image_mean, std = image_std) for image in images] # Normalize the image
    images = [image.transpose(2, 0, 1) for image in images] # Transpose the image
    return images

def add_image_tokens_to_prompt(prefix_prompt: str, bos_token: str, image_token: str, image_seq_lenght: int) -> str:
    return f"{image_token * image_seq_lenght}{bos_token}{prefix_prompt}\n"

class PoliGemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens:int, image_size:int):
        super().__init__()

        self.image_seq_len = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]
        # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range (128)
        ]
        # These tokens are used for object segmentation
        tokenizer.add_tokens (EXTRA_TOKENS)
        self. image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer
    
    def __call__(self, text: List[str], images: List[Image.Image], padding:str = "longest", truncation:bool = True) -> Dict:
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts. "
        
        pixel_values = process_image(
            images,
            size = (self.image_size, self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factor = 1/255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD
        )
        # List of images to stack of images with batch size
        pixel_values = torch.tensor(np.stack(pixel_values, axis = 0)) # [Batch_Size, Channel, Height, Width]

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_token = self.IMAGE_TOKEN,
                image_seq_lenght = self.image_seq_len
            )
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            padding = padding,
            truncation = truncation,
            return_tensors = "pt"
        )

        return {"pixel_values": pixel_values, **inputs}

