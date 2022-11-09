import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

from ..encoders.t5_encoder import T5EncoderModel

from loguru import logger


class FrozenHugEmbedderWithAdapter(nn.Module):
    def __init__(self, model_name, max_length, output_dim, feature_layer_index=-1, output_layer_norm=False, **model_kwargs):
        super().__init__()
        # how to load just the encoder of t5?
        self.feature_layer_index = feature_layer_index
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if "load_in_8bit" in model_kwargs:
            gpu_index = os.environ["LOCAL_RANK"]
            logger.info(f"Using 8-bit model. Rank: {gpu_index}")

            device_map = {'': int(gpu_index)}  # put everything on the same device
            if "device_map" in model_kwargs:
                device_map = model_kwargs["device_map"]

            logger.info(f"Device map: {device_map}")
            model_kwargs["device_map"] = device_map

        if "t5-" in model_name:
            logger.info("Using T5 encoder via ldm.modules.t5_encoder.T5EncoderModel (no decoder)")
            self.transformer = T5EncoderModel.from_pretrained(model_name, **model_kwargs)
            logger.info("T5 encoder loaded")
            logger.info(f"Memory usage: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
        else:
            self.transformer = AutoModel.from_pretrained(model_name, **model_kwargs)

        self.freeze()  # freeze the transformer

        if hasattr(self.transformer.config, "d_model"):
            transformer_hidden = self.transformer.config.d_model
        elif hasattr(self.transformer.config, "word_embed_proj_dim"):
            transformer_hidden = self.transformer.config.word_embed_proj_dim
        elif hasattr(self.transformer.config, "hidden_size"):
            transformer_hidden = self.transformer.config.hidden_size
        else:
            raise ValueError("Could not find transformer hidden size")

        self.blank_conditioning = nn.Embedding(max_length, output_dim)
        self.adapter = nn.Sequential(
            nn.Linear(transformer_hidden, 4 * transformer_hidden),
            nn.SiLU(),  # SiLU(x) = Swish(x) = x * sigmoid(x)
            nn.Linear(4 * transformer_hidden, output_dim),
        ) # only the adapter is trainable
        # Add LayerNorm and set gamma and beta to values from CLIP

        if not output_layer_norm:
            raise RuntimeError("You should always use LayerNorm for FrozenHugEmbedderWithAdapter")
            
        self.out_normalization = nn.LayerNorm(output_dim)

        self.model_name = model_name
        self.max_length = max_length
        self.output_dim = output_dim

    @property
    def device(self):
        return self.transformer.device

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def get_blank_conditioning(self, batch_size, seq_len):
        return self.blank_conditioning.weight[:seq_len].repeat(batch_size, 1, 1)

    def forward(self, text, pad_to_length=None):
        kwargs = {"padding": "max_length" if pad_to_length is not None else True}

        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=pad_to_length or self.max_length,
            return_tensors="pt",
            **kwargs,
        )
        batch_encoding = batch_encoding.to(self.device)
        outputs = self.transformer(**batch_encoding, output_hidden_states=True)

        z = outputs.hidden_states[self.feature_layer_index]
        z = self.adapter(z)
        z = self.out_normalization(z)

        return z
