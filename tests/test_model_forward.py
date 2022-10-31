import torch
import numpy as np
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.latent_diffusion import LatentDiffusion


def test_model_forward():
    config = OmegaConf.load("tests/configs/stable-diffusion-lora.yaml")
    model: LatentDiffusion = instantiate_from_config(config.model)

    batch_size = 3
    seq_len = 17
    context_dim = config.model.params.unet_config.params.context_dim

    conditioning = torch.randn(batch_size, seq_len, context_dim)
    x = torch.randn(batch_size, 4, 64, 64)

    out = model.forward(x, conditioning)
