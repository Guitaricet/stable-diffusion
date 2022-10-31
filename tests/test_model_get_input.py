import pytest

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.latent_diffusion import LatentDiffusion


def test_model_get_input():
    config = OmegaConf.load("tests/configs/stable-diffusion-lora.yaml")
    model: LatentDiffusion = instantiate_from_config(config.model)

    batch_size = 3
    assert batch_size == 3
    conditioning = [
        "text for the first sample",
        "text for the second sample with different len",
        "for the third sample",
    ]
    x = torch.randn(batch_size, 512, 512, 3)

    batch = {"image": x, "caption": conditioning}
    out = model.get_input(batch, "image")
