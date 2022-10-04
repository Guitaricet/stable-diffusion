"""
deepspeed main_nolightning.py \
    --base configs/stable-diffusion/v1-finetune-textcaps.yaml \
    --actual_resume checkpoints/stable-diffusion-v-1-4-original/sd-v1-4.ckpt \
    --max_epochs 1 \
    --eval_every 10
"""

import argparse, os, datetime
from operator import pos
from typing import Union

from pprint import pformat

import torch
import torchvision
import numpy as np
import wandb
import deepspeed

import torch.distributed as dist
import torch.utils.data.distributed

from torch.utils.data import random_split, DataLoader, Dataset, Subset
from omegaconf import OmegaConf
from PIL import Image

from pytorch_lightning import seed_everything
from einops import rearrange

from loguru import logger
from tqdm import tqdm

import ldm
import ldm.data.textcaps
from ldm.util import instantiate_from_config
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()


def parse_args(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml", required=True,
    help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
    )

    # Training arguments
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--actual_resume", type=str, default="", help="Path to model to actually resume from")

    # Evaluation arguments
    parser.add_argument("--eval_every", type=int, default=1000, help="evaluate every n steps")
    parser.add_argument("--pmls", type=str2bool, default=True, help="Use PMLS diffusion scheduler. If False, use DDIM")

    # Misc
    parser.add_argument( "-l", "--logdir", type=str, default="logs", help="logging directory")
    parser.add_argument("-s", "--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("-n", "--name", type=str, const=True, default="", nargs="?", help="postfix for logdir")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--project", type=str, default="stable_diffusion_text", help="wandb project name")
    args = parser.parse_args()
    return args


def load_model_from_config(config, ckpt, verbose=False):
    logger.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logger.info("missing keys:")
        logger.info(m)
    if len(u) > 0 and verbose:
        logger.info("unexpected keys:")
        logger.info(u)

    return model


def postprocess_image(image):
    image = image.detach().cpu()
    if image.dtype is torch.bfloat16:
        image = image.float()  # bfloat doesn't convert to numy
    image = image.numpy()
    image = rearrange(image, "... c h w -> ... h w c")
    image = (image + 1.0) / 2.0
    image = (255.0 * image).clip(0, 255).astype(np.uint8)
    return image


if __name__ == "__main__":
    args = parse_args()
    deepspeed.init_distributed()
    global_rank = dist.get_rank()

    if global_rank != 0:
        logger.remove()

    logger.info(f"Starting run with args: {pformat(vars(args))}")

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = now + args.name
    logdir = os.path.join(args.logdir, nowname)

    logger.info(f"Logging to {logdir}")
    ckptdir = os.path.join(logdir, "checkpoints")
    os.makedirs(ckptdir, exist_ok=True)

    seed_everything(args.seed)

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    config = OmegaConf.merge(*configs)

    if "lightning" in config:
        config.pop("lightning")

    # OmegaConf to dict
    deepspeed_config = OmegaConf.to_container(config.deepspeed, resolve=True)

    if args.actual_resume:
        model = load_model_from_config(config, args.actual_resume)
    else:
        model = instantiate_from_config(config.model)

    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=deepspeed_config,
    )
    model: Union[LatentDiffusion, deepspeed.DeepSpeedEngine]  # helps with type hinting

    dtype = torch.float32
    if model.bfloat16_enabled():
        dtype = torch.bfloat16
    if model.fp16_enabled():
        dtype = torch.float16

    # convert model inputs to dtype
    model.set_data_dtype(dtype)

    if global_rank == 0:
        config_dict = OmegaConf.to_container(config, resolve=True)
        wandb.init(
            project=args.project,
            config={**config_dict, **vars(args)},
        )
        # save config files
        for config_path in args.base:
            wandb.save(config_path, policy="now")

    train_dataset = ldm.data.textcaps.TextCapsBase(
        data_root=config.data.params.train.params.data_root,
        size=config.data.params.train.params.size,
        set="train",
    )
    val_dataset = ldm.data.textcaps.TextCapsBase(
        data_root=config.data.params.val.params.data_root,
        size=config.data.params.val.params.size,
        set="val",
    )

    batch_size = model.config["train_micro_batch_size_per_gpu"]
    train_loader = DataLoader(
        train_dataset,
        num_workers=config.data.params.num_workers,
        batch_size=batch_size,
        pin_memory=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=config.data.params.num_workers,
        batch_size=batch_size,
        pin_memory=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            shuffle=False,
        ),
    )

    val_diffusion_steps = 100
    val_start_code = torch.randn([batch_size, 4, 512 // 8, 512 // 8], dtype=dtype, device="cuda")
    val_sampler = PLMSSampler(model.module) if args.pmls else DDIMSampler(model.module)
    val_guidance_scale = 7.5
    val_eta = 0.0  # has to be zero for PMLS

    if global_rank == 0:
        wandb.config["generation_diffusion_steps"] = val_diffusion_steps
        wandb.config["generation_guidance_scale"] = val_guidance_scale

    # NOTE: we describe the shape in latent space, not in pixel space [4, 64, 64] is [3, 512, 512] image
    val_target_shape = [model.channels, model.image_size, model.image_size]

    # TODO: if training from scratch, add scaling like in LatentDiffusion.on_train_batch_start

    eval_every = args.eval_every
    save_every = 2000
    global_step = -1
    update_step = 0
    for epoch in range(args.max_epochs):
        for batch in tqdm(train_loader):
            global_step += 1

            loss, loss_dict = model.shared_step(batch)

            model.backward(loss)
            model.step()

            if wandb.run is not None and model.is_gradient_accumulation_boundary():
                update_step += 1
                wandb.log({
                    "loss": loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    **loss_dict,
                    },
                    step=global_step,
                )
            
            # --- Validation starts
            if global_step % eval_every == 0:
                val_batch = next(iter(val_loader))  # predict just one batch
                prompts = val_batch[model.cond_stage_key]
                all_generated_images = []
                all_prompts = []

                with torch.no_grad():
                    with model.ema_scope():
                        uc = model.get_learned_conditioning(batch_size * [""])
                        c = model.get_learned_conditioning(prompts)
                        assert uc.dtype is dtype
                        assert c.dtype is dtype

                        generated_latent_images, intermediates = val_sampler.sample(
                            num_steps=val_diffusion_steps,
                            conditioning=c,
                            unconditional_conditioning=uc,
                            shape=val_target_shape,
                            unconditional_guidance_scale=val_guidance_scale,
                            eta=val_eta,
                            x_T=val_start_code,
                            batch_size=batch_size,
                            verbose=False,
                        )

                        # intermediates is a dict with keys 'x_inter', 'pred_x0'
                        # x_inter is a list of tensors of shape [batch_size, 4, 64, 64]

                        generated_images = model.decode_first_stage(generated_latent_images)
                        generated_images_gathered = None
                        if global_rank == 0:
                            generated_images_gathered = [torch.zeros_like(generated_images) for _ in range(dist.get_world_size())]
                        dist.gather(generated_images, gather_list=generated_images_gathered, dst=0)

                        # gather prompts
                        prompts_gathered = None
                        if global_rank == 0:
                            prompts_gathered = [None] * dist.get_world_size()
                        dist.gather_object(prompts, object_gather_list=prompts_gathered, dst=0)

                        if global_rank == 0:
                            prompts_gathered = [item for sublist in prompts_gathered for item in sublist]
    
                            generated_images = torch.cat(generated_images_gathered, dim=0)
                            generated_images = postprocess_image(generated_images)  # return numpy array
                            all_generated_images.append(generated_images)
                            all_prompts.extend(prompts_gathered)
                
                # end of inference
                if global_rank == 0:
                    all_generated_images = np.concatenate(all_generated_images, axis=0)
                    wandb.log({
                        "val/generated_images_images": [wandb.Image(im, caption=p) for im, p in zip(generated_images, all_prompts)],
                        },
                        step=global_step,
                    )

            # --- Validation ends

            if global_step > 0 and global_step % save_every == 0:
                # save model using deepspeed
                logger.info(f"Saving model at step {global_step}")
                meta = {
                    "global_step": global_step,
                    "state_dict": model.module.state_dict(),
                    "config": OmegaConf.to_container(config, resolve=True),
                }

                model.save_checkpoint(ckptdir, client_state=meta)
                logger.info(f"Saved model at step {global_step}")
