"""
deepspeed main_nolightning.py \
    --base configs/stable-diffusion/v1-finetune-textcaps-opt.yaml \
    --actual_resume checkpoints/stable-diffusion-v-1-4-original/sd-v1-4.ckpt \
    --max_epochs 1 \
    --eval_every 10 \
    --train_only_adapters \
"""

import argparse, os, datetime, time
from typing import Union

from pprint import pformat

import torch
import torchvision
import numpy as np
import wandb
import deepspeed

import torch.distributed as dist
import torch.utils.data.distributed

from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from pytorch_lightning import seed_everything
from einops import rearrange, repeat

from loguru import logger
from tqdm import tqdm

from ldm.util import instantiate_from_config
from ldm.models.diffusion.latent_diffusion import LatentDiffusion
from ldm.models.diffusion.ddim_sampler import DDIMSampler
from ldm.models.diffusion.plms_sampler import PLMSSampler
from ldm.metrics.clip_score import CLIPScore
from ldm.metrics.fid import compute_fid
from ldm.modules.encoders.modules import FrozenCLIPEmbedder

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

torch.backends.cudnn.benchmark = True


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
    parser.add_argument("--train_only_adapters", action="store_true", default=False, help="Only train FFN between text model and diffusion model")
    parser.add_argument("--lora", action="store_true", default=False, help="Use LORA adapters for U-Net attention modules")
    parser.add_argument("--conditioning_drop", type=float, default=0.0, help="Probabiblity of dropping of the text-conditioning to improve classifier-free guidance sampling")

    # Evaluation arguments
    parser.add_argument("--eval_every", type=int, default=1000, help="evaluate every n steps")
    parser.add_argument("--save_every", type=int, default=1000, help="save every n steps")
    parser.add_argument("--pmls", type=str2bool, default=True, help="Use PMLS diffusion scheduler. If False, use DDIM")
    parser.add_argument("--clip_score_batch_size", type=int, default=32, help="Batch size for CLIP score evaluation")
    parser.add_argument("--eval_diffusion_steps", type=int, default=50, help="Number of diffusion steps used for image generation during evaluation")

    # Misc
    parser.add_argument( "-l", "--logdir", type=str, default="logs", help="logging directory")
    parser.add_argument("-s", "--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("-n", "--name", type=str, const=True, default="", nargs="?", help="postfix for logdir")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--project", type=str, default="stable_diffusion_text", help="wandb project name")
    args = parser.parse_args()
    return args


def param_count(param_list):
    return sum(p.numel() for p in param_list)


def load_model_from_config(config, ckpt, verbose=False):
    logger.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logger.info("missing keys:")
        logger.info(m[:3])
        logger.info(f"and {len(m) - 3} more")
    if len(u) > 0 and verbose:
        logger.info("unexpected keys:")
        logger.info(u[:3])
        logger.info(f"and {len(u) - 3} more")

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


def gather_tensor(x: torch.Tensor):
    gather_list = None
    if dist.get_rank() == 0:
        gather_list = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.gather(x, gather_list=gather_list, dst=0)

    if dist.get_rank() == 0:
        return torch.cat(gather_list, dim=0)


def gather_object(x):
    gather_list = None
    if dist.get_rank() == 0:
        gather_list = [None for _ in range(dist.get_world_size())]
    
    dist.gather_object(x, object_gather_list=gather_list, dst=0)

    if dist.get_rank() == 0:
        if isinstance(gather_list[0], list):
            return [item for sublist in gather_list for item in sublist]
        return gather_list


def all_gather_object(x):
    gather_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(obj=x, object_list=gather_list)

    gather_list = [item for sublist in gather_list for item in sublist]
    return gather_list


@torch.no_grad()
def generate_images_grid(*, sampler, model, prompts, images_per_prompt, diffusion_steps, target_shape, guidance_scale, eta, start_code):
    c = model.get_learned_conditioning(prompts)
    uc = model.get_unconditional_conditioning(len(prompts), seq_len=c.shape[1])

    generated_images = []

    for p_idx in range(len(prompts)):
        # generate one batch of images for each prompt
        # intermediates is a dict with keys 'x_inter', 'pred_x0'
        # x_inter is a list of tensors of shape [batch_size, 4, 64, 64]
        _c = repeat(c[p_idx], "... -> repeat ...", repeat=images_per_prompt)
        _uc = repeat(uc[p_idx], "... -> repeat ...", repeat=images_per_prompt)

        _generated_latent_images, intermediates = sampler.sample(
            num_steps=diffusion_steps,
            conditioning=_c,
            unconditional_conditioning=_uc,
            shape=target_shape,
            unconditional_guidance_scale=guidance_scale,
            eta=eta,
            x_T=start_code,
            batch_size=images_per_prompt,
            verbose=False,
        )

        _generated_images = model.decode_first_stage(_generated_latent_images)
        _generated_images = torchvision.utils.make_grid(_generated_images, nrow=images_per_prompt)  # [3, 512, 512 * batch_size]
        generated_images.append(_generated_images)

    generated_images = torch.stack(generated_images, dim=0)  # [batch_size, 3, 512, 512 * batch_size]
    generated_images = gather_tensor(generated_images)

    if dist.get_rank() == 0:
        generated_images = postprocess_image(generated_images)  # returns numpy array
        return generated_images


@torch.no_grad()
def generate_images(*, sampler, model, prompts, diffusion_steps, target_shape, guidance_scale, eta, start_code, gather=True):
    """
    Returns:
        generated_images: numpy *uint8* tensor of shape [batch_size, 512, 512, 3]
    """
    c = model.get_learned_conditioning(prompts)
    uc = model.get_unconditional_conditioning(len(prompts), seq_len=c.shape[1])

    generated_latent_images, intermediates = sampler.sample(
        num_steps=diffusion_steps,
        conditioning=c,
        unconditional_conditioning=uc,
        shape=target_shape,
        unconditional_guidance_scale=guidance_scale,
        eta=eta,
        x_T=start_code,
        batch_size=len(prompts),
        verbose=False,
    )

    generated_images = model.decode_first_stage(generated_latent_images)
    if not gather:
        generated_images = postprocess_image(generated_images)
        return generated_images

    generated_images = gather_tensor(generated_images)

    if dist.get_rank() == 0:
        generated_images = postprocess_image(generated_images)  # returns numpy array
        return generated_images


if __name__ == "__main__":
    DEVICE = "cuda"

    args = parse_args()
    if "LOCAL_RANK" in os.environ:
        # support torchrun
        args.local_rank = int(os.environ["LOCAL_RANK"])

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

    if args.lora or config.model.params.unet_config.params.use_lora:
        logger.info("Using LORA")
        config.model.params.unet_config.params.use_lora = True

    if args.actual_resume:
        model = load_model_from_config(config, args.actual_resume)
    else:
        model = instantiate_from_config(config.model)

    if args.train_only_adapters:
        adapter_parameters = [p for n, p in model.named_parameters() if "adapter" in n or "out_normalization" in n]
        blank_conditioning_parameters = [p for n, p in model.named_parameters() if "blank_conditioning" in n]
        lora_parameters = [p for n, p in model.named_parameters() if "lora" in n]

        logger.info(f"\tAdapter parameters: {param_count(adapter_parameters) / 1e6:.2f}M")
        logger.info(f"\tBlank conditioning parameters: {param_count(blank_conditioning_parameters) / 1e6:.2f}M")
        logger.info(f"\tLoRa parameters: {param_count(lora_parameters) / 1e6:.2f}M")
        trainable_parameters = adapter_parameters + blank_conditioning_parameters + lora_parameters
    else:
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    logger.info(f"Number of model parameters    : {param_count(model.parameters()) / 1e6:.2f}M")
    logger.info(f"Number of trainable parameters: {param_count(trainable_parameters) / 1e6:.2f}M")
    optimizer = torch.optim.Adam(trainable_parameters, lr=config.deepspeed.optimizer.params.lr)

    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
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
    clip_score = CLIPScore(device=DEVICE, dtype=dtype, distributed=False)  # compute them independently on different GPUs, then gather and average

    if global_rank == 0:
        config_dict = OmegaConf.to_container(config, resolve=True)
        wandb.init(
            project=args.project,
            config={**config_dict, **vars(args)},
        )
        # save config files
        for config_path in args.base:
            wandb.save(config_path, policy="now")

    train_dataset = instantiate_from_config(config.data.params.train)
    val_dataset = instantiate_from_config(config.data.params.val)

    batch_size = int(model.config["train_micro_batch_size_per_gpu"])
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

    val_diffusion_steps = args.eval_diffusion_steps
    if val_diffusion_steps < 50:
        logger.warning(f"Using a small number of diffusion steps for evaluation: {val_diffusion_steps}")

    val_start_code = torch.randn([batch_size, 4, 512 // 8, 512 // 8], dtype=dtype, device=DEVICE)
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
    save_every = args.save_every
    global_step = -1
    update_step = 0
    examples_seen = 0
    logged_real_images = False
    start_time = time.time()

    for epoch in range(args.max_epochs):
        logger.info(f"Starting epoch {epoch + 1} / {args.max_epochs}")
        for batch in tqdm(train_loader):
            global_step += 1

            x, c = model.get_input(batch, model.first_stage_key)  # no grad
            # x is [batch_size, 3, 512, 512]
            # c is list of srtings

            if args.conditioning_drop > 0.0:
                if not isinstance(model.cond_stage_model, FrozenCLIPEmbedder):
                    raise NotImplementedError("Conditioning drop is only implemented for FrozenCLIPEmbedder")
                # this trick will work with clip conditioning,
                # because model is already aware of its representations
                # and because it uses "" for blank conditioning
                for i in range(len(c)):
                    if torch.rand(1).item() < args.conditioning_drop:
                        c[i] = ""

            pad_to_max_len = False
            if all([len(cc) == 0 for cc in c]):
                pad_to_max_len = True

            loss, loss_dict = model(x, c, pad_to_max_len=pad_to_max_len)

            model.backward(loss)
            # model.cond_stage_model.blank_conditioning.grad
            # import remote_pdb; remote_pdb.set_trace()

            model.step()

            examples_seen += batch_size * dist.get_world_size()

            if model.is_gradient_accumulation_boundary():
                # it's important that this one is separate from the next if
                update_step += 1

            if wandb.run is not None and model.is_gradient_accumulation_boundary():
                examples_per_second = examples_seen / (time.time() - start_time)
                wandb.log({
                    "loss": loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "examples_seen": examples_seen,
                    "examples_per_second": examples_per_second,
                    **loss_dict,
                    },
                    step=global_step,
                )

            # --- Validation starts
            _validation_time = time.time()
            if global_step % eval_every == 0:
                logger.info(f"Starting validation at step {global_step}")
                all_generated_images = [] if global_rank == 0 else None
                all_real_images = [] if global_rank == 0 else None
                all_prompts = [] if global_rank == 0 else None

                for i, val_batch in enumerate(val_loader):
                    if i > 0: break  # only 1 batch

                    prompts = val_batch[model.cond_stage_key]
                    prompts_gathered = gather_object(prompts)
                    if global_rank == 0:
                        all_prompts.extend(prompts)

                    # TODO: move logging real imges to before the loop
                    if not logged_real_images:
                        images = val_batch[model.first_stage_key]
                        images = rearrange(images, "b h w c -> b c h w").to(DEVICE)

                        images_gathered = gather_tensor(images)
                        if global_rank == 0:
                            images_gathered = postprocess_image(images_gathered)
                            all_real_images.extend(images_gathered)

                    with model.ema_scope():
                        # returns [batch_size, 512, 512, 3] numpy array
                        generated_images = generate_images_grid(
                            sampler=val_sampler,
                            model=model,
                            prompts=prompts,
                            images_per_prompt=batch_size,
                            diffusion_steps=val_diffusion_steps,
                            target_shape=val_target_shape,
                            guidance_scale=val_guidance_scale,
                            eta=val_eta,
                            start_code=val_start_code,
                        )

                        if global_rank == 0:
                            assert generated_images.shape[0] == batch_size * dist.get_world_size(), generated_images.shape
                            all_generated_images.append(generated_images)
                        else:
                            assert generated_images is None

                # end of inference
                if global_rank == 0:
                    images_to_log = batch_size * dist.get_world_size()
                    all_generated_images = np.concatenate(all_generated_images, axis=0)
                    log_generated_images = all_generated_images[:images_to_log]
                    log_prompts = all_prompts[:images_to_log]

                    images_log = [wandb.Image(im, caption=p) for im, p in zip(log_generated_images, log_prompts)]
                    wandb.log({"val/generated_images_row": images_log}, step=global_step)

                    if not logged_real_images:
                        log_real_images = all_real_images[:images_to_log]
                        images_log = [wandb.Image(im, caption=p) for im, p in zip(log_real_images, log_prompts)]
                        wandb.log({"val/real_images": images_log}, step=global_step)

                logged_real_images = True  # notice that it is outsize of the if global_rank == 0

                # Now he same, but with a single generation per prompt and CLIP/FID scores
                all_generated_images = []
                all_real_images = []
                all_prompts = []

                n_val_batches = min(50, len(val_loader))
                _desc = f"Validation at step {global_step}, generating images for CLIP and FID scores"
                for i, val_batch in enumerate(tqdm(val_loader, total=n_val_batches, desc=_desc, disable=global_rank != 0)):
                    if i >= n_val_batches: break
                    prompts = val_batch[model.cond_stage_key]
                    all_prompts.extend(prompts)

                    images = val_batch[model.first_stage_key]
                    images = rearrange(images, "b h w c -> b c h w").to(DEVICE)
                    images = postprocess_image(images)
                    all_real_images.extend(images)

                    with model.ema_scope():
                        # returns [batch_size, 512, 512, 3] numpy array
                        if i < len(val_loader):  # last batch might be truncated
                            assert len(prompts) == batch_size, (len(prompts), batch_size)

                        generated_images = generate_images(
                            sampler=val_sampler,
                            model=model,
                            prompts=prompts,
                            diffusion_steps=val_diffusion_steps,
                            target_shape=val_target_shape,
                            guidance_scale=val_guidance_scale,
                            eta=val_eta,
                            start_code=val_start_code,
                            gather=False,
                        )
                        assert generated_images.shape == (batch_size, 512, 512, 3), generated_images.shape
                        all_generated_images.append(generated_images)

                assert len(all_generated_images) == n_val_batches
                assert all_generated_images[0].shape == (batch_size, 512, 512, 3)
                all_generated_images = np.concatenate(all_generated_images, axis=0)

                logger.info("Computing CLIP scores")
                clip_score_value = clip_score.compute(captions=all_prompts, images=all_generated_images, batch_size=args.clip_score_batch_size)

                logger.info("Computing FID scores")
                assert len(all_real_images) == len(all_generated_images)
                assert len(all_real_images) == batch_size * n_val_batches
                assert all_real_images[0].shape == (512, 512, 3), all_real_images[0].shape

                all_real_images = np.transpose(all_real_images, (0, 3, 1, 2))  # [batch_size, channels, height, width]
                all_generated_images = np.transpose(all_generated_images, (0, 3, 1, 2))  # [batch_size, channels, height, width]
                fid_score_value = compute_fid(all_real_images, all_generated_images, batch_size=args.clip_score_batch_size)  # doesn't support distirubted

                clip_score_value = gather_object(clip_score_value)
                fid_score_value = gather_object(fid_score_value)

                if global_rank == 0:
                    assert len(clip_score_value) == dist.get_world_size()
                    assert isinstance(clip_score_value[0], float), clip_score_value[0]
                    clip_score_value = np.mean(clip_score_value)

                    assert len(fid_score_value) == dist.get_world_size()
                    assert isinstance(fid_score_value[0], float), fid_score_value[0]
                    fid_score_value = np.mean(fid_score_value)

                    wandb.log({
                        "val/clip_score": clip_score_value,
                        "val/fid": fid_score_value,
                        "val/harmonic_mean": 2 * (clip_score_value * fid_score_value) / (clip_score_value + fid_score_value),
                        },
                        step=global_step,
                    )

                logger.info(f"Validation at step {global_step} took {int(time.time() - _validation_time)} seconds")
            # --- Validation ends

            if global_step > 0 and global_step % save_every == 0:
                # save model using deepspeed
                logger.info(f"Saving model at step {global_step}")
                meta = {
                    "global_step": global_step,
                    "state_dict": model.module.state_dict(),
                    "config": OmegaConf.to_container(config, resolve=True),
                }

                model.save_checkpoint(ckptdir, tag=str(global_step), client_state=meta)
                logger.info(f"Saved model to {ckptdir} at step {global_step}")

    logger.info(f"Training finished successfully")
