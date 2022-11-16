"""
python scripts/cache_text_features.py \
    --input_dir "/home/vlialin/data/text-laion-20M-images-with-text-train" \
    --start_shard 0 \
    --num_shards 1 \
    --model_name "facebook/opt-6.7b" \
    --batch_size 8 \
    --dtype "int8" \
    --max_length 128 \
    --extract_layer -2 \
    --device 0
"""
import os
import json
import time
import argparse
from glob import glob
from pprint import pformat

import torch
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import save_file

from loguru import logger
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="path to directory with images", required=True)

    parser.add_argument("--start_shard", required=True, type=int)
    parser.add_argument("--num_shards", required=True, type=int)

    parser.add_argument("--model_name", required=True)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--dtype", default="int8", choices=["float32", "float16", "bfloat16", "int8"],
        help="dtype to use for the model. Note that int8 will save float16 tensors"
    )
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--extract_layer", default=-1, type=int)
    parser.add_argument("--device", default=0, help='Either a GPU index or "auto" to distribute model across GPUs')

    args = parser.parse_args()

    if args.device != "auto":
        args.device = int(args.device)

    if not os.path.exists(args.input_dir):
        logger.error(f"Directory {args.input_dir} does not exist")
        exit(1)

    return args


@torch.no_grad()
def main(args):
    all_shards = sorted(glob(os.path.join(args.input_dir, "shard_*")))
    shards = all_shards[args.start_shard:args.start_shard + args.num_shards]

    if len(shards) == 0:
        logger.error(f"No shards found in {args.input_dir=}")
        logger.error(f"Reminder that shards are expected to be in {args.input_dir}/shard_000000, {args.input_dir}/shard_000001, etc")
        exit(1)
    
    logger.info(f"Starting with the shard {shards[0]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device_map = args.device
    if isinstance(device_map, int):
        device_map = {"": args.device}
    logger.info(f"Using device map {device_map}")

    if args.dtype == "int8":
        logger.info("Loading model in int8 mode")
        model = AutoModel.from_pretrained(args.model_name, load_in_8bit=True, device_map=device_map)
    else:
        logger.info(f"Loading model in {args.dtype} mode")
        # model = AutoModel.from_pretrained(args.model_name).to(args.device, dtype=getattr(torch, args.dtype))
        model = AutoModel.from_pretraine(args.model_name, device_map=device_map, dtype=getattr(torch, args.dtype))

    # trace the model
    logger.info("Tracing the model")
    model.eval()
    torch.jit.trace(model, torch.zeros(1, 1, dtype=torch.long).to(args.device)).to(args.device)

    if args.extract_layer < 0:
        _extract_layer = model.config.num_hidden_layers + args.extract_layer + 1
    file_suffix = os.path.basename(args.model_name) + "_layer" + str(_extract_layer) + ".safetensors"
    logger.info(f"Using file suffix: {file_suffix}")

    for shard_path in shards:
        _shard_time = time.time()
        logger.info(f"Processing shard {shard_path}")

        batch = []
        save_dir = shard_path
        if not os.path.exists(save_dir):
            logger.error(f"Save directory {save_dir} does not exist")
            exit(1)

        for meta_path in tqdm(glob(os.path.join(shard_path, "*.json")), desc=shard_path):
            id_ = os.path.basename(meta_path).rstrip(".json")
            meta = json.load(open(meta_path))

            caption = meta["caption"]
            batch.append({"id": id_, "caption": caption})

            if len(batch) == args.batch_size:
                captions_batch = [x["caption"] for x in batch]
                inputs = tokenizer(captions_batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to("cuda")
                output = model(**inputs, output_hidden_states=True)
                hidden_states = output.hidden_states[args.extract_layer].cpu()
                for i, item in enumerate(batch):
                    item_id = item["id"]
                    file_path = os.path.join(save_dir, f"{item_id}_{file_suffix}")
                    save_file({"hidden_states": hidden_states[i]}, file_path)
                batch = []

        _shard_time = (time.time() - _shard_time) / 60
        _shard = os.path.basename(shard_path)
        logger.info(f"Finished processing shard {_shard} in {_shard_time:.2f} minutes")


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Starting script with args: {pformat(vars(args))}")
    main(args)
    logger.info("Script finished successfully")
