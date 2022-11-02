"""Downloads LAION dataset shard by shard.
Stops downloading when there are N shards downloaded and waits until one of them is processed.
Then continues downloading.

python scripts/download_laion_sharded.py \
    --input_dir "/home/vlialin/data/text-laion-20M" \
    --output_dir "/home/vlialin/data/text-laion-20M-images" \
    --shard_size 100000 \
    --start_shard 21 \
    --num_shards 50 \

"""

import argparse
import os
import time

from img2dataset import download
from datasets import load_from_disk


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="path to directory with arrow files", required=True)
    parser.add_argument("--output_dir", help="path to directory where to save images", required=True)
    parser.add_argument("--shard_size", type=int, default=100_000)
    parser.add_argument("--start_shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)

    return parser.parse_args()


def main(args):
    # How to prepare data:
    # dataset = load_dataset("laion/laion2B-en-joined", split="train", streaming=True)
    # dataset = dataset.shuffle(seed=84, buffer_size=10_000)
    # dataset = dataset.take(20_000_000)
    # dataset = dataset.filter(lambda x:
    #     x["punsafe"] is not None and \
    #     x["punsafe"] < 0.5 and \
    #     x["pwatermark"] is not None and \
    #     x["pwatermark"] < 0.5
    # )

    script_start_time = time.time()
    dataset = load_from_disk(args.input_dir)

    for shard_idx in range(args.start_shard, args.start_shard + args.num_shards):
        shard_start_time = time.time()

        start_idx = shard_idx * args.shard_size
        end_idx = (shard_idx + 1) * args.shard_size
        print(f"Downloading shard {shard_idx} from {start_idx} to {end_idx}")
        shard = dataset.select(range(start_idx, end_idx))

        shard.to_parquet("tmp.parquet")

        # download images
        shard_idx_str = str(shard_idx).zfill(6)

        download(
            processes_count=16,
            thread_count=32,
            url_list="tmp.parquet",
            input_format="parquet",
            url_col="URL",
            caption_col="TEXT",
            save_additional_columns=['WIDTH', 'HEIGHT', 'similarity', 'hash', 'punsafe', 'pwatermark'],
            image_size=512,
            resize_mode="keep_ratio",
            output_folder=os.path.join(args.output_dir, f"shard_{shard_idx_str}"),
            output_format="files",
            enable_wandb=False,
            number_sample_per_shard=1000,
            distributor="multiprocessing",
            retries=2,
        )

        shard_proc_mins = (time.time() - shard_start_time) / 60
        print(f"Shard {shard_idx} downloaded in {shard_proc_mins:.2f} mins")

    print("Script finished successfully in", (time.time() - script_start_time) / 60, "mins")


if __name__ == "__main__":
    args = make_args()
    main(args)
