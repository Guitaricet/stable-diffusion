import time
import argparse
from glob import glob

from tqdm import tqdm
import datasets
from datasets.arrow_dataset import Dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    print("Started LAION download and filtering (metadata only)")
    print(f"Saving to {args.output_dir}")

    dataset: Dataset = datasets.load_dataset("laion/laion2B-en-joined", split="train", streaming=True)
    dataset = dataset.shuffle(seed=84, buffer_size=10_000)
    subset_size = 500_000_000  # 0.5B
    # 500_000_000
    # 278_315_372
    dataset = dataset.take(subset_size)

    # filtering the dataset
    # we use small pwatermark threshold, because watermarks are especially bad for the domain of in-scene text images
    # gettyimages is the most annoying watermark, so we double check on the url
    dataset = dataset.filter(lambda x:
        x["punsafe"] is not None and \
        x["punsafe"] < 0.5 and \
        x["pwatermark"] is not None and \
        x["pwatermark"] < 0.3 \
        and "gettyimages" not in x["URL"]
    )

    # filer is a lazy function, so we need to iterate of the dataset to actually filter it
    print("Started filtering. It is expected to take 1-2 hours")
    _time = time.time()
    dataset_list = []
    total_dataset_size = 0
    shard_idx = 0
    for example in tqdm(dataset, total=subset_size):
        total_dataset_size += 1
        dataset_list.append(example)

        if len(dataset_list) % 10_000_000 == 0:
            print(f"Saving shard at {total_dataset_size} examples")
            _shard = Dataset.from_list(dataset_list)
            _shard.save_to_disk(args.output_dir + f"/shard_{shard_idx}")
            shard_idx += 1
            dataset_list = []

    if len(dataset_list) > 0:
        print(f"Saving shard at {total_dataset_size} examples")
        _shard = Dataset.from_list(dataset_list)
        _shard.save_to_disk(args.output_dir + "/shard_" + str(len(dataset_list)))

    print(f"Filtering took {time.time() - _time} seconds")
    print(f"Filtered dataset size: {total_dataset_size}, {total_dataset_size / subset_size * 100:.2f}% of the original dataset")

    # merge shards
    print("Started merging shards")
    _time = time.time()
    all_datasets = []
    for shard_path in tqdm(glob(args.output_dir + "/shard_*")):
        all_datasets.append(Dataset.load_from_disk(shard_path))
    
    dataset = datasets.concatenate_datasets(all_datasets)
    dataset.save_to_disk(args.output_dir)
    print(f"Merging took {time.time() - _time} seconds")
