# based on https://github.com/rinongal/textual_inversion/blob/5862ea4e3500a1042595bc199cfe5335703a458e/ldm/data/personalized.py
import os
import json
from glob import glob

import numpy as np
import torch.utils.data
from PIL import Image
from datasets import Dataset as HFDataset

from tqdm.auto import tqdm
from loguru import logger


class TextyCaps(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 size,
                 interpolation="bicubic",
                 verbose=True,
                 ):
        """Expected directory structure:

        data_root/
            shard_000001/
                00029013.jpg
                00029013.json
                ...
            ...

        Args:
            data_root: path to the root of the dataset
            size: size of the image
            interpolation: interpolation method for resizing
            set: train/val/test
        """
        logger.info(f"Loading TextyCaps dataset from {data_root}")
        self.data_root = data_root

        hf_dataset_path = os.path.join(data_root, "dataset.arrow")
        if os.path.exists(hf_dataset_path):
            logger.info("Found cached Arrow dataset, loading...")
            self.data = HFDataset.load_from_disk(hf_dataset_path)
        else:
            logger.info("No cached Arrow dataset found. Converting to Arrow format to reduce future loading time...")
            data = self._convert_to_arrow(data_root)
            self.data = HFDataset.from_list(data)
            self.data.save_to_disk(hf_dataset_path)

        self.num_images = len(self.data)
        self._length = self.num_images

        self.size = size
        self.interpolation = {"bilinear": Image.Resampling.BILINEAR,
                              "bicubic": Image.Resampling.BICUBIC,
                              "lanczos": Image.Resampling.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        item = self.data[i]
        image_path, caption_str = item["image_path"], item["caption"]
        image_path = os.path.join(self.data_root, image_path)

        example = {}
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["caption"] = caption_str

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example

    @staticmethod
    def _convert_to_arrow(data_root, verbose=False):
        data = []
        n_missing_files = 0
        for shard in tqdm(glob(os.path.join(data_root, "shard_*")), disable=not verbose, desc="Loading TextyCaps"):
            for item in glob(os.path.join(shard, "*.json")):
                with open(item) as f:
                    image_path = f"{item[:-5]}.jpg"
                    if not os.path.exists(image_path):
                        n_missing_files += 1
                        continue
                    meta = json.load(f)
                    meta["image_path"] = image_path
                    data.append(meta)

        logger.info(f"{n_missing_files} missing files when loading data from {data_root}")
        logger.info(f"Remaining items: {len(data)}")
        return data
