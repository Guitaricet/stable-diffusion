# based on https://github.com/rinongal/textual_inversion/blob/5862ea4e3500a1042595bc199cfe5335703a458e/ldm/data/personalized.py
import os
import json

import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset


class TextCapsBase(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 interpolation="bicubic",
                 set="train",
                 ):
        """
        Args:
            data_root: path to the root of the dataset
            size: size of the image
            interpolation: interpolation method for resizing
            set: train/val/test
        """

        self.data_root = data_root
        with open(os.path.join(self.data_root, f"TextCaps_0.1_{set}.json")) as f:
            # keys are ['dataset_type', 'dataset_name', 'dataset_version', 'data']
            unflitered_data = json.load(f)["data"]

        self.data = []
        n_missing_files = 0
        for item in unflitered_data:
            if os.path.exists(os.path.join(self.data_root, item["image_path"])):
                self.data.append(item)
            else:
                n_missing_files += 1
        print(f"Removed {n_missing_files} items from {set} set because of missing files")
        print(f"Remaining items: {len(self.data)}")

        self.num_images = len(self.data)
        self._length = self.num_images 

        self.size = size
        self.interpolation = {"bilinear": PIL.Image.Resampling.BILINEAR,
                              "bicubic": PIL.Image.Resampling.BICUBIC,
                              "lanczos": PIL.Image.Resampling.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        item = self.data[i]
        image_path, caption_str = item["image_path"], item["caption_str"]
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
