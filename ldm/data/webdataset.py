import numpy as np
import webdataset as wds

from PIL import Image
from torch.utils.data import default_collate


class TextyCapsWebdataset:
    """Webdataset is highly recommended over regular datasets when using cloud.
    It is much faster to load and doesn't require downloading the whole dataset to the local machine.

    Args:
        url: e.g.,"https://storage.googleapis.com/BUCKET_NAME/texty-caps-train-shard_{000001..000064}.tar"
        image_size: size of the image, e.g. 512 means all images will be resized to 512x512
        batch_size: batch size, Webdataset does batching internally and when wrapping it in a dataloader
            you need to set batch_size=None
    """
    def __init__(self, url, image_size, batch_size, interpolation="bicubic", shuffle=True) -> None:
        self.url = f'pipe:curl -L -s {url} || true'
        self.size = image_size
        self.batch_size = batch_size
        self.interpolation = {"bilinear": Image.Resampling.BILINEAR,
                              "bicubic": Image.Resampling.BICUBIC,
                              "lanczos": Image.Resampling.LANCZOS,
                              }[interpolation]

        dataset = wds.WebDataset(self.url, nodesplitter=wds.split_by_node)

        if shuffle:
            dataset = dataset.shuffle(10_000)

        self.dataset = (
            dataset
            .decode("pil")
            .to_tuple("jpg;png", "json")
            .map_tuple(self.preprocess_image, self.preprocess_meta)
            .batched(self.batch_size, partial=False, collation_fn=self.collate_fn)
        )

    def collate_fn(self, batch):
        images, captions = default_collate(batch)
        return {"image": images, "captions": captions}

    def get_dataset(self):
        return self.dataset

    def preprocess_image(self, img):
        if not img.mode == "RGB":
            img = img.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(img).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        img = Image.fromarray(img)
        if self.size is not None:
            img = img.resize((self.size, self.size), resample=self.interpolation)

        img = np.array(img).astype(np.uint8)

        img = (img / 127.5 - 1.0).astype(np.float32)
        return img

    def preprocess_meta(self, target):
        target = target["caption"]
        return target
