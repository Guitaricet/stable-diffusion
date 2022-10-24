import torch
import torch.distributed
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModel, AutoProcessor


class BasicDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CLIPScore:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu", dtype=torch.float32, distributed=False):
        # "openai/clip-vit-large-patch14" vs "openai/clip-vit-base-patch32" ?
        self.dtype = dtype
        self.device = device
        self.distributed = distributed
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def compute(self, *, captions, images, batch_size=1, distributed=False):
        """
        Args:
            captions (list): list of strings
            images (list): list of PIL images
            batch_size (int): batch size
            distributed (bool): whether to use distributed sampler (not tested yet)
        Returns:
            score (float): CLIP score
        """
        if len(captions) != len(images):
            raise ValueError(f"captions and images must have the same length, got {len(captions)} and {len(images)}")

        if not distributed and torch.distributed.is_initialized():
            print("WARNING: CLIPScore is being run in a distributed environment, but distributed=False")

        # assert images.shape[1:] == (512, 512, 3), f"images must be of shape (N, 512, 512, 3), got {images.shape}"
        dataset = BasicDataset(captions, images)

        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset)

        def collate_fn(batch):
            captions, images = zip(*batch)
            return list(captions), list(images)

        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
        score_sum = torch.tensor(0, device=self.device, dtype=torch.float32)  # we want to keep the score in float32 to increase precision

        self.model = self.model.to(device=self.device, dtype=self.dtype)

        for batch_captions, batch_images in dataloader:
            inputs = self.processor(images=batch_images, text=batch_captions, return_tensors="pt", padding=True).to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            model_outputs = self.model(**inputs)

            text_features = model_outputs.text_embeds
            image_features = model_outputs.image_embeds

            # compute cosine similarity between image and text features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            score_sum += (text_features * image_features).sum().float()

        if distributed:
            torch.distributed.all_reduce(score_sum, op=torch.distributed.ReduceOp.SUM)
        
        self.model = self.model.to(device="cpu", dtype=torch.float32)

        return (score_sum / len(captions)).item()
