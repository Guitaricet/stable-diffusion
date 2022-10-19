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
        if len(captions) != len(images):
            raise ValueError(f"captions and images must have the same length, got {len(captions)} and {len(images)}")

        if not distributed and torch.distributed.is_initialized():
            print("WARNING: CLIPScore is being run in a distributed environment, but distributed=False")

        # assert images.shape[1:] == (512, 512, 3), f"images must be of shape (N, 512, 512, 3), got {images.shape}"
        dataset = BasicDataset(captions, images)

        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset)

        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        score_sum = torch.tensor(0, device=self.device, dtype=torch.float32)  # we want to keep the score in float32 to increase precision

        for batch_captions, batch_images in dataloader:
            batch_captions = list(batch_captions)

            # import remote_pdb; remote_pdb.set_trace()
            # assert batch_images.shape == (24, 512, 512, 3)

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

        return score_sum / len(captions)
