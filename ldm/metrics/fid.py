import requests
from PIL import Image
import numpy as np

import torch
import torch_fidelity


class ImagesDataset(torch.utils.data.Dataset):
    """torch_fidelity requires a dataset object"""
    def __init__(self, images):
        self.images = images
    
    def __getitem__(self, index):
        return self.images[index]
    
    def __len__(self):
        return len(self.images)


def compute_fid(
    real_images,
    generated_images,
    batch_size=64,
    cuda=None,
    verbose=True,
):
    """
    Compute the FID score between real and generated images.
    :param real_images: (torch.Tensor) Real images. Shape: (N, C, H, W)
    :param generated_images: (torch.Tensor) Generated images. Shape: (N, C, H, W)
    :param batch_size: (int) Batch size for FID computation.
    :param num_workers: (int) Number of workers for FID computation.
    :param device: (str) Device to use for FID computation.
    :param verbose: (bool) Print FID score.
    :return: (float) FID score.
    """
    cuda = cuda if cuda is not None else torch.cuda.is_available()

    real_images_dataset = ImagesDataset(real_images)
    generated_images_dataset = ImagesDataset(generated_images)

    fid = torch_fidelity.calculate_metrics(
        input1=real_images_dataset,
        input2=generated_images_dataset,
        cuda=cuda,
        batch_size=batch_size,
        fid=True,
        save_cpu_ram=True,
        verbose=verbose,
    )

    return fid["frechet_inception_distance"]
