import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random


class Augment:
    def __init__(self, image_size=(224, 224),
                 hflip=True, vflip=True, rotate=True,
                 color_jitter=True, grayscale=True,
                 blur=True, sharpness=True):

        self.resize = transforms.Resize(image_size)
        self.to_tensor = transforms.ToTensor()

        self.enable_hflip = hflip
        self.enable_vflip = vflip
        self.enable_rotate = rotate
        self.enable_jitter = color_jitter
        self.enable_gray = grayscale
        self.enable_blur = blur
        self.enable_sharp = sharpness

        self.jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.blur_op = transforms.GaussianBlur(3)
        self.gray_op = transforms.Grayscale(num_output_channels=3)
        self.sharp_op = transforms.RandomAdjustSharpness(2)

    def __call__(self, image, bbox):
        image = self.resize(image)
        w, h = image.size
        bbox = bbox.clone()

        if self.enable_hflip and random.random() < 0.5:
            image = F.hflip(image)
            bbox[0], bbox[2] = 1 - bbox[2], 1 - bbox[0]

        if self.enable_vflip and random.random() < 0.5:
            image = F.vflip(image)
            bbox[1], bbox[3] = 1 - bbox[3], 1 - bbox[1]

        if self.enable_rotate:
            rot = random.choice([0, 90, 180, 270])
            if rot == 90:
                image = image.rotate(90, expand=True)
                bbox = torch.tensor([bbox[1], 1 - bbox[2], bbox[3], 1 - bbox[0]])
            elif rot == 180:
                image = image.rotate(180, expand=True)
                bbox = torch.tensor([1 - bbox[2], 1 - bbox[3], 1 - bbox[0], 1 - bbox[1]])
            elif rot == 270:
                image = image.rotate(270, expand=True)
                bbox = torch.tensor([1 - bbox[3], bbox[0], 1 - bbox[1], bbox[2]])

        if self.enable_jitter:
            image = self.jitter(image)
        if self.enable_gray:
            image = self.gray_op(image)
        if self.enable_sharp:
            image = self.sharp_op(image)
        if self.enable_blur:
            image = self.blur_op(image)

        image = self.to_tensor(image)
        return image, bbox
