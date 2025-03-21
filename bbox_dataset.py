import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class BBoxDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        image = Image.open(img_path).convert('RGB')
        with open(label_path, 'r') as f:
            label = list(map(float, f.read().strip().split()))
        bbox = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            if isinstance(self.transform, transforms.Compose):
                image = self.transform(image)
            else:
                image, bbox = self.transform(image, bbox)

        return image, bbox
