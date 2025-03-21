import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from bbox_dataset import BBoxDataset
from models import BBoxCNN
from loss import init_loss_fn
from trainer import train
from config import Config
from augment import Augment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Config
cfg = Config()
image_size = cfg.image_size
lr = cfg.lr
use_scheduler = cfg.use_scheduler
loss_fn = cfg.loss_fn
batch_size = cfg.batch_size
shuffle = cfg.shuffle
epochs = cfg.epochs
backbone= cfg.backbone
freeze_backbone=cfg.freeze_backbone
eval_every = cfg.eval_every
train_images = cfg.train_images
train_labels = cfg.train_labels
val_images = cfg.test_images
val_labels = cfg.test_labels
checkpoints_dir = cfg.checkpoints_dir

os.makedirs(checkpoints_dir, exist_ok=True)


# Train transforms
train_transform = Augment(
    hflip=cfg.hflip,
    vflip=cfg.vflip,
    rotate=cfg.rotate,
    color_jitter=cfg.color_jitter,
    grayscale=cfg.grayscale,
    blur=cfg.blur,
    sharpness=cfg.sharpness
)

# Test transforms
test_transform = transforms.Compose([transforms.Resize(image_size),
                                     transforms.ToTensor(),
                                     ])

# Load datasets
train_dataset = BBoxDataset(train_images, train_labels, transform=train_transform)
test_dataset = BBoxDataset(val_images, val_labels, transform=test_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# Initialize model
model = BBoxCNN(backbone_name=backbone, freeze_backbone=freeze_backbone).to(device)

# Initialize loss function
criterion = init_loss_fn(loss_fn)

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Initialize scheduler
if use_scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
else:
    scheduler = None


if __name__ == "__main__":

    # Train model
    train(model,
          train_loader, test_loader,
          criterion, optimizer,
          epochs=epochs, eval_every=eval_every,
          device=device,
          use_scheduler=False, scheduler=scheduler,
          checkpoints_dir=checkpoints_dir)
