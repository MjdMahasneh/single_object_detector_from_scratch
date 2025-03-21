from torchvision.ops import box_iou
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from bbox_dataset import BBoxDataset
from models import BBoxCNN
from config import Config

def evaluate(model, dataloader, device):
    """ Evaluate model on test set """
    model.eval()
    total_iou = 0
    count = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)

            iou = box_iou(preds, targets).diag()  # IoU per sample
            total_iou += iou.sum().item()
            count += targets.size(0)
    # print(f"Avg IoU: {total_iou / count:.4f}")

    # set model back to train mode
    model.train()

    return total_iou / count





if __name__ == "__main__":

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    cfg = Config()

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size)),
        transforms.ToTensor(),
    ])

    # Load test dataset
    test_dataset = BBoxDataset(cfg.test_images, cfg.test_labels, transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    # Load model weights
    model = BBoxCNN(backbone_name=cfg.backbone, freeze_backbone=cfg.freeze_backbone).to(device)
    checkpoint_path = cfg.checkpoints_dir + '/best_model_SE.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    # Run evaluation
    iou = evaluate(model, test_loader, device)
    print(f"Avg IoU: {iou:.4f}")

