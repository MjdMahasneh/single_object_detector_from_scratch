import cv2
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from models import BBoxCNN
from config import Config

def predict_and_visualize(model, image_dir, image_size=(224, 224)):
    """ Predict bounding boxes on images and visualize the results """
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)

        # Load original image (for display size)
        original_image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = original_image.size

        # Resize for model input
        input_image = transform(original_image)
        input_tensor = input_image.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(input_tensor)[0].cpu().numpy()

        # Map normalized predictions to original size
        x1 = int(pred[0] * orig_w)
        y1 = int(pred[1] * orig_h)
        x2 = int(pred[2] * orig_w)
        y2 = int(pred[3] * orig_h)

        # Draw on original image (OpenCV expects BGR)
        img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Prediction', img_cv)
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    cfg = Config()
    checkpoint_path = cfg.checkpoints_dir + '/best_model_SE.pth'  # update as needed

    # Load model
    model = BBoxCNN(backbone_name=cfg.backbone, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    # Run prediction and visualization
    predict_and_visualize(model, cfg.test_images, cfg.image_size)
