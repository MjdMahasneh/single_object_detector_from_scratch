import os
import cv2

def visualize_dataset(image_dir, label_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        with open(label_path, 'r') as f:
            coords = list(map(float, f.read().strip().split()))
            x1 = int(coords[0] * w)
            y1 = int(coords[1] * h)
            x2 = int(coords[2] * w)
            y2 = int(coords[3] * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('BBox', img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to break
            break

    cv2.destroyAllWindows()



# Example usage:
images_path = "./data/train/images"
labels_path = "./data/train/labels"
visualize_dataset(images_path, labels_path)



