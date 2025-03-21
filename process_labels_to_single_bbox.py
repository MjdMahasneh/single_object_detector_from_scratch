import os
import glob

def parse_and_merge_label(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = map(float, parts)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes.append([x1, y1, x2, y2])

    if not boxes:
        return [0, 0, 0, 0]

    x1s, y1s, x2s, y2s = zip(*boxes)
    merged = [min(x1s), min(y1s), max(x2s), max(y2s)]
    return merged

def convert_and_save_labels(label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))

    for file in label_files:
        merged = parse_and_merge_label(file)
        base = os.path.basename(file)
        out_path = os.path.join(output_dir, base)
        with open(out_path, 'w') as f:
            f.write(' '.join(map(str, merged)))

# Example usage:
subsets = ["train", "valid", "test"]
for split in subsets:
    labels_path = "/raw_data/"+ split +"/labels"
    convert_and_save_labels(labels_path, "./output/"+ split +"/labels")
