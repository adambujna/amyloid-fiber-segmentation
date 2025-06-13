import os
import re
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.yolo11 import Yolo11SegmenterEM


def parse_testfile_name(name: str):
    """Parses SNR, cluster level, background color, and fiber count from a filename."""
    match = re.search(r'image(\d+)_snr([\d.]+)_cluster([\d.?]+)_bg(\d+)_sn(\d+)_n(\d+)', name)
    if not match:
        raise ValueError(f"Filename '{name}' does not match expected format.")
    return {
        "image_id": int(match.group(1)),
        "snr": float(match.group(2)),
        "clustering": float(match.group(3)),
        "background_color": int(match.group(4)),
        "fiber_color": int(match.group(5)),
        "num_fibers": int(match.group(6)),
    }


def mask_metrics(pred_masks, gt_masks, iou_threshold=0.5):
    def iou(mask1, mask2):
        inter = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return inter / union if union != 0 else 0

    matched_gt = set()
    tp, fp, fn = 0, 0, 0

    for i, gmask in enumerate(gt_masks):
        if i in matched_gt:
            continue
        found_match = False
        for pmask in pred_masks:
            if iou(pmask, gmask) >= iou_threshold:
                matched_gt.add(i)
                found_match = True
                break
        if found_match:
            tp += 1
        else:
            fn += 1

    fp = len(pred_masks) - tp

    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1


def load_gt_masks_yolo(annotation_file: str, image_shape: tuple[int, int]) -> list[np.ndarray]:
    with open(os.path.abspath(annotation_file)) as f:
        lines = f.readlines()
    annotations = [(line.strip().split(" ")[0], line.strip().split(" ")[1:]) for line in lines]
    masks = []
    for _, contour in annotations:
        mask = np.zeros(image_shape, dtype=np.uint8)
        points = [[[int(float(contour[i]) * image_shape[1]), int(float(contour[i + 1]) * image_shape[0])] for i
                  in range(0, len(contour), 2)]]
        cv.fillPoly(mask, np.array(points), [255])
        masks.append(mask)

    return masks


def evaluate_model(model: Yolo11SegmenterEM, test_image_dir: str, test_annotation_dir: str, output_file: str = None):
    results = []
    for image_file in tqdm(os.listdir(test_image_dir)):
        if not image_file.endswith(".png"):
            continue
        filename = os.path.basename(image_file)
        meta = parse_testfile_name(filename)
        img = cv.imread(os.path.join(test_image_dir, image_file))
        size = img.shape

        if isinstance(model, Yolo11SegmenterEM):
            annotation_file = os.path.join(os.path.abspath(test_annotation_dir), f'{filename[:-4]}.txt')
            gt_masks = load_gt_masks_yolo(annotation_file, size[:2])
        else:
            raise ValueError(f"Model type '{type(model)}' is not supported.")

        pred_masks = model.predict(img, plot=False)
        pred_masks = [cv.resize(m, size[:2][::-1]) for m in pred_masks if m.shape != size[:2][::-1]]

        accuracy, precision, recall, f1 = mask_metrics(pred_masks, gt_masks)

        results.append({
            **meta,
            "filename": filename,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "num_gt": len(gt_masks),
            "num_pred": len(pred_masks),
        })

        print(f"\t\nEvaluated {filename}\nF1={f1:.2f}\nPrecision={precision:.2f}\nRecall={recall:.2f}")

    if output_file:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Saved results to {output_file}")


if __name__ == "__main__":
    model = Yolo11SegmenterEM('../models/best.pt')
    evaluate_model(model,
                   test_image_dir='/Users/ab/Documents/AIDS/Bachelors Thesis/Code/data/test_synthetic_images/images',
                   test_annotation_dir='/Users/ab/Documents/AIDS/Bachelors Thesis/Code/data/test_synthetic_images/labels',
                   output_file='output.csv')
