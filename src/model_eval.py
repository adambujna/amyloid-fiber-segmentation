import os
import re
import json
import cv2 as cv
import numpy as np
import pandas as pd
import pycocotools.mask as maskutils
from tqdm import tqdm
from torch import cuda, mps
from ultralytics import YOLO, SAM


def parse_testfile_name(name: str):
    """Parses SNR, cluster level, background color, and fiber count from a filename."""
    match = re.search(r'snr([\d.]+)_cluster([\d.?]+)_bg(\d+)_sn(\d+)_n(\d+)_image_(\d+)', name)
    if not match:
        raise ValueError(f"Filename '{name}' does not match expected format.")
    return {
        "image_id": int(match.group(6)),
        "snr": float(match.group(1)),
        "clustering": float(match.group(2)),
        "background_color": int(match.group(3)),
        "fiber_color": int(match.group(4)),
        "num_fibers": int(match.group(5)),
    }


def iou_matrix(pred_masks, gt_masks):
    pred_masks = np.array(pred_masks).astype(bool)
    gt_masks = np.array(gt_masks).astype(bool)

    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))

    for i, pmask in enumerate(pred_masks):
        for j, gmask in enumerate(gt_masks):
            inter = np.logical_and(pmask, gmask).sum()
            union = np.logical_or(pmask, gmask).sum()
            iou_matrix[i, j] = inter / union if union > 0 else 0.0

    return iou_matrix


def mask_metrics(pred_masks, gt_masks, iou_threshold=0.5):
    if len(pred_masks) == 0 and len(gt_masks) == 0:
        return 1.0, 1.0, 1.0, 1.0  # Perfect match if nothing is predicted and nothing is present

    if len(pred_masks) == 0:
        return 0.0, 0.0, 0.0, 0.0

    if len(gt_masks) == 0:
        return 0.0, 0.0, 0.0, 0.0

    iou = iou_matrix(pred_masks, gt_masks)

    tp = 0

    matched_gt = set()
    for i in range(iou.shape[0]):
        best_j = np.argmax(iou[i])
        if iou[i, best_j] >= iou_threshold and best_j not in matched_gt:
            tp += 1
            matched_gt.add(best_j)

    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp

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


def load_gt_masks_sam(annotation_file: str) -> list[np.ndarray]:
    with open(os.path.abspath(annotation_file), "r") as f:
        ann = json.load(f)

    masks = []
    for a in ann["annotations"]:
        rle = a["segmentation"]
        masks.append(maskutils.decode(rle).astype(np.uint8) * 255)

    return masks


def evaluate_model(model: YOLO | SAM | str, test_image_dir: str, test_annotation_dir: str,
                   batch: int = 32, output_file: str = None, use_sam_annots: bool = True):
    if isinstance(model, str):
        try:
            model = YOLO(model)
        except FileNotFoundError:
            try:
                model = SAM(model)
            except FileNotFoundError:
                raise FileNotFoundError(f"{model} is not a supported model weights file.")

    if cuda.is_available():
        device = 'cuda'
    elif mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    image_names = [x for x in os.listdir(test_image_dir) if x.endswith(".png")]
    image_file_list = [os.path.join(os.path.abspath(test_image_dir), x) for x in image_names]

    results = []
    for i in tqdm(range(0, len(image_file_list), batch)):
        batch_files = image_file_list[i:i + batch]
        predictions = model.predict(batch_files, device=device, verbose=False)
        for image_file, result in zip(batch_files, predictions):
            if result.masks is None:
                continue
            filename = os.path.basename(image_file)
            meta = parse_testfile_name(filename)

            if isinstance(model, YOLO) and not use_sam_annots:
                annot_file = os.path.join(os.path.abspath(test_annotation_dir), f'{os.path.splitext(image_file)[0]}.txt')
                img = cv.imread(os.path.join(test_image_dir, image_file))
                size = img.shape
                gt_masks = load_gt_masks_yolo(annot_file, size[:2])
            elif isinstance(model, SAM) or use_sam_annots:
                annot_file = os.path.join(os.path.abspath(test_annotation_dir), f'{os.path.splitext(image_file)[0]}.json')
                gt_masks = load_gt_masks_sam(annot_file)
                size = gt_masks[0].shape
            else:
                raise ValueError("Model type is not supported.")

            pred_masks = []
            for mask in result.masks.data:
                mask = mask.cpu().numpy().astype(np.uint8)
                pred_masks.append(cv.resize(mask, size[:2][::-1], interpolation=cv.INTER_MAX))

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
    model = YOLO('../models/best.pt')
    evaluate_model(model,
                   test_image_dir='../data/datasets/sam_dataset/test/images',
                   test_annotation_dir='../data/datasets/sam_dataset/test/images',
                   batch=32,
                   use_sam_annots=True,
                   output_file='output.csv')
