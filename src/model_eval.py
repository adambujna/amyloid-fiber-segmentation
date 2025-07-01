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

from src.annotation_utils import sample_points, load_gt_masks_sam, load_gt_masks_yolo
from src.image_processing import get_length


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


def mask_metrics(pred_masks, gt_masks, iou_threshold=0.5, length_error=False):
    if len(pred_masks) == 0 and len(gt_masks) == 0:
        return 1.0, 1.0, 1.0, 1.0  # Perfect match if nothing is predicted and nothing is present

    if len(pred_masks) == 0:
        return 0.0, 0.0, 0.0, 0.0

    if len(gt_masks) == 0:
        return 0.0, 0.0, 0.0, 0.0

    iou = iou_matrix(pred_masks, gt_masks)

    tp = 0

    matched_gt = set()
    length_error_all = []
    for i in range(iou.shape[0]):
        best_j = np.argmax(iou[i])
        if iou[i, best_j] >= iou_threshold and best_j not in matched_gt:
            tp += 1
            if length_error:
                pred_length = get_length(pred_masks[i])
                gt_length = get_length(gt_masks[best_j])
                len_err = abs(gt_length - pred_length) / max(1.0, gt_length)
                length_error_all.append(len_err)
            matched_gt.add(best_j)

    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp

    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1, np.median(length_error_all)


def evaluate_model(model: YOLO | SAM | str, test_image_dir: str, test_annotation_dir: str,
                   batch: int = 32, output_file: str = None, use_sam_annots: bool = True, verbose=True):
    if isinstance(model, str):
        try:
            model = YOLO(model)
        except FileNotFoundError:
            try:
                model = SAM(model)
            except FileNotFoundError:
                raise FileNotFoundError(f"{model} is not a supported model weights file.")

    if isinstance(model, SAM):
        batch = 1   # SAM does not currently support batched predictions

    if cuda.is_available():
        device = 'cuda'
    elif mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    if verbose:
        print("Using device:", device)

    image_names = [x for x in os.listdir(test_image_dir) if x.endswith(".png")]
    image_file_list = [os.path.join(os.path.abspath(test_image_dir), x) for x in image_names]

    results = []
    for i in tqdm(range(0, len(image_file_list), batch)):
        batch_files = image_file_list[i:i + batch]
        predictions = model.predict(batch_files, device=device, conf=0.2, verbose=False)
        for image_file, result in zip(batch_files, predictions):
            if result.masks is None:
                continue
            filename = os.path.basename(image_file)
            try:
                meta = parse_testfile_name(filename)
            except ValueError:
                print(f"WARNING: Could not parse metadata from file {filename}.")
                meta = {}

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

            accuracy, precision, recall, f1, length_error = mask_metrics(pred_masks, gt_masks, length_error=True)

            results.append({
                **meta,
                "filename": filename,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "length_error": length_error,
                "num_gt": len(gt_masks),
                "num_pred": len(pred_masks),
            })

            if verbose:
                print(f"\t\nEvaluated {filename}\nF1={f1:.2f}\nPrecision={precision:.2f}\nRecall={recall:.2f}")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Saved results to {output_file}")

    return results


def evaluate_model_prompt(
        model: SAM,
        test_image_dir: str,
        test_annotation_dir: str,
        n_points: int = 1,
        output_file: str | None = None,
        use_sam_annots: bool = True,
        seed: int | None = None,
        verbose: bool = True):
    """
    Evaluate a SAM model with n point prompts for each image sampled from ground‑truth masks.
    Does not support batched prediction.
    """

    if isinstance(model, str):
        try:
            model = SAM(model)
        except FileNotFoundError:
            try:
                model = YOLO(model)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"{model} is not a supported model weights file."
                )

    is_sam = isinstance(model, SAM)

    if cuda.is_available():
        device = "cuda"
    elif mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if verbose:
        print("Using device:", device)

    if seed is not None:
        np.random.seed(seed)

    image_names = [x for x in os.listdir(test_image_dir) if x.endswith(".png")]
    image_paths = [os.path.join(test_image_dir, x) for x in image_names]

    results = []

    for img_path in tqdm(image_paths):
        filename = os.path.basename(img_path)
        try:
            meta = parse_testfile_name(filename)
        except ValueError:
            print(f"WARNING: Could not parse metadata from file {filename}.")
            meta = {}

        if use_sam_annots:
            annot_path = os.path.join(
                test_annotation_dir, f"{os.path.splitext(filename)[0]}.json"
            )
            gt_masks = load_gt_masks_sam(annot_path)
            img_size = gt_masks[0].shape
        else:
            annot_path = os.path.join(
                test_annotation_dir, f"{os.path.splitext(filename)[0]}.txt"
            )
            img = cv.imread(img_path)
            img_size = img.shape[:2]
            gt_masks = load_gt_masks_yolo(annot_path, img_size)

        point_coords_all = []
        point_labels_all = []

        for gt in gt_masks:
            pts = sample_points(gt, n_points=n_points)
            if pts.size == 0:
                continue
            point_coords_all.append(pts)
            point_labels_all.append(np.ones(len(pts), dtype=np.int32))

        point_coords = np.array(point_coords_all) if point_coords_all else None
        point_labels = np.array(point_labels_all) if point_labels_all else None

        if is_sam and point_coords is not None and point_labels is not None:
            predictions = model.predict(
                img_path,
                points=point_coords,
                labels=point_labels,
                conf=0.2,
                device=device,
                verbose=False
            )
            masks_pred_raw = predictions[0].masks.data if predictions[0].masks else []
        else:
            # YOLO fallback
            predictions = model.predict(img_path, device=device, conf=0.2, verbose=False)
            masks_pred_raw = predictions[0].masks.data if predictions[0].masks else []

        pred_masks = [
            cv.resize(m.cpu().numpy().astype(np.uint8),
                      img_size[:2][::-1], interpolation=cv.INTER_NEAREST)
            for m in masks_pred_raw
        ]

        accuracy, precision, recall, f1, length_error = mask_metrics(pred_masks, gt_masks, length_error=True)

        results.append({
            **meta,
            "filename": filename,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "length_error": length_error,
            "num_gt": len(gt_masks),
            "num_pred": len(pred_masks),
        })

        if verbose:
            print(f"\nEvaluated {filename}"
                  f"\nF1={f1:.2f}  Precision={precision:.2f}  Recall={recall:.2f}")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Saved results to {output_file}")

    return results


if __name__ == "__main__":
    model = YOLO('../model_checkpoints/yolo11nano-fibersegmentation.pt')
    evaluate_model(model,
                   test_image_dir='../data/datasets/sam_dataset/test/images',
                   test_annotation_dir='../data/datasets/sam_dataset/test/images',
                   batch=1,
                   use_sam_annots=True,
                   output_file='../data/results/yolo11nano.csv',
                   verbose=False)
