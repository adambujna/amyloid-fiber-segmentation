import os
import json
import numpy as np
import cv2 as cv
import pycocotools.mask as mask_utils
from datetime import datetime


def create_annotation_entry(mask, annotation_id, image_size):
    mask = np.asfortranarray(mask)

    rle = mask_utils.encode(mask)
    rle["counts"] = rle["counts"].decode("ascii")

    bbox = mask_utils.toBbox(rle).tolist()

    area = float(mask_utils.area(rle))

    return {
        "id": annotation_id,
        "bbox": bbox,
        "area": area,
        "segmentation": {
            "counts": rle["counts"],
            "size": list(image_size)
        }
    }


def create_image_entry(img_name, image_size, image_id):
    height, width = image_size
    return {
        "image_id": image_id,
        "license": 1,
        "file_name": img_name,
        "height": height,
        "width": width,
        "date_captured": datetime.now().isoformat()
    }


def sample_points(mask, n_points=1):
    """Return n_points random coordinates inside a mask for SAM prompting."""
    ys, xs = np.where(mask > 1)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    idx = np.random.choice(len(xs), size=n_points, replace=len(xs) < n_points)
    return np.stack([xs[idx], ys[idx]], axis=1)


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
        masks.append(mask_utils.decode(rle).astype(np.uint8) * 255)

    return masks
