import os
import json
import numpy as np
import cv2 as cv
import pycocotools.mask as mask_utils
from datetime import datetime


def create_annotation_entry(mask: np.ndarray, annotation_id: int, image_size: tuple[int, int]) -> dict:
    """
    Creates an annotation dictionary in SA1B format for a single binary mask.

    Parameters
    ----------
    mask : np.ndarray
        The binary segmentation mask.
    annotation_id : int
        The unique ID for this specific annotation instance.
    image_size : tuple[int, int]
        The (height, width) of the full image.

    Returns
    -------
    dict: A dictionary representing the annotation in SA1B format.
    """
    mask = np.asfortranarray(mask)

    rle = mask_utils.encode(mask)   # Encoding into compressed RLE using pycocotools
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


def create_image_entry(img_name: str, image_size: tuple[int, int], image_id: int) -> dict:
    """
    Creates an image information dictionary in SA1B format for a single image.

    Parameters
    ----------
    img_name : str
        The file name of the image.
    image_size : tuple[int, int]
        The (height, width) of the image.
    image_id : int
        The unique ID for this image.

    Returns
    -------
    dict: A dictionary representing the image label in SA1B format.
    """
    height, width = image_size
    return {
        "image_id": image_id,
        "license": 1,
        "file_name": img_name,
        "height": height,
        "width": width,
        "date_captured": datetime.now().isoformat()
    }


def sample_points(mask: np.ndarray, n_points: int = 1) -> np.ndarray:
    """
    Gets the coordinates of `n_points` random points from within a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        The binary mask from which to sample points.
    n_points : int, optional
        The number of points to sample.
        Default is 1.

    Returns
    -------
    np.ndarray: A NumPy array of shape (n_points, 2) containing [x, y] coordinates of the `n_points` points.
    """
    ys, xs = np.where(mask > 0)     # This works for both (0-1) and (0-255) binary masks
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    idx = np.random.choice(len(xs), size=n_points, replace=len(xs) < n_points)

    return np.stack([xs[idx], ys[idx]], axis=1)


def load_gt_masks_yolo(annotation_file: str, image_shape: tuple[int, int]) -> list[np.ndarray]:
    """
    Loads segmentation masks from a YOLO-format annotation text file.

    Parameters
    ----------
    annotation_file : str
        The path to the YOLO-format .txt annotation file.
    image_shape : tuple[int, int]
        The (height, width) of the image, used for non-normalizing coordinates.

    Returns
    -------
    list[np.ndarray]: A list of binary masks, one for each segmented object.
    """
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
    """
    Loads segmentation masks from a SA1B (SAM-format) JSON file.

    Parameters
    ----------
    annotation_file : str
        The path to the JSON annotation file.

    Returns
    -------
    list[np.ndarray]: A list of binary masks, one for each segmented object.
    """
    with open(os.path.abspath(annotation_file), "r") as f:
        ann = json.load(f)

    masks = []
    for a in ann["annotations"]:    # Decode the RLE of every segmentation in the JSON
        rle = a["segmentation"]
        masks.append(mask_utils.decode(rle).astype(np.uint8) * 255)

    return masks
