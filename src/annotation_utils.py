import numpy as np
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


def read_real_em_annotation(annot_file):
    raise NotImplementedError
