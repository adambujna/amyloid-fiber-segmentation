import os
import json
import cv2 as cv
import numpy as np
import pycocotools.mask as mask_utils
from src.tiling import tile_image
from src.annotation_utils import create_image_entry, create_annotation_entry


def save_image(image_name: str, image: np.ndarray, save_dir: str) -> None:
    """
    Saves an NumPy array as an image.

    Parameters
    ----------
    image_name : str
        The base name for the output image file (without extension).
    image : np.ndarray
        The image array to be saved.
    save_dir : str
        The directory where the image will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    cv.imwrite(os.path.join(save_dir, f'{image_name}.png'), image)


def resize_image(image_path: str, size: int | tuple[int, int], save_dir: str = None,
                 sam_format: bool = False, sam_json: str = None, new_label_dir: str = None) -> None:
    """
    Resizes an image and, if applicable, its corresponding annotations.
    Saves image as PNG.

    If `size` is an integer, it scales the image's largest dimension to `size` while maintaining the aspect ratio.
    If `size` is a tuple (width, height), it stretches the image to those exact dimensions.

    It also resizes the segmentation masks within a corresponding JSON annotation file
    to match the new image dimensions.

    Parameters
    ----------
    image_path : str
        Path to the image to be resized.
    size : int | tuple[int, int]
        The target size for the output image.
    save_dir : str, optional
        Directory to save the resized image.
        If None, overwrites the original image.
        Default is None.
    sam_format : bool, optional
        If True, the function will also attempt to resize annotations in a
        corresponding SA1B formatted JSON file.
        Default is False.
    sam_json : str, optional
        Path to the SA1B JSON annotation file to be resized.
        Required if `sam_format` is True.
    new_label_dir : str, optional
        Directory to save the resized annotation file.
        If None, overwrites the original JSON.
        Default is None.
    """

    if save_dir is None:    # Overwrites if different save_dir not specified.
        save_dir = os.path.dirname(image_path)
    os.makedirs(save_dir, exist_ok=True)
    new_path = os.path.join(save_dir, os.path.basename(image_path))

    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not find image at {image_path}")

    if isinstance(size, int):
        h, w = image.shape[:2]
        scale = size / max(w, h)
        size = (int(w * scale), int(h * scale))
    # Also tried Lanczos interpolation from OpenCV but maintained noise extremely well, so not worth.
    size = size[:2]
    resized_img = cv.resize(image, size, interpolation=cv.INTER_AREA)
    cv.imwrite(new_path, resized_img)

    if sam_format and sam_json is not None:     # Resizing sam annotations
        if new_label_dir is None:
            new_label_dir = os.path.dirname(sam_json)
        os.makedirs(new_label_dir, exist_ok=True)

        with open(sam_json, 'r') as f:
            data = json.load(f)

        # Resize image entry
        data['image']['width'] = size[0]
        data['image']['height'] = size[1]

        # Resize annotations
        new_annots = []
        for ann in data.get('annotations', []):
            segm = ann.get('segmentation')
            if segm is None:
                continue
            mask = mask_utils.decode(segm)

            # Resize mask
            resized_mask = cv.resize(mask, size, interpolation=cv.INTER_NEAREST)
            new_annots.append(create_annotation_entry(resized_mask, ann.get('id', 0), size[::-1]))
        data['annotations'] = new_annots

        # Save new json
        new_json_path = os.path.join(new_label_dir, os.path.basename(sam_json))
        with open(new_json_path, 'w') as f:
            json.dump(data, f)


def resize_images_dir(image_dir: str, size: int | tuple[int, int], save_dir: str = None,
                      sam_format: bool = False, sam_json_dir: str = None, new_label_dir: str = None,
                      valid_extensions: tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> None:
    """
    Resizes all images in a source directory by applying `resize_image` to each.

    Can optionally resize the corresponding SA1B formatted JSON annotations.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing images to resize.
    size : int | tuple[int, int]
        The target size, passed to `resize_image`.
    save_dir : str, optional
        Destination directory for resized images.
        If None, overwrites the original images.
        Default is None.
    sam_format : bool, optional
        If True, enables resizing of corresponding SA1B annotations from JSON.
    sam_json_dir : str, optional
        The directory containing the original JSON annotation files.
        Required if `sam_format` is True.
    new_label_dir : str, optional
        The directory where resized JSON files will be saved.
    valid_extensions : tuple[str, ...], optional
        A tuple of file extensions to be considered as valid images.
    """
    if save_dir is None:
        save_dir = image_dir
    os.makedirs(save_dir, exist_ok=True)

    fnames = os.listdir(image_dir)
    fnames = [f for f in fnames if f.lower().endswith(valid_extensions)]

    for i, filename in enumerate(fnames):
        if not filename.lower().endswith(valid_extensions):     # Skip invalid images
            continue

        print(f'\r({i+1}/{len(fnames)}) Resizing {filename} to {size}', end='', flush=True)

        # Get image's path
        src_path = os.path.join(image_dir, filename)
        # Get JSON path
        json_path = None
        if sam_format and sam_json_dir is not None:
            json_path = os.path.join(sam_json_dir, f"{os.path.splitext(filename)[0]}.json")

        # Resize image and, if applicable, annotation
        resize_image(src_path, size=size, save_dir=save_dir,
                     sam_format=sam_format, sam_json=json_path, new_label_dir=new_label_dir)


def save_label_yolo(image_name: str, contours: list[np.ndarray], image_size: tuple[int, int], label_dir: str) -> None:
    """
    Saves fiber contours in YOLO label annotation_format for segmentation.

    This function normalizes the contours to be relative to the image dimensions (0-1) and
    saves them one line per object.

    Parameters
    ----------
    image_name : str
        The name for the output label file (without extension).
    contours : list[np.ndarray]
        A list of contours, where each contour is a NumPy array of points.
    image_size : tuple[int, int]
        The (height, width) of the image, used for normalization.
    label_dir : str
        The directory where the .txt label file will be saved.
    """
    os.makedirs(label_dir, exist_ok=True)

    # Normalize contours to be relative to image size
    contours = [[[point[0, 0] / image_size[1], point[0, 1] / image_size[0]]
                 for point in contour] for contour in contours]
    # Format them into YOLO annotation_format as '{label index} {x1} {y1} {x2} {y2} ... {xn} {yn}'
    formatted_contours = "\n".join(f'0 {" ".join(f"{pt:.7f}" for coord in contour for pt in coord)}'
                                   for contour in contours)

    with open(os.path.join(label_dir, f'{image_name}.txt'), 'w') as file:
        file.write(formatted_contours)


def save_label_sam(image_name: str, masks: list[np.ndarray], label_dir: str) -> None:
    """
    Saves fiber mask in SA1B-Dataset formatted JSON.

    Creates an image information dictionary with `create_image_entry` and
    annotation dictionary with `create_annotation_entry` and
    merges them into the JSON entry of an image with labels.

    Parameters
    ----------
    image_name : str
        The base name for the output JSON file (without extension).
    masks : list[np.ndarray]
        A list of binary segmentation masks for the objects in the image.
    label_dir : str
        The directory where the .json label file will be saved.
    """
    os.makedirs(label_dir, exist_ok=True)
    image_id = int(image_name.split('_')[-1])

    image_entry = create_image_entry(image_name, masks[0].shape[:2], image_id)
    annots = []
    for i, m in enumerate(masks):
        annots.append(create_annotation_entry(m, image_id*1000+1, m.shape[:2]))

    output = {
        "image": image_entry,
        "annotations": annots
    }

    with open(os.path.join(label_dir, f"{image_name}.json"), "w") as f:
        json.dump(output, f)


def split_image_datapoint_yolo(image_path: str, label_path: str = None,
                               size: tuple[int, int] = (1024, 1024), min_overlap: int = 256,
                               image_save_dir: str = None, label_save_dir: str = None,
                               verbose: int = 0) -> tuple[list[str], list[str]]:
    """
    Reads an existing image and its labels in YOLO annotation_format and saves them as several overlapping mosaic images with new
    labels in YOLO annotation_format.

    Parameters
    ----------
    image_path: str
        Path to the image which will be split.
    label_path: str
        Path to the label of the image which will be split.
    size: tuple[int, int]
        Size of every mosaic tile image.
    min_overlap: int, optional
        The minimum number of pixels of overlap between two neighboring mosaics. Defaults to 128
    image_save_dir: str, optional
        Directory where the tile images will be saved. None saves them to the directory of the original image.
        Defaults to None.
    label_save_dir: str, optional
        Directory where new labels of tile images will be saved. None saves them to the directory of the original label.
        Defaults to None.
    verbose: int, optional
        How much information will be printed about the process of calculating overlaps and numbers of tiles required.
        Defaults to Zero.

    Returns
    -------
    Tuple[list[str], list[str]]: Paths to the newly created images and labels of newly created labels.
    """

    image_save_dir = image_save_dir or os.path.dirname(image_path)
    if label_save_dir is None:
        label_save_dir = os.path.dirname(label_path) if label_path else None
    os.makedirs(image_save_dir, exist_ok=True)
    if label_save_dir:
        os.makedirs(label_save_dir, exist_ok=True)

    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not find image at {image_path}")

    label_lines = []
    if label_save_dir:
        with open(label_path, 'r') as f:
            label_lines = f.read().splitlines()

    imsize = image.shape[:2]
    annotations = []
    for line in label_lines:
        parts = line.split(' ')
        class_id = int(parts[0])
        # Translate fraction coordinates to pixels (and make them int)
        points = [(int(imsize[1]*float(parts[i-1])), int(imsize[0]*float(parts[i]))) for i in range(2, len(parts), 2)]
        annotations.append((class_id, points))

    if verbose > 0:
        print(image_path)
    tiles, positions = tile_image(image, annotations, size=size, min_overlap=min_overlap, verbose=verbose)

    # Save tiles as separate images with location embedding.
    original_name = os.path.splitext(os.path.basename(image_path))[0]
    new_image_names = []
    for (img, labels), pos in zip(tiles, positions):
        y1, x1, y2, x2 = pos
        new_name = f"{y1}_{x1}_{y2}_{x2}_{original_name}"
        new_image_names.append(new_name)

        cv.imwrite(os.path.join(image_save_dir, f'{new_name}.png'), img)

        if label_save_dir:
            with open(os.path.join(label_save_dir, f'{new_name}.txt'), 'w') as f:
                for lbl, poly in labels:
                    frac_coords = [(point[0] / size[1], point[1] / size[0]) for point in poly]
                    label_text = f'{str(lbl)} {" ".join(f"{pt:.7f}" for coord in frac_coords for pt in coord)}'
                    f.write(label_text + '\n')

    # List of new images
    impaths = [f'{os.path.abspath(image_save_dir)}/{name}.png' for name in new_image_names]
    lblpaths = [f'{os.path.abspath(label_save_dir)}/{name}.txt' for name in new_image_names]
    return impaths, lblpaths


def split_image_datapoint_sam(image_path: str, mask_path: str = None,
                              size: tuple[int, int] = (1024, 1024), min_overlap: int = 256,
                              image_save_dir: str = None, mask_save_dir: str = None,
                              verbose: int = 0) -> tuple[list[str], list[str]]:
    """
    Reads an existing image and its labels in SA1B annotation_format and
    saves them as several overlapping mosaic images with new labels in SA1B annotation_format.

    Parameters
    ----------
    image_path : str
        Path to the source image to be split.
    mask_path : str, optional
        Path to the corresponding JSON annotation file.
        If None, it is inferred from the image path.
    size : tuple[int, int], optional
        The (height, width) of each output tile.
        Default is (1024, 1024).
    min_overlap : int, optional
        Minimum pixel overlap between adjacent tiles.
        Default is 256.
    image_save_dir : str, optional
        Directory to save the output image tiles.
        Default is the source image directory if None.
    mask_save_dir : str, optional
        Directory to save the new JSON annotation files.
        Default is the image_save_dir if None.
    verbose : int, optional
        Verbosity level for printing progress information.
        Default is 0.

    Returns
    -------
    Tuple[list[str], list[str]]: Paths to the newly created images and labels of newly created labels.
    """
    if mask_path is None:
        mask_path = os.path.splitext(image_path)[0] + ".json"

    image_save_dir = image_save_dir or os.path.dirname(image_path)
    mask_save_dir = mask_save_dir or image_save_dir
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    with open(mask_path, "r") as f:
        src_json = json.load(f)

    # Decode all source masks once
    decoded_masks = []
    for ann in src_json["annotations"]:
        rle = ann["segmentation"]
        decoded_masks.append(mask_utils.decode(rle))

    tiles, positions = tile_image(img,
                                  labels=None,
                                  size=size,
                                  min_overlap=min_overlap,
                                  verbose=verbose)

    orig_stem = os.path.splitext(os.path.basename(image_path))[0]
    img_id = orig_stem.split("_")[-1]
    new_img_id = int(img_id)*100

    out_img_paths, out_json_paths = [], []

    for i, ((tile_img, _), (y1, x1, y2, x2)) in enumerate(zip(tiles, positions)):
        curr_im_id = new_img_id + i

        tile_ann_entries = []
        for j, m in enumerate(decoded_masks):
            crop = m[y1:y2, x1:x2]
            if crop.any():
                tile_ann_entries.append(
                    create_annotation_entry(crop, curr_im_id + j, size)
                )

        # build image entry
        base_name = f"{y1}_{x1}_{y2}_{x2}_{orig_stem}"
        img_fname = f"{base_name}.png"
        img_entry = create_image_entry(base_name, size, curr_im_id)

        # save tile image
        cv.imwrite(os.path.join(image_save_dir, img_fname), tile_img)

        # save tile JSON
        tile_json = {"image": img_entry, "annotations": tile_ann_entries}
        json_path = os.path.join(mask_save_dir, f"{base_name}.json")
        with open(json_path, "w") as f:
            json.dump(tile_json, f)

        out_img_paths.append(os.path.abspath(os.path.join(image_save_dir, img_fname)))
        out_json_paths.append(os.path.abspath(json_path))

        if verbose:
            print(f"Saved {base_name}: {len(tile_ann_entries)} objects")

    return out_img_paths, out_json_paths


def split_images_dir(image_dir: str, label_dir: str, size: tuple[int, int] = (1024, 1024), min_overlap: int = 256,
                     image_save_dir: str = None, label_save_dir: str = None, annotation_format: str = 'yolo',
                     verbose: int = 0) -> tuple[list[str], list[str]]:
    """
    Applies `split_image_datapoint_yolo` or `split_image_datapoint_sam` to all image-label/mask pairs in the given path.

    Parameters
    ----------
    image_dir: str
        Directory containing input images.
    label_dir: str
        Directory containing label files or mask paths corresponding to the images.
    size : tuple[int, int], optional
        Size of each output tile image.
        Default is (1024, 1024).
    min_overlap: int, optional
        Minimum overlap between tiles.
        Default is 256.
    image_save_dir: str, optional
        Directory to save the output tile images.
        Default is image_dir if None.
    label_save_dir: str, optional
        Directory to save the output tile labels.
        Default is label_dir if None.
    annotation_format: str, optional
        Format of the labels ('sam' or 'yolo').
        Default is 'yolo'.
    verbose: int, optional
        Verbosity level.

    Returns
    -------
    Tuple[List[str], List[str]]
        Lists of file paths to newly created image and label/mask tiles.
    """
    new_images = []
    new_labels = []

    all_image_paths = [os.path.join(image_dir, name)
                       for name in os.listdir(image_dir)
                       if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

    for i, image_path in enumerate(all_image_paths):
        if verbose > 0:
            print(f'\r({i+1}/{len(all_image_paths)}) Processing {image_path}', end='', flush=True)

        filename = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_dir, f"{filename}.txt" if annotation_format == 'yolo' else f"{filename}.json")

        if annotation_format == 'yolo':
            image_paths, label_paths = split_image_datapoint_yolo(
                image_path=image_path,
                label_path=label_path,
                size=size,
                min_overlap=min_overlap,
                image_save_dir=image_save_dir,
                label_save_dir=label_save_dir,
                verbose=max(0, verbose-1))
        else:
            image_paths, label_paths = split_image_datapoint_sam(
                image_path=image_path,
                mask_path=label_path,
                size=size,
                min_overlap=min_overlap,
                image_save_dir=image_save_dir,
                mask_save_dir=label_save_dir,
                verbose=max(0, verbose-1))

        new_images.extend(image_paths)
        new_labels.extend(label_paths)

    return new_images, new_labels
