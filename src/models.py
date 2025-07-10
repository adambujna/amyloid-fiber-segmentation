import os
import numpy as np
import cv2 as cv
from ultralytics import YOLO, SAM
from torch import cuda, mps
from abc import abstractmethod

from src.tiling import tile_image, merge_masks
from src.gui import plot_masks


class BaseSegmenter:
    """
    A base class for segmenting images by resizing/tiling them.
    This class implements the common methods for preprocessing, tiling, and merging.

    Does not perform predictions itself.
    """

    def __init__(self, image_size: tuple[int, int] = (1024, 1365),
                 input_size: tuple = (1024, 1024), min_overlap: int = None):
        self.tile_height = input_size[0]
        self.tile_width = input_size[1]
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.model = None

        if min_overlap is None:
            min_overlap = int(min(self.tile_height, self.tile_width) * 0.2)
        self.min_overlap = min_overlap

        if cuda.is_available():
            self.device = 'cuda'
        elif mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

    @abstractmethod
    def _predict_on_tile(self, tile: np.ndarray, **kwargs) -> list[np.ndarray]:
        """Implemented by the model-specific subclasses."""
        raise TypeError("This method is specific to `Yolo11SegmenterEM` or `SAM2SegmenterEM` and cannot "
                        "be called from `BaseSegmenter`.")

    def __call__(self, img: np.ndarray | str, plot: bool = False, save: bool = False, **kwargs) -> list[np.ndarray]:
        imname = None
        if isinstance(img, str):
            imname = img
            img = cv.imread(img)
            if img is None:
                raise FileNotFoundError(f'Could not find image at {imname}')

        img_resized = cv.resize(img, (self.image_width, self.image_height), interpolation=cv.INTER_AREA)

        # Tile image
        image_grid, pos_grid = self._get_image_grid(img_resized)

        # Predict on each tile individually
        mask_grid = [[[] for _ in row] for row in image_grid]

        for row_idx, row_tiles in enumerate(image_grid):
            for col_idx, tile in enumerate(row_tiles):
                masks = self._predict_on_tile(tile, **kwargs)
                mask_grid[row_idx][col_idx] = masks

        # Merge each row left-to-right
        merged_rows = []
        for row_masks, row_positions in zip(mask_grid, pos_grid):
            merged_m, merged_p = row_masks[0], row_positions[0]
            for next_m, next_p in zip(row_masks[1:], row_positions[1:]):
                merged_m, merged_p = self._merge_pairwise(merged_m, merged_p, next_m, next_p)
            merged_rows.append((merged_m, merged_p))

        # Merge rows top-to-bottom
        final_masks, final_pos = merged_rows[0]
        for next_masks, next_pos in merged_rows[1:]:
            final_masks, final_pos = self._merge_pairwise(final_masks, final_pos, next_masks, next_pos)

        # Plot
        if plot:
            if imname is None:
                imname = 'prediction.png'
            plot_masks(final_masks, img, save, os.path.basename(imname))

        return final_masks

    def _get_image_grid(self, img: np.ndarray):
        """Tiles image into overlapping tiles."""
        tiles, positions = tile_image(img, size=(self.tile_height, self.tile_width), min_overlap=self.min_overlap)
        images = [img for (img, annot) in tiles]

        n_height = int(np.ceil((self.image_height - self.tile_height) / (self.image_height - self.min_overlap)) + 1)

        image_grid = np.array(images).reshape((n_height, -1, self.tile_height, self.tile_width, 3))
        pos_grid = np.array(positions).reshape((n_height, -1, 4))

        return image_grid, pos_grid

    @staticmethod
    def _merge_pairwise(m1, p1, m2, p2):
        merged = merge_masks(p1, p2, m1, m2)
        merged_pos = (
            min(p1[0], p2[0]), min(p1[1], p2[1]),
            max(p1[2], p2[2]), max(p1[3], p2[3])
        )
        return merged, merged_pos


class Yolo11SegmenterEM(BaseSegmenter):
    """YOLO Segmenter."""
    def __init__(self, weights: str, **kwargs):
        super().__init__(**kwargs)
        self.model = YOLO(weights)

    def _predict_on_tile(self, tile: np.ndarray, **kwargs) -> list[np.ndarray]:
        results = self.model.predict(tile,
                                     conf=0.15,
                                     device=self.device,
                                     iou=0.2,
                                     verbose=False)[0]
        masks = []

        if results.masks is not None:
            for mask in results.masks.data:
                mask = mask.cpu().numpy().astype(np.uint8)
                masks.append(cv.resize(mask, (self.tile_height, self.tile_width), interpolation=cv.INTER_MAX))

        return masks


class SAM2SegmenterEM(BaseSegmenter):
    """SAM segmenter."""

    def __init__(self, weights: str, **kwargs):
        super().__init__(**kwargs)
        self.model = SAM(weights)

    def _predict_on_tile(self, tile: np.ndarray, **kwargs) -> list[np.ndarray]:
        points = kwargs.get('points')  # SAM needs points, get it from kwargs
        results = self.model.predict(tile,
                                     conf=0.5,
                                     iou=0.7,
                                     device=self.device,
                                     verbose=False,
                                     points=points)[0]
        masks = []

        if results.masks is not None:
            for mask in results.masks.data:
                mask = mask.cpu().numpy().astype(np.uint8)
                masks.append(cv.resize(mask, (self.tile_height, self.tile_width), interpolation=cv.INTER_MAX))

        return masks


if __name__ == '__main__':
    # Imports oinly required for SAM2
    from src.annotation_utils import load_gt_masks_sam, sample_points

    sample_image = '../data/datasets/real_dataset/images_raw/test/015_XG_short fibril_ua__10kx_56.6px_100nm.png_2002.png'
    sample_labels = '../data/datasets/real_dataset/images_raw/test/015_XG_short fibril_ua__10kx_56.6px_100nm.png_2002.json'

    model = Yolo11SegmenterEM(weights='../model_checkpoints/yolo11large-finetuned.pt')

    # Only required for SAM2
    gt_masks = load_gt_masks_sam(sample_labels)
    points = []
    for m in [cv.resize(m, (1365, 1024)) for m in gt_masks]:
        points.append(sample_points(m, 1))

    # Passing `points` arg also only required for SAM
    output = model(sample_image, plot=True, save=False, points=np.array(points))
