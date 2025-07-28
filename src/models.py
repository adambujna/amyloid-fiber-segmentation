import os
import numpy as np
import cv2 as cv
from ultralytics import YOLO, SAM
from torch import cuda, mps
from abc import abstractmethod

from src.tiling import tile_image, merge_masks, BoundingBox
from src.gui import plot_masks


class BaseSegmenter:
    """
    A base class for segmenting images by resizing and tiling them, and then merging results.
    This class provides a shared framework for handling large images that do not match or
    exceed the input size of a model.

    Does not perform predictions itself.

    Attributes
    ----------
    tile_height : int
        The height of each tile.
    tile_width : int
        The width of each tile.
    image_height : int
        The target height to which input images are resized.
    image_width : int
        The target width to which input images are resized.
    model : object
        The segmentation model, loaded by a subclass.
    min_overlap : int
        The minimum overlap in pixels between adjacent tiles.
    device : str
        The pytorch device ('cuda', 'mps', or 'cpu') selected for inference.
    """
    def __init__(self, image_size: tuple[int, int] = (1024, 1365),
                 input_size: tuple = (1024, 1024), min_overlap: int = None):
        """
        Initializes the BaseSegmenter with configuration for tiling and processing.

        Parameters
        ----------
        image_size : tuple[int, int], optional
            The (height, width) to which the full input image will be resized before tiling.
            Default is (1024, 1365).
        input_size : tuple, optional
            The (height, width) of the individual tiles that the model expects as input.
            Defaults to (1024, 1024).
        min_overlap : int, optional
            The minimum number of pixels of overlap between adjacent tiles.
            If None, defaults to 20% of the smaller tile dimension.
        """
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
        """
        Implemented by the model-specific subclasses.

        Parameters
        ----------
        tile : np.ndarray
            The image tile to be processed by the model.
        **kwargs
            Additional model-specific arguments (e.g., prompt points for SAM).

        Returns
        -------
        list[np.ndarray]: A list of binary masks.
        """
        raise TypeError("This method is specific to `Yolo11SegmenterEM` or `SAM2SegmenterEM` and cannot "
                        "be called from `BaseSegmenter`.")

    def __call__(self, img: np.ndarray | str, plot: bool = False, save: bool = False, **kwargs) -> list[np.ndarray]:
        """
        Performs end-to-end segmentation on a full image.

        This method executes the prediction pipeline of:
        1. Loading and resizing the image.
        2. Splitting it into a grid of overlapping tiles.
        3. Running inference on each tile.
        4. Merging the resulting masks.

        Parameters
        ----------
        img : np.ndarray | str
            The input image as a NumPy array or a string path to the image file.
        plot : bool, optional
            If True, displays the final segmentation masks overlaid on the original
            image.
            Default is False.
        save : bool, optional
            If True and `plot` is also True, saves the prediction plot image to a file.
            Default is False.
        **kwargs
            Additional arguments to be passed to the `_predict_on_tile` method.

        Returns
        -------
        list[np.ndarray]: A list of the final, merged binary segmentation masks for the entire image.
        """
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

    def _get_image_grid(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Tiles image into a 2D grid of overlapping tiles.

        Parameters
        ----------
        img : np.ndarray
            The resized image to be tiled.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. A 2D NumPy array of image tiles.
            2. A 2D NumPy array of the corresponding tile positions (x1, y1, x2, y2).
        """
        tiles, positions = tile_image(img, size=(self.tile_height, self.tile_width), min_overlap=self.min_overlap)
        images = [img for (img, annot) in tiles]

        n_height = int(np.ceil((self.image_height - self.tile_height) / (self.image_height - self.min_overlap)) + 1)

        image_grid = np.array(images).reshape((n_height, -1, self.tile_height, self.tile_width, 3))
        pos_grid = np.array(positions).reshape((n_height, -1, 4))

        return image_grid, pos_grid

    @staticmethod
    def _merge_pairwise(masks_1: list[np.ndarray], p1: BoundingBox,
                        masks_2: list[np.ndarray], p2: BoundingBox) -> tuple[list[np.ndarray], BoundingBox]:
        """
        Merges the segmentation masks from two adjacent, overlapping tiles.

        Parameters
        ----------
        masks_1 : list[np.ndarray]
            List of masks from the first tile.
        p1 : tuple[x1, y1, x2, y2]
            Position of the first tile (x1, y1, x2, y2).
        masks_2 : list[np.ndarray]
            List of masks from the second tile.
        p2 : tuple[x1, y1, x2, y2]
            Position of the second tile.

        Returns
        -------
        tuple[list[np.ndarray], tuple[x1, y1, x2, y2]]:
            Tuple (list of merged masks, bounding box of merged tiles)
        """
        merged = merge_masks(p1, p2, masks_1, masks_2)
        merged_pos = (
            min(p1[0], p2[0]), min(p1[1], p2[1]),
            max(p1[2], p2[2]), max(p1[3], p2[3])
        )
        return merged, merged_pos


class Yolo11SegmenterEM(BaseSegmenter):
    """
    A segmenter for electron microscopy images using a YOLOv11 model.

    This class inherits from `BaseSegmenter` and implements the prediction logic for the YOLO model.
    """
    def __init__(self, weights: str, **kwargs):
        """
        Initializes the YOLOv11 segmenter.

        Parameters
        ----------
        weights : str
            Path to the YOLO model's pre-trained weights file (.pt).
        **kwargs
            Arguments passed to the `BaseSegmenter` constructor.
        """
        super().__init__(**kwargs)
        self.model = YOLO(weights)

    def _predict_on_tile(self, tile: np.ndarray, **kwargs) -> list[np.ndarray]:
        """
        Runs YOLOv11 inference on a single image tile.

        Parameters
        ----------
        tile : np.ndarray
            The image to be processed.
        **kwargs
            Additional arguments (unused).

        Returns
        -------
        list[np.ndarray]: A list of predicted binary masks.
        """
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
    """
    A segmenter for electron microscopy images using a SAM v2.1 model.

    This class inherits from `BaseSegmenter` and implements the prediction logic for the SAM model.
    The SAM model can take prompts (points/masks/bboxes/labels) for guided segmentation.
    """
    def __init__(self, weights: str, **kwargs):
        """
        Initializes the SAMv2.1 segmenter.

        Parameters
        ----------
        weights : str
            Path to the SAM model's pre-trained weights file (.pt).
        **kwargs
            Arguments passed to the `BaseSegmenter` constructor.
        """
        super().__init__(**kwargs)
        self.model = SAM(weights)

    def _predict_on_tile(self, tile: np.ndarray, **kwargs) -> list[np.ndarray]:
        """
        Runs SAMv2.1 inference on a single image tile.

        Parameters
        ----------
        tile : np.ndarray
            The image to be processed.
        **kwargs
            Additional arguments, expected to contain 'points', 'masks', 'bboxes', 'labels' for SAM prompts.
            Prompting is optional.

        Returns
        -------
        list[np.ndarray]: A list of predicted binary masks.
        """
        results = self.model.predict(tile,
                                     conf=0.5,
                                     iou=0.7,
                                     device=self.device,
                                     verbose=False,
                                     points=kwargs.get('points'),
                                     bboxes=kwargs.get('bboxes'),
                                     masks=kwargs.get('masks'),
                                     labels=kwargs.get('labels')
                                     )[0]
        masks = []

        if results.masks is not None:
            for mask in results.masks.data:
                mask = mask.cpu().numpy().astype(np.uint8)
                masks.append(cv.resize(mask, (self.tile_height, self.tile_width), interpolation=cv.INTER_MAX))

        return masks


if __name__ == '__main__':
    # Imports only required for SAM2
    from src.annotation_utils import load_gt_masks_sam, sample_points

    sample_image = '../data/datasets/real_dataset/images_raw/test/015_XG_short fibril_ua__10kx_56.6px_100nm.png_2002.png'
    sample_labels = '../data/datasets/real_dataset/images_raw/test/015_XG_short fibril_ua__10kx_56.6px_100nm.png_2002.json'

    model = Yolo11SegmenterEM(weights='../model_checkpoints/yolo11large-finetuned.pt')

    # Only required for SAM2
    gt_masks = load_gt_masks_sam(sample_labels)
    points_prompt = []
    for m in [cv.resize(m, (1365, 1024)) for m in gt_masks]:
        points_prompt.append(sample_points(m, n_points=1))

    # Passing `points` arg also only required for SAM
    _ = model(sample_image, plot=True, save=False, points=np.array(points_prompt))
