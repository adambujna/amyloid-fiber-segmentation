import numpy as np
import cv2 as cv
from ultralytics import YOLO

from src.tiling import tile_image
from src.tiling import merge_masks
from src.gui import plot_masks


class Yolo11SegmenterEM:
    def __init__(self, weights: str, image_size: tuple[int, int] = (1024, 1365),
                 input_size: tuple = (1024, 1024), min_overlap: int = None):
        self.tile_height = input_size[0]
        self.tile_width = input_size[1]
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.model = YOLO(weights)
        if min_overlap is None:
            min_overlap = int(min(self.tile_height, self.tile_width) * 0.2)
        self.min_overlap = min_overlap

    def predict(self, img: np.ndarray | str, plot: bool = False) -> list[np.ndarray]:
        if isinstance(img, str):
            img = cv.imread(img)
            if img is None:
                raise FileNotFoundError(f'Could not find image at {img}')

        img_resized = cv.resize(img, (self.image_width, self.image_height), interpolation=cv.INTER_AREA)

        # Tile image
        image_grid, pos_grid = self._get_image_grid(img_resized)

        # Predict and collect individual binary masks per tile
        mask_grid = [[[] for _ in row] for row in image_grid]

        for row_idx, row_tiles in enumerate(image_grid):
            for col_idx, tile in enumerate(row_tiles):
                results = self.model.predict(tile, verbose=False)[0]
                masks = []

                if results.masks is not None:
                    for mask in results.masks.data:
                        mask = mask.numpy().astype(np.uint8)
                        masks.append(cv.resize(mask, (self.tile_height, self.tile_width), interpolation=cv.INTER_MAX))

                mask_grid[row_idx][col_idx] = masks

        # Merge rows left-to-right
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

        if plot:
            plot_masks(final_masks, img)

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


if __name__ == '__main__':
    sample_image = '/Users/ab/Documents/AIDS/Bachelors Thesis/Code/data/images/image_0.png'
    model = Yolo11SegmenterEM(weights='/Users/ab/Documents/AIDS/Bachelors Thesis/Code/models/best.pt')

    output = model.predict('/Users/ab/Documents/AIDS/Bachelors Thesis/Code/data/annotated data sets/set '
                           '1/007_XG_medium fibril_ua__10kx_56.6px_100nm.tif', plot=True)
