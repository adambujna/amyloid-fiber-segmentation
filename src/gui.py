import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def plot_masks(masks: list[np.ndarray], background: str | np.ndarray):
    if isinstance(background, str):
        background = cv.imread(background, cv.IMREAD_GRAYSCALE)
        if background is None:
            raise FileNotFoundError(f"Could not load background image from {background}")
    else:
        background = background.copy()[:, :, 0]

    combined = np.zeros_like(background, dtype=np.uint16)
    for i, mask in enumerate(masks):
        mask = cv.resize(mask, background.shape[:2][::-1], interpolation=cv.INTER_LINEAR)
        combined = np.where(mask > 0, i + 1, combined)

    plt.figure(figsize=(12, 8))
    plt.imshow(background, cmap='gray')
    plt.imshow(combined, cmap='nipy_spectral', alpha=0.5, interpolation='none')
    plt.title("Merged Mask IDs")
    plt.show()
