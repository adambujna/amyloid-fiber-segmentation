import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def plot_masks(masks: list[np.ndarray], background: str | np.ndarray,
               save: bool = False, figname: str = 'masks.png') -> None:
    """
    Plots binary masks overlaid on top of an image.

    Parameters
    ----------
    masks : list[np.ndarray]
        The list of binary masks.
    background : str | np.ndarray
        The background image as a NumPy array or path to the background image.
    save : bool, optional
        Whether to save the figure as a png file.
        Default is False.
    figname : str, optional
        The saved plot's image file name.
        Default is 'masks.png'.
    """
    if isinstance(background, str):
        imname = background
        background = cv.imread(background, cv.IMREAD_GRAYSCALE)
        if background is None:
            raise FileNotFoundError(f"Could not load background image from {imname}")
    else:
        background = background.copy()[:, :, 0]

    H, W = background.shape[:2]
    combined = np.zeros((H, W), dtype=np.uint16)

    # Centroids for writing mask numbers to distinguish them
    centroids = []
    for idx, m in enumerate(masks, start=1):
        m = cv.resize(m, (W, H), interpolation=cv.INTER_NEAREST)
        combined = np.where(m > 0, idx, combined)

        ys, xs = np.where(m > 0)
        if len(xs):
            xc, yc = xs.mean(), ys.mean()
            centroids.append((xc, yc))
        else:
            centroids.append((None, None))

    # Plot
    plt.figure(figsize=(12, 8))
    plt.imshow(background, cmap='gray')     # Plot background image
    plt.imshow(combined, cmap='nipy_spectral', alpha=0.3, interpolation='none')     # Plot masks

    # Plot a number next to each mask
    for idx, (xc, yc) in enumerate(centroids, start=1):
        if xc is None:
            continue
        plt.text(
            xc + 10, yc + 10, str(idx),
            color="white", fontsize=8, weight="light",
            ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")]
        )

    plt.title("Merged Mask IDs")
    plt.tight_layout()
    if save:
        plt.savefig(figname,
                    dpi=300)
    plt.show()
