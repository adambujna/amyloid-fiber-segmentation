import os
import cv2 as cv
import numpy as np
from random import randint
from typing import Tuple
from matplotlib import pyplot as plt
from src.data_utils import save_image, save_label_yolo, save_label_sam
from src.image_processing import add_noise

# Type
FloatRange = Tuple[float, float]


def get_noise_patch(noise_path: str, start_coords: tuple[int, int] = None,
                    size: tuple[int, int] = (153, 206)) -> np.ndarray:
    """
    Loads a large noise image and extracts a smaller patch from it.

    This function is used to get a unique noise pattern for each synthetic image
    without needing to generate new noise every time.

    Parameters
    ----------
    noise_path : str
        The file path to the large, pre-generated noise image.
    start_coords : tuple[int, int], optional
        The (y, x) top-left coordinates for cropping the patch.
        If None, a random position is chosen.
        Default is None.
    size : tuple[int, int], optional
        The (height, width) of the patch to extract.
        Noise looks different at different scales.
        Default is (153, 206).

    Returns
    -------
    np.ndarray: The extracted and normalized noise patch as a NumPy array.
    """
    noise = cv.imread(noise_path, cv.IMREAD_GRAYSCALE)
    if noise is None:
        raise ValueError("Noise image could not be loaded.")

    h, w = noise.shape
    if start_coords is None:
        start_coords = (np.random.randint(0, h - size[0]), np.random.randint(0, w - size[1]))

    output = noise[start_coords[0]:start_coords[0] + size[0], start_coords[1]:start_coords[1] + size[1]]
    output -= np.min(output)
    output *= (255 // np.max(output))

    return output


def noise_based_sample(noise: np.ndarray, clustering: float = 2, num_samples: int = 1,
                       jitter: float = 0.01, direction_sampling_window: int = 23) -> tuple[np.ndarray, list[int]]:
    """
    Samples points and directions from a noise map.

    This method samples point locations from the noise map, treating pixel
    intensity as a probability distribution.
    For each point, it determines a direction by sampling from a small window around the point,
    where pixels with higher intensity are more likely to be the target direction.

    Parameters
    ----------
    noise : np.ndarray
        The 2D noise map used as a probability distribution for sampling.
    clustering : float, optional
        An exponent applied to the noise map to increase contrast,
        making sampling from high-intensity peaks more likely.
        Default is 2.
    num_samples : int, optional
        The number of points to sample.
        Default is 1.
    jitter : float, optional
        The maximum random offset (relative to image size)
        to apply to the final normalized coordinates to add positional variance.
        Default is 0.01.
    direction_sampling_window : int, optional
        The size in pixels of the square window around each point used to sample its direction.
        Default is 23.

    Returns
    -------
    tuple[np.ndarray, list[int]]: A tuple containing the normalized [y, x] coordinates and directions.
    """
    if num_samples <= 0:
        return np.array([]), []
    # Choose random indexes based on noise as probability distribution
    p = noise.astype(np.float64).clip(30, 255) ** clustering
    flat = p.flatten()
    flat = flat / flat.sum()
    samples = np.random.choice(a=flat.size, p=flat, size=num_samples, replace=True)
    samples = np.array([np.unravel_index(s, p.shape) for s in samples], dtype=np.float64)

    # Sampling starting directions from windows around each point
    directions = []

    half_win = direction_sampling_window // 2
    for y, x in samples:
        y, x = int(y), int(x)
        y0 = max(0, y - half_win)
        y1 = min(p.shape[0], y + half_win + 1)
        x0 = max(0, x - half_win)
        x1 = min(p.shape[1], x + half_win + 1)

        local_patch = p[y0:y1, x0:x1]
        local_flat = local_patch.flatten()
        local_flat = local_flat / local_flat.sum()

        grid_y, grid_x = np.meshgrid(
            np.linspace(-1, 1, local_patch.shape[0]),
            np.linspace(-1, 1, local_patch.shape[1]),
            indexing='ij'
        )
        # Angles of pixels in the local window
        local_angles = (np.degrees(np.arctan2(grid_y, grid_x)) + 360) % 360
        angle_choices = local_angles.flatten()
        # Pick pixels based on noise and sample its angle
        sampled_angle = np.random.choice(angle_choices, p=local_flat)
        directions.append(int(sampled_angle))

    # Normalize to [0, 1] coordinate system
    samples[:, 0] /= p.shape[0] - 1
    samples[:, 1] /= p.shape[1] - 1

    if jitter is not None and jitter > 0:  # Randomly jitter locations
        x_offsets = (np.random.random(num_samples) * jitter) - jitter / 2
        y_offsets = (np.random.random(num_samples) * jitter) - jitter / 2

        samples[:, 0] += y_offsets
        samples[:, 1] += x_offsets

    return samples.clip(0, 1), directions


def noise_based_sample_grad(noise: np.ndarray, clustering: float = 2, num_samples: int = 1,
                            jitter: float = 0.01) -> tuple[np.ndarray, list[int]]:
    """
    Samples points from a noise map based on its intensity
    and directions from a noise map based on its gradient.

    This method samples point locations similarly to `noise_based_sample`.
    However, it determines the direction for each point by calculating the
    gradient of the noise map at that location.
    The final direction is set perpendicular to the gradient,
    causing samples to align along the "ridges" or contours of the noise map.

    Parameters
    ----------
    noise : np.ndarray
        The 2D noise map used as a probability distribution for sampling.
    clustering : float, optional
        An exponent applied to the noise map to increase contrast,
        making sampling from high-intensity peaks more likely.
        Default is 2.
    num_samples : int, optional
        The number of points to sample.
        Default is 1.
    jitter : float, optional
        The maximum random offset (relative to image size)
        to apply to the final normalized coordinates to add positional variance.
        Default is 0.01.

    Returns
    -------
    tuple[np.ndarray, list[int]]: A tuple containing the normalized [y, x] coordinates and directions.
    """
    if num_samples <= 0:
        return [], []
    # Choose random indexes based on noise as probability distribution
    p = noise.astype(np.float64).clip(30, 255) ** clustering
    flat = p.flatten()
    flat = flat / flat.sum()
    samples = np.random.choice(a=flat.size, p=flat, size=num_samples, replace=True)
    samples = np.array([np.unravel_index(s, p.shape) for s in samples], dtype=np.float64)

    # Sampling starting directions from windows around each point
    directions = []
    # Get gradient maps
    gy, gx = np.gradient(p)

    for y, x in samples:
        y, x = int(y), int(x)
        dx = gx[y, x]
        dy = gy[y, x]
        # Angle perpendicular to gradient
        angle = np.rad2deg(np.arctan2(dy, dx)) + (90 if np.random.random() < 0.5 else 270)
        angle += np.random.normal(loc=0, scale=15)
        directions.append(int(angle % 360))

    # Normalize to [0, 1] coordinate system
    samples[:, 0] /= p.shape[0] - 1
    samples[:, 1] /= p.shape[1] - 1

    if jitter is not None and jitter > 0:  # Randomly jitter locations
        x_offsets = (np.random.random(num_samples) * jitter) - jitter / 2
        y_offsets = (np.random.random(num_samples) * jitter) - jitter / 2

        samples[:, 0] += y_offsets
        samples[:, 1] += x_offsets

    return samples.clip(0, 1), directions


def noise_based_sample_combined(noise: np.ndarray, clustering: float = 2,
                                num_samples: int = 1, jitter: float = 0.01,
                                direction_sampling_window: int = 23) -> tuple[np.ndarray, list[int]]:
    """
    Samples points from a noise map based on its intensity
    and directions from a noise map based on a combination of its intensity and gradient.

    This method calculates the sampled direction based on both the local intensity distribution
    (like `noise_based_sample`)
    and the gradient (like `noise_based_sample_grad`).
    It then computes a weighted average of these two directions, where the weight
    is determined by the local gradient's strength.

    Parameters
    ----------
    noise : np.ndarray
        The 2D noise map used as a probability distribution for sampling.
    clustering : float, optional
        An exponent applied to the noise map to increase contrast,
        making sampling from high-intensity peaks more likely.
        Default is 2.
    num_samples : int, optional
        The number of points to sample.
        Default is 1.
    jitter : float, optional
        The maximum random offset (relative to image size)
        to apply to the final normalized coordinates to add positional variance.
        Default is 0.01.
    direction_sampling_window : int, optional
        The size in pixels of the square window around each point used to sample its direction.
        Default is 23.

    Returns
    -------
    tuple[np.ndarray, list[int]]: A tuple containing the normalized [y, x] coordinates and directions.
    """
    if num_samples <= 0:
        return [], []
    # Choose random indexes based on noise as probability distribution
    p = noise.astype(np.float64).clip(30, 255) ** clustering
    flat = p.flatten()
    flat = flat / flat.sum()
    samples = np.random.choice(a=flat.size, p=flat, size=num_samples, replace=True)
    samples = np.array([np.unravel_index(s, p.shape) for s in samples], dtype=np.float64)

    # Sampling starting directions from windows around each point
    directions = []
    gy, gx = np.gradient(noise)

    half_win = direction_sampling_window // 2
    for y, x in samples:
        y, x = int(y), int(x)
        y0 = max(0, y - half_win)
        y1 = min(p.shape[0], y + half_win + 1)
        x0 = max(0, x - half_win)
        x1 = min(p.shape[1], x + half_win + 1)

        local_patch = p[y0:y1, x0:x1]
        local_flat = local_patch.flatten()
        local_flat = local_flat / local_flat.sum()

        grid_y, grid_x = np.meshgrid(
            np.linspace(-1, 1, local_patch.shape[0]),
            np.linspace(-1, 1, local_patch.shape[1]),
            indexing='ij'
        )
        local_angles = np.arctan2(grid_y, grid_x)
        angle_choices = local_angles.flatten()

        # Sample based on the local window
        sampled_angle = np.random.choice(angle_choices, p=local_flat)

        # Sample based on the gradient
        dx = gx[y, x]
        dy = gy[y, x]
        grad_angle = np.arctan2(dy, dx) + (0.5 * np.pi if np.random.random() < 0.5 else 1.5 * np.pi)

        # Create weighted average of gradient and local window choice
        # Gradient strength determines gradient weight in direction contribution
        grad_strength = np.hypot(dx, dy)

        grad_weight = grad_strength / (grad_strength + 1)
        noise_weight = 1.0 - grad_weight

        # Combine angles based on weights
        x = grad_weight * np.cos(grad_angle) + noise_weight * np.cos(sampled_angle)
        y = grad_weight * np.sin(grad_angle) + noise_weight * np.sin(sampled_angle)
        final_angle = (np.rad2deg(np.arctan2(y, x)) + 360) % 360

        directions.append(int(final_angle))

    # Normalize to [0, 1] coordinate system
    samples[:, 0] /= p.shape[0] - 1
    samples[:, 1] /= p.shape[1] - 1

    if jitter is not None and jitter > 0:  # Randomly jitter locations
        x_offsets = (np.random.random(num_samples) * jitter) - jitter / 2
        y_offsets = (np.random.random(num_samples) * jitter) - jitter / 2

        samples[:, 0] += y_offsets
        samples[:, 1] += x_offsets

    return samples.clip(0, 1), directions


def get_fibre(image: np.ndarray, start_x: int, start_y: int,
              start_direction: int) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """
    Generates the geometry of a single synthetic fiber as a binary masks.

    The fiber is rendered as a curved segment of an ellipse with randomized
    axes and length.
    The function creates two boolean masks: one for the
    main fiber body and another for the lighter inner core.

    The function also calculates the offset of the center needed to align the fiber
    with a starting direction and location and calculates the end location and direction
    to allow for chaining of several fiber segments if desired.

    Parameters
    ----------
    image : np.ndarray
        A template image used to determine the output mask dimensions.
    start_x : int
        The x-coordinate for the start of the fiber.
    start_y : int
        The y-coordinate for the start of the fiber.
    start_direction : int
        The initial direction of the fiber in degrees.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, tuple[float, float, float]]
        A tuple containing:
        - The boolean mask for the outer fiber.
        - The boolean mask for the inner core.
        - A tuple with the (x, y, angle) of the fiber's endpoint.
    """
    a, b = [randint(50, 2000), randint(50, 2000)]  # Random axes
    angle = randint(1, min(180, int(60000 / max(a, b))))  # Limit fiber length
    start_direction = start_direction - 90  # OpenCV starts writing at 90 deg angles from starting point

    # Compute the fiber's center based on the direction and desired start of fiber
    x = int(start_x - (np.cos(np.deg2rad(start_direction)) * a))
    y = int(start_y - (np.sin(np.deg2rad(start_direction)) * a))

    # Compute fibers ending location
    rotation_angle = np.deg2rad(start_direction)
    theta = np.deg2rad(angle)
    x_rotated = (a * np.cos(theta)) * np.cos(rotation_angle) - (b * np.sin(theta)) * np.sin(rotation_angle)
    y_rotated = (a * np.cos(theta)) * np.sin(rotation_angle) + (b * np.sin(theta)) * np.cos(rotation_angle)
    x_end = int(x + x_rotated)
    y_end = int(y + y_rotated)

    width = randint(9, 11)
    inner_width = int(width / (randint(3, 8) / 2))

    # Create empty masks
    mask = np.zeros_like(image, dtype=np.uint8)
    inner_mask = np.zeros_like(image, dtype=np.uint8)

    # Draw the fiber and its inner core
    ellipse_args = {'axes': (a, b), 'angle': start_direction, 'startAngle': 0, 'endAngle': angle, 'color': 1}
    cv.ellipse(mask, center=(x, y), thickness=width, **ellipse_args)
    cv.ellipse(inner_mask, center=(x, y), thickness=inner_width, **ellipse_args)

    # cv.circle(mask, center=(x_end, y_end), radius=20, color=1, thickness=2)
    # cv.circle(inner_mask, center=(start_x, start_y), radius=20, color=1, thickness=2)

    return mask.astype(bool), inner_mask.astype(bool), (x_end, y_end, start_direction + angle)


def generate_fiber_image(image_size: tuple[int, int] = (2472, 3296),
                         num_fibers: int = max(0, int(np.random.normal(loc=100, scale=20))),  # Random fiber count,
                         snr: float = 4.5, background_color: int = 135, fiber_color: int = None,
                         clustering: float = 2.0,
                         save: bool = False, save_format: str = 'yolo',
                         image_name: str = None,
                         save_dir: str = None, label_dir: str = None,
                         gui: bool = False) -> np.ndarray:
    """
    Generates an image containing multiple simulated fibers along with annotations.

    Parameters
    ----------
    image_size : tuple[int, int], optional
        The (height, width) of the output image.
        Default is (2472, 3296).
    num_fibers : int, optional
        The number of fibers to generate in the image.
    snr : float, optional
        The target Signal-to-Noise Ratio for the final image.
        Default is 4.5.
    background_color : int, optional
        The base grayscale intensity of the image background.
        Default is 135.
    fiber_color : int, optional
        The base intensity for fibers.
        If None, it's derived from the background color.
        Default is None.
    clustering : float, optional
        The clustering factor passed to the noise sampling function.
        Default is 2.0.
    save : bool, optional
        If True, saves the image and its annotations.
        Default is False.
    save_format : str, optional
        The format for saving labels ('yolo', 'sam', or 'all').
        Default is 'yolo'.
    image_name : str, optional
        The base name for the output files.
        Required if `save` is True.
    save_dir : str, optional
        The directory to save the output image.
        Required if `save` is True.
    label_dir : str, optional
        The directory to save the output annotations.
        Required if `save` is True.
    gui : bool, optional
        If True, displays the generated image in a Matplotlib window.
        Default is False.

    Returns
    -------
    np.ndarray
        The generated synthetic image as a NumPy array.
    """
    if fiber_color is None:
        fiber_color = background_color / 1.53

    # Create image of desired size with specified background color
    new_img = np.full(shape=image_size, dtype=np.uint8, fill_value=background_color)

    # Calculate noise strength
    # SNR = signal / noise -> noise = signal / SNR
    bg_sg_r = background_color / fiber_color
    signal = abs(fiber_color - background_color)
    noise_strength = signal / snr

    # Contours and mask collectors for annotation generation
    contours = []
    masks = []

    # Get fiber starting locations with noise sampling
    noise = get_noise_patch('../data/images/noise/cellular-noise-1.png')
    fiber_locs, dirs = noise_based_sample_combined(noise=noise,
                                                   clustering=clustering,
                                                   num_samples=num_fibers,
                                                   jitter=0.05)
    fiber_locs = [(y * image_size[0], x * image_size[1]) for y, x in fiber_locs]

    for (start_y, start_x), start_direction in zip(fiber_locs, dirs):  # Make fibers

        # Generate fiber masks
        fiber_mask, inner_mask, _ = get_fibre(new_img, start_x, start_y, start_direction)

        # Apply fiber patterns to the image with color jittering so fibers are not all equal
        new_img[fiber_mask] = (new_img[fiber_mask] * (1 / max(1.1, bg_sg_r + 0.05 * np.random.random()))).astype(
            np.uint8)
        new_img[inner_mask] = (new_img[inner_mask] * (1.2 + 0.05 * np.random.random())).astype(np.uint8)

        if save:
            if save_format not in ['yolo', 'sam', 'all']:
                raise ValueError('save_format must be yolo or sam or all')
            if save_format == 'yolo' or save_format == 'all':
                # Extract contours
                contours_data, _ = cv.findContours(fiber_mask.astype(np.uint8),
                                                   mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
                for contour in contours_data:
                    epsilon = 0.003 * cv.arcLength(contour, closed=True)
                    approx = cv.approxPolyDP(contour, epsilon=epsilon, closed=True)
                    contours.append(approx)
            if save_format == 'sam' or save_format == 'all':
                masks.append(fiber_mask)

    # Apply noise and blur
    new_img = add_noise(new_img, noise_strength=noise_strength)

    if save:
        if save_dir is None or label_dir is None:
            raise ValueError('save_dir and label_dir must be specified if saving images.')
        if image_name is None:
            raise ValueError('image_name must be specified if saving image.')
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        save_image(image_name, new_img, save_dir)
        if save_format not in ['yolo', 'sam', 'all']:
            raise ValueError('save_format must be yolo or sam or all')
        if save_format == 'yolo' or save_format == 'all':
            save_label_yolo(image_name, contours, image_size, label_dir)
        if save_format == 'sam' or save_format == 'all':
            if not masks:
                masks = [np.zeros_like(new_img, dtype=np.uint8)]
            save_label_sam(image_name, masks, label_dir)

    if gui:
        fig, ax = plt.subplots(figsize=(12, 8), layout='compressed')
        ax.imshow(new_img, cmap='gray', vmin=0, vmax=255)
        plt.suptitle(f'Fibers: {num_fibers}')
        plt.show()

    return new_img


def generate_multiple_images(num_images: int = 1,
                             start_index: int = 0,
                             image_size: tuple[int, int] = (2472, 3296),
                             snr_range: tuple[float, float] = (1.5, 12.0),
                             clustering_range: tuple[int, int] = (0, 10),
                             bg_color_range: tuple[int, int] = (30, 220),
                             bg_sn_ratio: tuple[float, float] = None,
                             fibers: tuple[int, int] = (100, 20),
                             save: bool = False, save_format: str = 'yolo',
                             image_list_file: str = None, save_dir: str = None, label_dir: str = None,
                             gui: bool = False):
    """
    Generates multiple synthetic fiber images with varied properties.

    This function acts as a wrapper around `generate_fiber_image`.
    It loops a specified number of times, and in each iteration, it randomizes
    the image parameters (like SNR, clustering, fiber count) based on the
    provided ranges before calling the single image generator.

    It also manages the creation of a text file listing the paths to all generated images.

    Parameters
    ----------
    num_images : int, optional
        The total number of images to generate.
        Default is 1.
    start_index : int, optional
        The starting index for numbering the output image files.
        Default is 0.
    image_size : tuple[int, int], optional
        The (height, width) of all generated images.
    snr_range : tuple[float, float], optional
        The (min, max) range from which to sample the SNR for each image.
    clustering_range : tuple[int, int], optional
        The (min, max) range for the clustering parameter.
    bg_color_range : tuple[int, int], optional
        The (min, max) range for the background color.
    bg_sn_ratio : tuple[float, float], optional
        The (min, max) range for the background-to-signal intensity ratio.
    fibers : tuple[int, int], optional
        A tuple of (mean, std_dev) for a Normal distribution from which the
        number of fibers for each image is sampled.
    save : bool, optional
        If True, saves all generated images and labels.
    save_format : str, optional
        The annotation format ('yolo', 'sam', or 'all').
    image_list_file : str, optional
        Path to the output .txt file that will list all generated image paths.
        Required if `save` is True.
    save_dir : str, optional
        Directory to save the output images.
        Required if `save` is True.
    label_dir : str, optional
        Directory to save the output labels.
        Required if `save` is True.
    gui : bool, optional
        If True, images as they are being generated in a live window.
    """
    image_names = []

    ax = None
    if gui:  # Placeholder plot
        fig, ax = plt.subplots(figsize=(12, 8), layout='compressed')
        temp = np.zeros(shape=(2472, 3296), dtype=np.uint8)
        im = ax.imshow(temp, cmap='gray', vmin=0.0, vmax=255.0)
        plt.colorbar(im, ax=ax, aspect=20, fraction=0.08)

    for i, j in enumerate(range(start_index, start_index + num_images)):  # Generates images
        # Get random values for image parameters based on specified ranges
        snr = np.random.uniform(snr_range[0], snr_range[1])
        clustering = randint(clustering_range[0], clustering_range[1])
        bg_color = randint(bg_color_range[0], bg_color_range[1])

        if bg_sn_ratio is not None:
            sn_color = int(bg_color / np.random.uniform(bg_sn_ratio[0], bg_sn_ratio[1]))
        else:
            sn_color = int(bg_color / 1.53)

        num_fibers = max(0, int(np.random.normal(loc=fibers[0], scale=fibers[1])))
        print(f"Generating image {i + 1}/{num_images}\n",
              "snr:", snr, "cluster:", clustering, "bgcolor:", bg_color, "sncolor:", sn_color, "nfibs:", num_fibers)

        # Generate image
        name = f'snr{"{:.3f}".format(snr)}_cluster{clustering}_bg{bg_color}_sn{sn_color}_n{num_fibers}_image_{j}'
        image_names.append(name)
        new_img = generate_fiber_image(image_name=name,
                                       image_size=image_size,
                                       num_fibers=num_fibers,
                                       snr=snr,
                                       background_color=bg_color,
                                       fiber_color=sn_color,
                                       clustering=clustering,
                                       save=save, save_format=save_format,
                                       save_dir=save_dir, label_dir=label_dir,
                                       gui=False)
        if gui:
            if ax is None:
                continue

            ax.clear()
            ax.imshow(new_img, cmap='gray', vmin=0.0, vmax=255.0)

            if save:
                ax.set_title(f'Fibres: {num_fibers}', pad=20)
            else:
                ax.set_title(f'Fibres: {num_fibers}')

            plt.suptitle(f'Image {i + 1}/{num_images}')
            if save:
                ax.text(0, 0, f'{os.path.abspath(save_dir)}/{name}.png', va='bottom')
            plt.draw()
            plt.pause(0.5)

    if gui:  # necessary to keep open after all images are done
        plt.show()

    if save:
        if save_dir is None or label_dir is None or image_list_file is None:
            raise ValueError('save_dir, label_dir and image_list_file must be specified when saving images.')
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        impaths = [f'{os.path.abspath(save_dir)}/{name}.png' for name in image_names]
        with open(image_list_file, 'w') as f:  # Write paths to images
            f.write("\n".join(impaths))
        print(f"Generated {num_images} images in {save_dir} and saved annotations in {image_list_file}.")
    else:
        print(f"Generated {num_images} images")


if __name__ == '__main__':
    SAVING = True
    IMAGE_LIST_PATH = '../data/datasets/source_dataset/test.txt'
    IMAGE_SAVE_DIR = '../data/datasets/source_dataset/images/test'
    LABEL_SAVE_DIR = '../data/datasets/source_dataset/labels/test'

    generate_multiple_images(num_images=1000,
                             start_index=2200,
                             image_size=(2472, 2472),
                             snr_range=(2.5, 12.5),
                             clustering_range=(0, 10),
                             bg_color_range=(30, 220),
                             bg_sn_ratio=(1.2, 1.8),
                             fibers=(45, 15),
                             save=SAVING,
                             save_format='all',
                             image_list_file=IMAGE_LIST_PATH,
                             save_dir=IMAGE_SAVE_DIR,
                             label_dir=LABEL_SAVE_DIR,
                             gui=False)

    # generate_fiber_image(image_size=(2472, 3296),
    #                      num_fibers=100,
    #                      snr=2.0,
    #                      background_color=200,
    #                      clustering=10,
    #                      save=False,
    #                      gui=True)
