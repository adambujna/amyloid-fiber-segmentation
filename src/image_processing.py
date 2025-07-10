import cv2 as cv
import numpy as np

# Constants
SqrtPi = np.sqrt(np.pi)


def random_noise(image: np.ndarray, noise_level: float = 10.0) -> np.ndarray:
    """
    Adds Gaussian noise to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    noise_level : float
        Standard deviation of the noise.

    Returns
    -------
    np.ndarray: Image with noise added.
    """
    noise = np.random.normal(loc=0, scale=noise_level, size=image.shape).astype(np.float64)
    noise = cv.add(image.astype(np.float64), noise)
    return noise.clip(0, 255).astype(np.uint8)


def add_noise(image: np.ndarray, noise_strength: float = 10.0) -> np.ndarray:
    """
    Applies noise and Gaussian blurring to generated fiber image.

    The reduction in standard deviation which applying Gaussian blurring to a signal produces is:
    std_r ≈ std_IN / (std_G * 2√pi)
    where std_r is the standard deviation of the new signal, std_IN is the standard deviation of the input signal,
    and std_G is the standard deviation of the gaussian filter.

    With this, we can calculate the necessary noise level to produce an image with the specified noise level
    but with Gaussian blurring applied afterward.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    noise_strength : float
        Standard deviation of the noise.

    Returns
    -------
    np.ndarray: Noisy image produced by adding noise to the input image.
    """

    # Low grain noise
    std_G = 2
    std_IN_LOW = noise_strength * (std_G * 2 * SqrtPi) * 0.5    # Noise required is split in two because low as well
    image1 = random_noise(image, noise_level=std_IN_LOW)
    # After gaussian image with std_IN noise level will have a noise level of noise_strength.
    image1 = cv.GaussianBlur(image1, ksize=(9, 9), sigmaX=std_G)

    # High grain noise
    std_G_small = 0.1
    std_IN_HIGH = noise_strength * (std_G_small * 2 * SqrtPi) * 0.5
    image1 = random_noise(image1, noise_level=std_IN_HIGH)
    image1 = cv.GaussianBlur(image1, ksize=(5, 5), sigmaX=std_G_small)

    return image1


def get_length(mask: np.ndarray) -> float:
    """
    Applies the Zhang-Suen thinning algorithm to get a 1 px wide skeleton
     whose sum is the length of the object in pixels.

    Parameters
    ----------
    mask : np.ndarray
        Input binary mask.

    Returns
    -------
    float: The length of the skeleton.
    """
    mask *= 255 / np.max(mask)
    skeleton = cv.ximgproc.thinning(mask, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)
    return np.sum(skeleton > 0)


if __name__ == "__main__":
    sample_mask = np.zeros((1024, 1024)).astype(np.uint8)
    cv.fillPoly(sample_mask, [np.array([[889, 335], [792, 482], [793, 485], [797, 484],
                                        [893, 338], [892, 335], [889, 335]])], (255, 255, 255))
    print(get_length(sample_mask))
    cv.imshow("sample_mask", sample_mask)
    cv.waitKey(0)
