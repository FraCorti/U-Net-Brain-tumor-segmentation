from scipy import ndimage
import imageio
import numpy as np


def load_image(filename):
    image = imageio.imread(filename)
    # input image is color png depicting grayscale, just use first plane from here on
    image = image[:, :, 1].astype(np.float64)
    # print(image.shape)
    # print('image: min = ', np.min(image), ' max = ', np.max(image))

    return image


def rmse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)

    return np.sqrt(mse)


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0

    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def compute_normalization(image):
    return np.linalg.norm(image)
