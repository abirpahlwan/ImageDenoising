import numpy as np


def gaussian(source):
    row, column, channel = source.shape
    mean = 0.005
    var = 0.005
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, column, channel))
    gauss = gauss.reshape(row, column, channel)
    noisy = source
    return noisy


def salt_pepper(image, SNR):
    channel, row, column = image.shape
    mask = np.random.choice((0, 1, 2), size=(1, row, column), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    mask = np.repeat(mask, channel, axis=0)   # Copy by channel to have the same shape as source
    print(mask)
    # mask = image + mask
    image[mask == 1] = 255                    # salt noise
    image[mask == 2] = 0
    return image


def salt_pepper_v2(image, amount):
    s_vs_p = 0.5
    out = np.copy(image)

    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 200

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 100

    return out
