import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image


# File paths
bird_image_path = r'data/bird.jpg'
bird_with_noise_image_path = r'data/bird-with-noise.jpg'
bird_with_noise_resized_image_path = r'data/bird-with-noise-resized.jpg'

# bird = np.array(Image.open(bird_image_path).resize(80, 36).convert('L'), dtype=np.uint8)
# print(bird.shape)
# plt.imshow(bird)

bird_with_noise = np.array(Image.open(bird_with_noise_resized_image_path).convert('L'), dtype=np.uint8)
# print(bird_with_noise.shape)
# plt.imshow(bird_with_noise)

# HPF
hpf_kernel_3x3 = np.array([[0,  1,  0],
                           [1,  4,  1],
                           [0,  1,  0]])

hpf_kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                           [-1, -1,  1, -1, -1],
                           [-1,  1,  2,  1, -1],
                           [-1, -1,  1, -1, -1],
                           [-1, -1, -1, -1, -1]])

# LPF
lpf_kernel_3x3 = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])


# hpf_bird_with_noise = ndimage.convolve(bird_with_noise, hpf_kernel_3x3)
# print(hpf_bird_with_noise.shape)
# plt.imshow(hpf_bird_with_noise, cmap='gray')
# Image.fromarray(hpf_bird_with_noise).save(r'data/hpf_bird_with_noise_4.jpg')

lpf_bird_with_noise = ndimage.convolve(bird_with_noise, lpf_kernel_3x3)
Image.fromarray(lpf_bird_with_noise).save(r'data/lpf_bird_with_noise.jpg')
