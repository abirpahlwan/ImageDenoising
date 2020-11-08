import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage

import noise
import kernel


# File paths
bird_image_path = r'data/birdo.jpg'
# bird_with_noise_image_path = r'data/bird-with-noise.jpg'
# bird_with_noise_resized_image_path = r'data/bird-with-noise-resized.jpg'

bird_image = cv2.imread(bird_image_path, cv2.IMREAD_UNCHANGED)
# cv2.imshow('bird_image', bird_image)

bird_with_noise = noise.salt_pepper_v2(bird_image, 0.1)
cv2.imshow('bird_with_noise', bird_with_noise)

hpf_bird_with_noise = ndimage.convolve(bird_with_noise, kernel.hpf_edge_kernel_3x3 / 9)
# print(hpf_bird_with_noise.shape)
cv2.imshow('hpf_bird_with_noise', hpf_bird_with_noise)

lpf_bird_with_noise = ndimage.convolve(bird_with_noise, kernel.lpf_average_kernel_5x5)
# print(lpf_bird_with_noise.shape)
cv2.imshow('lpf_bird_with_noise', lpf_bird_with_noise)

cv2.waitKey(0)
cv2.destroyAllWindows()
