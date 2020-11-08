import numpy as np


# HPF
hpf_edge_kernel_3x3 = np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]])

hpf_basic_kernel_5x5 = np.array([[ 0, -1, -1, -1,  0],
                                 [-1,  2, -4,  2, -1],
                                 [-1, -4, 13, -4, -1],
                                 [-1,  2, -4,  2, -1],
                                 [ 0, -1, -1, -1,  0]])

hpf_kernel_3x3 = np.array([[0,  1,  0],
                           [1,  8,  1],
                           [0,  1,  0]])

hpf_kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                           [-1, -1,  1, -1, -1],
                           [-1,  1,  2,  1, -1],
                           [-1, -1,  1, -1, -1],
                           [-1, -1, -1, -1, -1]])

# LPF
lpf_average_kernel_5x5 = np.array(np.ones((5, 5), dtype=np.uint8))/25
