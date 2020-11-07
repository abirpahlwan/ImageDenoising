import cv2

import noise


bird_image_path = r'data/hawk.jpg'
image = cv2.imread(bird_image_path, cv2.IMREAD_GRAYSCALE)

# noisy_bird = noise.salt_pepper(image.transpose(2, 1, 0), 0.9)
# noisy_bird = noisy_bird.transpose(2, 1, 0)

noisy_bird = noise.salt_pepper_v2(image, 0.1)

cv2.imshow('bird', noisy_bird)
cv2.waitKey(0)
cv2.destroyAllWindows()
