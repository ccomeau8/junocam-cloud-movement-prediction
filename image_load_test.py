import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

matplotlib.use('Qt5Agg')
file_path = 'data/featured/JNCE_2022272_45C00054_V01-artenhanced.png'
loaded = cv2.imread(file_path, cv2.IMREAD_COLOR)
width = 1080
width_scale = width / loaded.shape[1]
height = int(width_scale * loaded.shape[0])
img = cv2.resize(loaded, (width, height), fy=width_scale)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = loaded
print(f"Initial Shape : {loaded.shape}, New Shape: {img.shape}")

cv2.imshow('Initial Image', img)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharp = cv2.filter2D(img, -1, kernel)
blur = cv2.medianBlur(sharp, 3)

cv2.imshow('Blurred', blur)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 50, 250)

gray = cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)

sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Find x and y gradients
sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
sobelX2 = cv2.Sobel(gray, cv2.CV_64F, 2, 0)  # Find x and y gradients
sobelY2 = cv2.Sobel(gray, cv2.CV_64F, 0, 2)  # Find x and y gradients

print(f"{gray.dtype}")
magnitude = np.sqrt(sobelX ** 2.0 + sobelY ** 2.0)
angle = np.arctan2(sobelY, sobelX) * (180 / np.pi)
# abs_cos = np.abs(np.cos(sobelY/sobelX)*(180/np.pi))

magnitude2 = np.sqrt(sobelX2 ** 2.0 + sobelY2 ** 2.0)
angle2 = np.arctan2(sobelY2, sobelX2) * (180 / np.pi)

vis = np.concatenate((cv2.normalize(magnitude, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX),
                      cv2.normalize(angle, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)), axis=1)
vis2 = np.concatenate((cv2.normalize(magnitude2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX),
                      cv2.normalize(angle2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)), axis=1)

print(np.min(magnitude), np.max(magnitude), np.min(angle), np.min(angle))

print(f'{magnitude.shape=}, {angle.shape=}, {sobelX.shape=}, {sobelY.shape=}')
cv2.imshow('Gradient', vis)
cv2.imshow('2nd order Gradient', vis2)

gap_size = 5
kernel_size = 5
x_coords = np.arange(kernel_size, width - kernel_size, gap_size, dtype=int)
y_coords = np.arange(kernel_size, height - kernel_size, gap_size, dtype=int)

x_grid, y_grid = np.meshgrid(x_coords, y_coords)
u = np.zeros((len(y_coords), len(x_coords)))
v = np.zeros((len(y_coords), len(x_coords)))

for row_idx, y in enumerate(y_coords):
    for col_idx, x in enumerate(x_coords):
        # print(f"{y - kernel_size} {y + kernel_size + 1} {x - kernel_size} {x + kernel_size + 1}")
        x_kernel_slice = sobelX[y - kernel_size:y + kernel_size + 1, x - kernel_size:x + kernel_size + 1]
        y_kernel_slice = sobelY[y - kernel_size:y + kernel_size + 1, x - kernel_size:x + kernel_size + 1]

        mag_slice = np.sqrt(x_kernel_slice ** 2, y_kernel_slice ** 2)

        max_dir_idx = np.unravel_index(np.abs(x_kernel_slice).argmax(), mag_slice.shape)
        # print(f"{kernel_slice.shape=}")

        u[row_idx, col_idx] = x_kernel_slice[max_dir_idx]
        v[row_idx, col_idx] = y_kernel_slice[max_dir_idx]

field_angles = np.arctan2(v, u)
abs_cos = np.abs(np.cos(field_angles))

# for row_idx, (row_x, row_y) in enumerate(zip(x_grid, y_grid)):
#     for col_idx, (x,y) in enumerate(zip(row_x, row_y)):

# u = sobelX[y_grid, x_grid]
# v = sobelY[y_grid, x_grid]

colormap = cm.inferno
dirs = abs_cos
norm = Normalize()
norm.autoscale(dirs)
print(f"{u.shape} {colormap(norm(dirs)).shape}")
plt.quiver(x_grid, y_grid, u, v, units='xy', angles='xy')
plt.imshow(rgb_img)

plt.figure()
plt.streamplot(x_grid, y_grid, abs(u), abs(v))
plt.imshow(rgb_img)
plt.show()

cv2.waitKey(0)
