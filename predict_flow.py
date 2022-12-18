import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from fft_performance import generate_image_fft, generate_random_baseline, advect_image


def generate_single_flow_image_vector(image, x, y, u, v):
    u_image = np.zeros_like(image)
    v_image = np.zeros_like(image)

    mag = np.sqrt(u ** 2 + v ** 2)

    u_image = cv2.circle(u_image, (x, y), int(mag), u, -1)
    v_image = cv2.circle(v_image, (x, y), int(mag), v, -1)
    return u_image, v_image


if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    file_path = 'data/featured/Stitched_Image_2.png'
    loaded = cv2.imread(file_path, cv2.IMREAD_COLOR)

    img_width = 480
    width_scale = img_width / loaded.shape[1]
    img_height = int(width_scale * loaded.shape[0])
    img = cv2.resize(loaded, (img_width, img_height), fy=width_scale)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # .astype(float) * (1.0 / 255.0)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp = cv2.filter2D(gray, -1, kernel)
    blur = cv2.medianBlur(sharp, 3)

    sobelX = cv2.Sobel(blur, cv2.CV_64F, 1, 0)  # Find x and y gradients
    sobelY = cv2.Sobel(blur, cv2.CV_64F, 0, 1)

    magnitude = np.sqrt(sobelX ** 2.0 + sobelY ** 2.0)
    angle = np.arctan2(sobelY, sobelX)
    # vis = np.concatenate((cv2.normalize(magnitude, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX),
    #                       cv2.normalize(angle, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)), axis=1)
    hist = np.histogram(magnitude)
    thresh = np.percentile(magnitude, 80)
    edges = magnitude > thresh
    # canny = np.zeros_like(edges)
    canny = cv2.Canny(gray, 50, 250, L2gradient=False) == 255
    canny2 = cv2.Canny(gray, 50, 250, L2gradient=True)

    # circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, img_height / 8,
    #                            param1=250, param2=50,
    #                            minRadius=5, maxRadius=img_height // 4)

    # circle_img = np.copy(rgb_img)
    #
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         center = (i[0], i[1])
    #         # circle center
    #         cv2.circle(circle_img, center, 1, (0, 100, 100), 3)
    #         # circle outline
    #         radius = i[2]
    #         cv2.circle(circle_img, center, radius, (255, 0, 255), 3)

    masked = np.copy(rgb_img)
    cnt_img = np.copy(rgb_img)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(cnt_img, contours, -1, (0, 255, 0), cv2.FILLED, 8, hierarchy)

    (y_edges, x_edges) = np.where(canny)
    # u_vec = sobelX[canny]
    # v_vec = sobelY[canny]
    print(f"Len = {len(canny)}")
    grad_mag = 5

    # Rotated by 90 degrees to point vectors along edges

    angle = angle + np.pi / 2

    vec_mag = 5

    u_vec = vec_mag * np.cos(angle[canny])
    v_vec = vec_mag * np.sin(angle[canny])

    u_images = np.zeros(gray.shape)
    v_images = np.zeros(gray.shape)
    u_counts = np.zeros(gray.shape)
    v_counts = np.zeros(gray.shape)

    print("Generating flow image stack")
    for i, (vec_x, vex_y, u_val, v_val) in enumerate(zip(x_edges, y_edges, u_vec, v_vec)):
        u_single, v_single = generate_single_flow_image_vector(gray, vec_x, vex_y, u_val, v_val)
        u_images += u_single
        v_images += v_single
        u_counts += (u_single > 0.01)
        v_counts += (v_single > 0.01)

    print("Combining")
    u_images[u_counts > 0] /= u_counts[u_counts > 0]
    v_images[v_counts > 0] /= v_counts[v_counts > 0]
    x_grads = u_images.astype(int)  # np.ma.masked_equal(u_images, 0).mean(axis=2)
    y_grads = v_images.astype(int)  # np.ma.masked_equal(v_images, 0).mean(axis=2)
    print("Done")
    mesh_x, mesh_y = np.meshgrid(range(img_width), range(img_height))

    u_gauss = np.zeros_like(gray)
    v_gauss = np.zeros_like(gray)
    u_gauss[canny] = u_vec
    v_gauss[canny] = v_vec
    plt.figure()
    plt.imshow(rgb_img)
    # plt.quiver(x_edges, y_edges, u_vec, v_vec, units='xy', angles='xy')
    plt.quiver(mesh_x, mesh_y, u_gauss, v_gauss, units='xy', angles='xy')

    plt.title("initial plot")

    u_gauss = cv2.GaussianBlur(u_gauss, (3*vec_mag, 3*vec_mag), 10*vec_mag).astype(int)
    v_gauss = cv2.GaussianBlur(v_gauss, (3*vec_mag, 3*vec_mag), 10*vec_mag).astype(int)

    # x_grads = np.zeros_like(gray)
    # y_grads = np.zeros_like(gray)
    # x_grads[canny] = u
    # y_grads[canny] = v

    input_mags = generate_image_fft(gray)

    baseline, baseline_mags = generate_random_baseline(gray, 10)

    test_img = advect_image(img, x_grads, y_grads)
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    test_mags = generate_image_fft(gray_test)

    gauss_img = advect_image(img, u_gauss, v_gauss)
    gray_gauss = cv2.cvtColor(gauss_img, cv2.COLOR_BGR2GRAY)
    gauss_mags = generate_image_fft(gray_gauss)

    masked[canny] = [255, 0, 0]

    cv2.imwrite('data/output/canny.png', cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
    cv2.imwrite('data/output/gauss.png', gauss_img)
    cv2.imwrite('data/output/group.png', test_img)

    plt.figure()
    # plt.subplot(131)

    plt.imshow(rgb_img)
    # plt.quiver(x_edges, y_edges, u_vec, v_vec, units='xy', angles='xy',minshaft = 1, minlength=0)
    plt.quiver(mesh_x, mesh_y, x_grads, y_grads, units='xy', angles='xy', minshaft = 1, minlength=0)

    plt.figure()
    plt.imshow(test_img)
    plt.figure()
    plt.imshow(gauss_img)
    plt.figure()
    plt.subplot(211)
    plt.imshow(x_grads)
    plt.subplot(212)
    plt.imshow(y_grads)
    plt.figure()
    plt.imshow(masked), plt.xticks([]), plt.yticks([])
    # plt.figure()
    # plt.imshow(cnt_img)
    plt.figure()
    plt.imshow(gray, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.figure(), plt.imshow(baseline, cmap='gray')
    plt.title('Baseline Max Magnitude = 10'), plt.xticks([]), plt.yticks([])
    plt.figure(), plt.imshow(test_img, cmap='gray')
    plt.title('Advected'), plt.xticks([]), plt.yticks([])

    plt.figure()
    plt.plot(input_mags)
    plt.plot(baseline_mags)
    plt.plot(test_mags)
    plt.plot(gauss_mags)
    plt.legend(['Input', 'Baseline', 'Grouping', 'Gauss'])
    plt.title('Radially Averaged Power Spectral Density')
    plt.ylim(bottom=0)

    plt.show()
