import numpy as np
import cv2
import matplotlib
from pysteps.utils.spectral import rapsd
from matplotlib import pyplot as plt

rng = np.random.default_rng()


def generate_random_baseline(image, max_mag=1, output_fft_image=False):
    is_color = len(image.shape) == 3

    x_grads = rng.integers(-max_mag, max_mag, image.shape[0:2], endpoint=True)
    y_grads = rng.integers(-max_mag, max_mag, image.shape[0:2], endpoint=True)

    out_image = advect_image(image, x_grads, y_grads)
    if is_color:
        gray_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2GRAY).astype(float) * (1.0 / 255.0)
    else:
        gray_image = out_image

    if output_fft_image:
        fft_mags, fft_image = generate_image_fft(gray_image, True)
        return out_image, fft_mags, fft_image
    else:
        fft_mags = generate_image_fft(gray_image)
        return out_image, fft_mags


def advect_image(image, x_grads, y_grads):
    is_color = len(image.shape) == 3

    if is_color:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    new_image = image.copy()
    # new_image = np.zeros_like(gray_image)

    movement_stacks = [[[] for i in range(width)] for j in range(height)]
    for y in range(height):
        for x in range(width):
            x_grad = x_grads[y, x]
            y_grad = y_grads[y, x]
            new_x_pos = x - x_grad
            new_y_pos = y - y_grad
            if (0 <= new_x_pos < width) and (0 <= new_y_pos < height):
                movement_stacks[new_y_pos][new_x_pos].append(image[y, x])
                # movement_stacks[y][x].append(gray_image[new_y_pos, new_x_pos])
    outsides = 0
    max_len = 0
    for y in range(height):
        for x in range(width):
            if len(movement_stacks[y][x]) == 0:
                outsides += 1
                continue
            if len(movement_stacks[y][x]) > max_len:
                max_len = len(movement_stacks[y][x])
            new_image[y, x] = np.mean(movement_stacks[y][x],
                                      axis=0)
            # sum(movement_stacks[y][x]) / len(movement_stacks[y][x])
    print(f"{outsides=} {max_len=}")
    return new_image


def generate_image_fft(gray_image, output_fft_image=False):
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    psd = 20 * np.log10(rapsd(magnitude_spectrum))
    # powers = magnitude_spectrum.sum(axis=0)
    # powers = powers[len(powers) // 2:]
    # powers[1:] *= 2
    # mags = 20 * np.log10(powers)
    if output_fft_image:
        return psd, 20 * np.log10(magnitude_spectrum)
    return psd


if __name__ == "__main__":
    matplotlib.use('Qt5Agg')
    file_path = 'data/featured/pj45-three-north-circumpolar-cyclones.png'
    loaded = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img_width = 480
    width_scale = img_width / loaded.shape[1]
    img_height = int(width_scale * loaded.shape[0])
    img = cv2.resize(loaded, (img_width, img_height), fy=width_scale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float) * (1.0 / 255.0)
    print(f"{np.min(gray)=} {np.max(gray)=}")
    # gray = cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = loaded
    print(f"Initial Shape : {loaded.shape}, New Shape: {img.shape}")
    print(gray.dtype)
    input_mags = generate_image_fft(gray)

    baseline, baseline_mags, baseline_fft = generate_random_baseline(img, 10, output_fft_image=True)
    baseline5, baseline5_mags, baseline5_fft = generate_random_baseline(img, 5, output_fft_image=True)
    baseline20, baseline20_mags, baseline20_fft = generate_random_baseline(img, 20, output_fft_image=True)

    print(f"{np.min(baseline_mags)=} {np.max(baseline_mags)=}")

    img_center = (img_width // 2, img_height // 2)
    noise_mask_size = 50
    noise_mask = np.ones_like(gray, bool)
    print("Here")
    noise_mask[img_center[1] - noise_mask_size: img_center[1] + noise_mask_size,
    img_center[0] - noise_mask_size: img_center[0] + noise_mask_size] = 0
    signal_mask = 1 - noise_mask
    # input_noise = np.average(fft[noise_mask])
    # input_center = np.average(fft[signal_mask])
    # input_signal = np.average(fft)
    # input_std_dev = np.std(fft)
    # baseline_noise = np.average(baseline_fft[noise_mask])
    # baseline_center = np.average(baseline_fft[signal_mask])
    # baseline_signal = np.average(baseline_fft)
    # baseline_std_dev = np.std(baseline_fft)
    #
    # print(np.all(fft == baseline_fft))
    input_f = np.fft.fft2(gray)
    input_fshift = np.fft.fftshift(input_f)
    input_magnitude_spectrum = np.abs(input_fshift)

    cv2.imwrite('data/output/baseline_5.png', baseline5)
    cv2.imwrite('data/output/baseline_10.png', baseline)
    cv2.imwrite('data/output/baseline_20.png', baseline20)

    cv2.imwrite('data/output/input_fft.png', 40 * np.log10(input_magnitude_spectrum))
    cv2.imwrite('data/output/baseline_5_fft.png', 2*baseline5_fft)
    cv2.imwrite('data/output/baseline_10_fft.png', 2*baseline_fft)
    cv2.imwrite('data/output/baseline_20_fft.png', 2*baseline20_fft)

    plt.subplot(121), plt.imshow(gray, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(20 * np.log10(input_magnitude_spectrum), cmap='gray')
    plt.title(
        f'Input FFT'), plt.xticks(
        []), plt.yticks([])
    # plt.subplot(223), plt.imshow(baseline, cmap='gray')
    # plt.title('Random Baseline'), plt.xticks([]), plt.yticks([])
    # plt.subplot(224), plt.imshow(20 * np.log10(baseline_fft), cmap='gray')
    # plt.title(
    #     f'Random Baseline FFT, Signal={baseline_signal:.3f} Center={baseline_center:.3f} Noise={baseline_noise:.3f} Stddev={input_std_dev:.3f}'), plt.xticks(
    #     []), plt.yticks([])

    plt.show()
    plt.figure()
    # plt.title('Radially Averaged Power Spectral Density of Random Baselines')
    plt.plot(input_mags)
    plt.plot(baseline5_mags)
    plt.plot(baseline_mags)
    plt.plot(baseline20_mags)
    plt.legend(['Input', 'Mag<=5', 'Mag<=10', 'Mag<=20'])
    plt.ylim(bottom=0)
    plt.xlabel('Wavenumber')
    plt.ylabel('Power')
    plt.show()

    plt.figure()

    plt.subplot(221), plt.imshow(rgb_img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(222), plt.imshow(baseline5)
    plt.title('Max Magnitude = 5'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(baseline)
    plt.title('Max Magnitude = 10'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(baseline20)
    plt.title('Max Magnitude = 20'), plt.xticks([]), plt.yticks([])
    plt.show()
