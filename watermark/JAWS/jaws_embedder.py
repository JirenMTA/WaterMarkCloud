import math
import os
import torch
import utils
import cv2
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))

class SystemJASW:
    def __init__(self, K, M, alpha):
        self.K = K
        self.M = M
        self.alpha = alpha

    @staticmethod
    def corr2d_unnormalized(base: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        assert len(base.shape) == 2 and len(pattern.shape) == 2
        N1, N2 = pattern.shape
        M1, M2 = base.shape

        h_flipped = np.flip(pattern)

        base_padded = np.pad(base, ((0, N1 - 1), (0, N2 - 1)))
        h_padded = np.pad(h_flipped, ((0, M1 - 1), (0, M2 - 1)))

        F = np.fft.fft2(base_padded)
        H = np.fft.fft2(h_padded)

        G = np.fft.ifft2(F * H)
        return np.real(G)

    @staticmethod
    def conv(img_input, mask_input):
        return convolve2d(img_input, mask_input, boundary="symm", mode="same")

    @staticmethod
    def generate_sample_P(K, M):
        np.random.seed(K)
        P = np.random.normal(0, 1, (M, M))
        return P

    @staticmethod
    def shift(P, delta_x, delta_y):
        return np.roll(np.roll(P, delta_x, axis=0), delta_y, axis=1)

    @staticmethod
    def compute_W(P, b):
        delta_x = b
        delta_y = b
        shifted_P = SystemJASW.shift(P, delta_x, delta_y)
        W_t = P - shifted_P
        return W_t

    @staticmethod
    def show_img(img):
        plt.figure()
        plt.imshow(img, cmap='gray')

    @staticmethod
    def calculate_psnr(original, distorted):
        original = original.astype(np.float64)
        distorted = distorted.astype(np.float64)

        mse = np.mean((original - distorted) ** 2)

        if mse == 0:
            return float('inf')

        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return psnr

    def insert_watermark(self, img, beta, b):
        P = SystemJASW.generate_sample_P(self.K, self.M)
        W = SystemJASW.compute_W(P, b)

        container_wm_ed = np.zeros_like(img)
        N1, N2 = container_wm_ed.shape

        for row in range(N1):
            for col in range(N2):
                container_wm_ed[row][col] = img[row][col] + \
                                            self.alpha * beta[row][col] * W[row % self.M][col % self.M]

        return container_wm_ed

    def extract_watermark(self, container_wm_ed):
        P = SystemJASW.generate_sample_P(self.K, self.M)
        N1, N2 = container_wm_ed.shape
        W_hat = np.zeros((self.M, self.M))
        S = math.floor(N1 / self.M) * math.floor(N2 / self.M)

        for m1 in range(self.M):
            for m2 in range(self.M):
                for row in range(math.floor(N1 / self.M)):
                    for col in range(math.floor(N2 / self.M)):
                        W_hat[m1][m2] += container_wm_ed[row * self.M + m1][col * self.M + m2]

        W_hat = W_hat / S
        B = SystemJASW.corr2d_unnormalized(W_hat, P)

        x = np.argmin(B)
        row_x = x // B.shape[0]
        col_x = x % B.shape[1]

        y = np.argmax(B)
        row_y = y // B.shape[0]
        col_y = y % B.shape[1]

        shift_x = row_x - row_y
        shift_y = col_x - col_y

        shift_x = (shift_x + self.M * 2) % self.M
        shift_y = (shift_y + self.M * 2) % self.M
        # print(f"Первый пик: {row_x}, {col_x}")
        # print(f"Второй пик: {row_y}, {col_y}")
        # print(f"Сдвиг: ({shift_x}, {shift_y})")

        return shift_x

# SystemJASW.show_img(channels_norm[0])
# SystemJASW.show_img(container_watermarked)
# plt.show()


#print(np.max(pred))
#print(np.where(pred > 0))
# utils.show_img(mask)
#
# print(pred)


