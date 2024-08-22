import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_img(img_path):
    img_arr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if len(img_arr.shape) == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    return img_arr

def show_img(img_array):
    plt.imshow(img_array)
    plt.show()

def show_image_by_ax(ax, arr, title):
    ax.set_title(title)
    ax.imshow(arr)

def show_images(imgs, window_title):
    fig = plt.figure(figsize=(8, 4))
    plt.suptitle(window_title, fontsize=16)
    n = len(imgs)
    cols = 3
    rows = (n + cols - 1) // cols

    for i, (img, title) in enumerate(imgs):
        ax_img = plt.subplot(rows, cols, i+1)
        show_image_by_ax(ax_img, img, title)
        ax_img.axis('off')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.tight_layout()
    plt.show()

def concat_channels(r, g, b):
    if len(r.shape) == 2:
        r = np.expand_dims(r, axis=-1)
    if len(g.shape) == 2:
        g = np.expand_dims(g, axis=-1)
    if len(b.shape) == 2:
        b = np.expand_dims(b, axis=-1)
    img = np.concatenate((r,g,b), axis=-1)
    return img

def save_img(path, img_arr):
    img_to_save = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img_to_save)


if __name__ == "__main__":
    img = cv2.imread(".\\1_true_dataset\\truecolor_patch_192_10_by_12_LC08_L1TP_002053_20160520_20170324_01_T1.jpg")
    # print(img)
    # print(img.shape)
    cv2.imshow("edc", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
