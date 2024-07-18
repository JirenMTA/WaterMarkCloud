import watermark.watermark as wm
import utils
import cv2
import numpy as np
host_test_path = './image_test_wm/host.jpg'
watermark_test_path = './image_test_wm/watermark.jpg'
MODEL_PATH = './models/pretrained_model2/model'

def visualize_difference(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    img2 = cv2.resize(img2, (img1.shape[0], img1.shape[1]))

    if len(img1.shape) == 2:
        img1 = np.expand_dims(img1, -1)
    if len(img2.shape) == 2:
        img2 = np.expand_dims(img2, -1)

    diff = np.abs(np.average(img1 - img2, axis=-1))
    #print(f'Число неверных писелей: {len(np.where(diff != 0 )[0])}')
    utils.show_img(diff)

def false_approach():
    watermark = utils.read_img(watermark_test_path)
    lst_channel_name = ['red', 'green', 'blue', 'nir']
    channels = []
    for channel in lst_channel_name:
        channel_path = f'./image_test_wm/host_{channel}.jpg'
        channel_arr = cv2.imread(channel_path, 0)
        channels.append(channel_arr)

    host = utils.read_img(host_test_path)
    merge_channels = np.stack(channels, axis=2)
    container, channels_wm_ed, mask_1 = wm.false_watermark_into_cloud(host, merge_channels, watermark)
    extracted, mask_2 = wm.false_extract_wm_from_cloud(container, channels_wm_ed)

    utils.show_img(mask_1)
    utils.show_img(mask_2)
    visualize_difference(mask_1, mask_2)

def true_approach():
    watermark = utils.read_img(watermark_test_path)
    lst_channel_name = ['red', 'green', 'blue', 'nir']
    channels = []
    for channel in lst_channel_name:
        channel_path = f'./image_test_wm/host_{channel}.jpg'
        channel_arr = cv2.imread(channel_path, 0)
        channels.append(channel_arr)

    host = utils.read_img(host_test_path)
    merge_channels = np.stack(channels, axis=2)
    container, key = wm.watermark_into_cloud(host, merge_channels, watermark)
    extracted = wm.extract_wm_from_cloud(container, key)

    utils.save_img('./image_test_wm/container.png', container)
    utils.save_img('./image_test_wm/extracted_watermark.png', extracted)

    visualize_difference(host, container)

if __name__ == "__main__":
    true_approach()
    #false_approach()