import pickle
import cv2
import numpy as np
import os
import torch
import test_model2.test_model2 as md2
import utils

current_dir = os.path.dirname(os.path.abspath(__file__))

def save_key(path, key):
    with open(path, 'wb') as f:
        pickle.dump(key, f)

def load_key(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def insert_watermark(host_image, watermark_image):
    h, w = host_image.shape[:2]
    watermark_image = cv2.resize(watermark_image, (w, h))

    watermark_image = np.where(watermark_image > 127, 1, 0)

    if len(host_image.shape) == 2:
        host_image = np.expand_dims(host_image, -1)
    if len(watermark_image.shape) == 2:
        watermark_image = np.expand_dims(watermark_image, -1)
    container = (host_image & ~1) | watermark_image
    return container

def extract_watermark(watermarked_image):
    if len(watermarked_image.shape) == 2:
        watermarked_image = np.expand_dims(watermarked_image, -1)
    extracted = watermarked_image & 1
    return extracted

def watermark_into_cloud(host_arr, channels, watermark_arr):
    MODEL_PATH = os.path.join(current_dir, '../models/pretrained_model2/model')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = md2.UNET(4, 2).to(device)
    unet.eval()
    unet.load_state_dict(torch.load(MODEL_PATH))

    with torch.no_grad():
        size_one_input = 2**(channels.itemsize*8)
        input_arr = np.transpose(channels, (2, 0, 1))/size_one_input
        input_tensor = torch.tensor(input_arr, dtype=torch.float32)
        input_tensor = torch.stack([input_tensor]).to(device)
        pred = unet.forward(input_tensor)
        pred_mask = md2.predb_to_mask(pred, 0)*255

    mask = pred_mask
    #utils.show_img(mask)
    square_mask = len(np.where(mask==255)[0])
    w_host, h_host = host_arr.shape[:2]
    if square_mask < 1/8*(w_host*h_host):          # Если площадь облаков достаточно меньше по сравнению с контейнером
         assert "Cloud size in image is too small" # то остановит

    idx_cloud_before = np.where(mask==255)
    idx_row = np.copy(idx_cloud_before[0])
    idx_col = np.copy(idx_cloud_before[1])
    indices = np.arange(idx_row.shape[0])
    np.random.shuffle(indices)

    shuffled_idx_row = idx_row[indices]
    shuffled_idx_col = idx_col[indices]

    idx_cloud = (shuffled_idx_row, shuffled_idx_col)
    save_key(os.path.join(current_dir, './keys/key.pkl'), idx_cloud)

    img_cloud = host_arr[idx_cloud]
    fix_size = int(np.sqrt(len(img_cloud)))
    cloud_for_wm = img_cloud[:fix_size**2]
    cloud_for_wm = np.reshape(cloud_for_wm, (fix_size, fix_size,-1))

    cloud_wm_ed = insert_watermark(cloud_for_wm, watermark_arr)
    img_cloud[:fix_size**2] = np.reshape(cloud_wm_ed, (fix_size**2, -1))
    host_img_wm_ed = np.copy(host_arr)
    host_img_wm_ed[idx_cloud] = np.copy(img_cloud)

    return host_img_wm_ed, idx_cloud

def extract_wm_from_cloud(container, idx_cloud):
    img_cloud = container[idx_cloud]

    fix_size = int(np.sqrt(len(img_cloud)))
    cloud_for_wm = img_cloud[:fix_size ** 2]
    cloud_for_wm = np.reshape(cloud_for_wm, (fix_size, fix_size,-1))

    extracted_wm = extract_watermark(cloud_for_wm)
    extracted_wm = np.array(extracted_wm, dtype=np.uint8)*255
    return extracted_wm

def false_watermark_into_cloud(host_arr, channels, watermark_arr):
    MODEL_PATH = os.path.join(current_dir, '../models/pretrained_model2/model')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = md2.UNET(4, 2).to(device)
    unet.eval()
    unet.load_state_dict(torch.load(MODEL_PATH))

    with torch.no_grad():
        size_one_input = 2**(channels.itemsize*8)
        input_arr = np.transpose(channels, (2, 0, 1))/size_one_input
        input_tensor = torch.tensor(input_arr, dtype=torch.float32)
        input_tensor = torch.stack([input_tensor]).to(device)
        pred = unet.forward(input_tensor)
        pred_mask = md2.predb_to_mask(pred, 0)*255

    mask = pred_mask
    square_mask = len(np.where(mask==255)[0])
    w_host, h_host = host_arr.shape[:2]
    if square_mask < 1/8*(w_host*h_host):          # Если площадь облаков достаточно меньше по сравнению с контейнером
         assert "Cloud size in image is too small" # то остановит

    idx_cloud = np.where(mask==255)
    img_cloud = host_arr[idx_cloud]
    fix_size = int(np.sqrt(len(img_cloud)))
    cloud_for_wm = img_cloud[:fix_size**2]
    cloud_for_wm = np.reshape(cloud_for_wm, (fix_size, fix_size,-1))

    cloud_wm_ed = insert_watermark(cloud_for_wm, watermark_arr)
    img_cloud[:fix_size**2] = np.reshape(cloud_wm_ed, (fix_size**2, -1))
    host_img_wm_ed = np.copy(host_arr)
    host_img_wm_ed[idx_cloud] = np.copy(img_cloud)

    img_cloud_in_channels = channels[idx_cloud]
    fix_size_in_channels = int(np.sqrt(len(img_cloud_in_channels)))
    cloud_for_wm_in_channels = img_cloud_in_channels[:fix_size_in_channels ** 2]
    cloud_for_wm_in_channels = np.reshape(cloud_for_wm_in_channels, (fix_size_in_channels, fix_size_in_channels, -1))

    cloud_wm_ed_in_channels = insert_watermark(cloud_for_wm_in_channels, watermark_arr)
    img_cloud_in_channels[:fix_size_in_channels ** 2] = np.reshape(cloud_wm_ed_in_channels, (fix_size_in_channels ** 2, -1))
    host_img_wm_ed_in_channels = np.copy(channels)
    host_img_wm_ed_in_channels[idx_cloud] = np.copy(img_cloud_in_channels)

    return host_img_wm_ed, host_img_wm_ed_in_channels, mask

def false_extract_wm_from_cloud(container, channels_wm_ed):
    MODEL_PATH = os.path.join(current_dir, '../models/pretrained_model2/model')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = md2.UNET(4, 2).to(device)
    unet.eval()
    unet.load_state_dict(torch.load(MODEL_PATH))

    with torch.no_grad():
        size_one_input = 2 ** (channels_wm_ed.itemsize * 8)
        input_arr = np.transpose(channels_wm_ed, (2, 0, 1)) / size_one_input
        input_tensor = torch.tensor(input_arr, dtype=torch.float32)
        input_tensor = torch.stack([input_tensor]).to(device)
        pred = unet.forward(input_tensor)
        pred_mask = md2.predb_to_mask(pred, 0) * 255

    mask = pred_mask
    idx_cloud = np.where(mask==255)
    img_cloud = container[idx_cloud]

    fix_size = int(np.sqrt(len(img_cloud)))
    cloud_for_wm = img_cloud[:fix_size ** 2]
    cloud_for_wm = np.reshape(cloud_for_wm, (fix_size, fix_size,-1))

    extracted_wm = extract_watermark(cloud_for_wm)
    extracted_wm = np.array(extracted_wm, dtype=np.uint8)*255
    return extracted_wm, mask
