import os
from test_model4.test_model4 import SegmentationCloudModel
import torch
import numpy as np
import cv2
import utils
from watermark.JAWS.jaws_embedder import SystemJASW

host_test_path = 'image_test_wm/JAWS/host.jpg'
watermark_test_path = 'image_test_wm/JAWS/watermark.jpg'
current_dir = os.path.dirname(os.path.abspath(__file__))

def get_model(device):
    MODEL_PATH = os.path.join(current_dir, f"models/trained_model4/model4_unet.pth")
    model = SegmentationCloudModel('unet', "resnet34", in_channels=4, out_classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def get_mask(channels, device, model):
    with torch.no_grad():
        merge_input = np.stack([channels])
        tensor_mrg_input = torch.tensor(merge_input, dtype=torch.float32).to(device)
        pred = model.forward(tensor_mrg_input).cpu()[0][0]
        pred = torch.nn.functional.sigmoid(pred)
        mask = np.array(pred)
    return mask

def get_input():
    watermark = utils.read_img(watermark_test_path)
    lst_channel_name = ['red', 'green', 'blue', 'nir']
    channels = []
    for channel in lst_channel_name:
        channel_path = f'image_test_wm/JAWS/host_{channel}.jpg'
        channel_arr = cv2.imread(channel_path, 0)
        channels.append(channel_arr)

    host = utils.read_img(host_test_path)
    return host, channels, watermark

if __name__ == "__main__":
    host, channels, watermark = get_input()
    channels_norm = np.array(channels) / 255.
    host_norm = np.array(host) / 255.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)

    value_to_hide = 25

    sys_insert_watermark = SystemJASW(K=100, M=256, alpha=0.02)
    mask = get_mask(channels_norm, device, model)
    container_watermarked = np.copy(host_norm)
    container_watermarked[:, :, 0] = sys_insert_watermark.insert_watermark(host_norm[:, :, 0], mask, value_to_hide)

    sys_reveal_watermark = SystemJASW(K=100, M=256, alpha=0.02)
    reveal_b = sys_reveal_watermark.extract_watermark(container_watermarked[:, :, 0])

    if value_to_hide == reveal_b:
        print("Successfully extracted hidden information!")
    else:
        print(f"Failed extracted hidden information!, b={value_to_hide}")

    PSNR = SystemJASW.calculate_psnr(host_norm[:,:,0], container_watermarked[:,:,0])


    print(f"PSNR = {PSNR}")

    utils.show_images(
    [
        (host_norm, "Before"),
        (container_watermarked, "After"),
        (np.abs(container_watermarked[:, :, 0] - host_norm[:, :, 0]), "Diff"),
    ], "JAWS")
