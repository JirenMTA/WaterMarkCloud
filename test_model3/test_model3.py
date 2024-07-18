import torch
import torch.nn as nn
import torchsummary
import utils
import os
import cv2
import numpy as np
import data
from torch.utils.data import DataLoader, Dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "..\\models\\pretrained_model3\\unet_model.pth")

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNet, self).__init__()

        # Downsampling
        self.down1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.down2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Upsampling
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv4 = self.conv_block(256, 128)

        self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv5 = self.conv_block(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        bottleneck = self.bottleneck(p2)

        u4 = self.up4(bottleneck)
        u4 = torch.cat([u4, c2], dim=1)
        c4 = self.up_conv4(u4)

        u5 = self.up5(c4)
        u5 = torch.cat([u5, c1], dim=1)
        c5 = self.up_conv5(u5)

        outputs = self.out_conv(c5)

        return outputs

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

input_size = (4, 384, 384)

def test_and_show_1_input():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = UNet(in_channels=input_size[0], out_channels=1).to(device)
    unet.load_state_dict(torch.load(MODEL_PATH))
    unet.eval()

    with torch.no_grad():
        input_r = cv2.imread(os.path.join(current_dir, "../1_true_dataset/"
                            "red_patch_192_10_by_12_LC08_L1TP_002053_20160520_20170324_01_T1.jpg"),
                             cv2.IMREAD_GRAYSCALE)
        input_g = cv2.imread(os.path.join(current_dir, "../1_true_dataset/"
                            "green_patch_192_10_by_12_LC08_L1TP_002053_20160520_20170324_01_T1.jpg"),
                             cv2.IMREAD_GRAYSCALE)
        input_b = cv2.imread(os.path.join(current_dir, "../1_true_dataset/"
                            "blue_patch_192_10_by_12_LC08_L1TP_002053_20160520_20170324_01_T1.jpg"),
                             cv2.IMREAD_GRAYSCALE)
        input_nir = cv2.imread(os.path.join(current_dir, "../1_true_dataset/"
                              "nir_patch_192_10_by_12_LC08_L1TP_002053_20160520_20170324_01_T1.jpg"),
                               cv2.IMREAD_GRAYSCALE)
        input_gt = cv2.imread(os.path.join(current_dir, "../1_true_dataset/"
                             "gt_patch_192_10_by_12_LC08_L1TP_002053_20160520_20170324_01_T1.jpg"),
                              cv2.IMREAD_GRAYSCALE)

        merge_input = cv2.merge((input_r, input_g, input_b, input_nir)) / np.iinfo(input_r.dtype).max
        merge_input = np.transpose(merge_input, (2, 0, 1))
        merge_input = np.stack([merge_input])
        tensor_mrg_input = torch.tensor(merge_input, dtype=torch.float32).to(device)

        pred = unet.forward(tensor_mrg_input)
        pred_mask = torch.sigmoid(pred)
        pred_mask = torch.where(pred_mask >= 0.5, torch.tensor(1.0), torch.tensor(0.0)).cpu()[0][0]

        utils.show_images(
            [(input_gt, "Ground truth"),
            (pred_mask, "Ground predict")],
            "Model 3 (UNet without pre-trained base)"
        )

def write_losses_of_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = UNet(in_channels=input_size[0], out_channels=1).to(device)
    unet.eval()
    unet.load_state_dict(torch.load(MODEL_PATH))

    dataset = data.Dataset38Cloud('train', 500)
    size_one_element_input = 2 ** (dataset[0][0].itemsize * 8)
    size_one_element_label = 2 ** (dataset[0][1].itemsize * 8)
    dl = DataLoader(dataset, batch_size=32, shuffle=False)

    with open(os.path.join(current_dir, "../result_loss.txt"), 'a') as f:
        print("-------------------------------------------------------------------", file=f)
        print("******************** UNET without base ********************", file=f)
        loss = 0
        curr_num_input = 0
        with torch.no_grad():
            for i, (inputs, masks) in enumerate(dl):
                inputs = inputs.clone().detach().float().to(device)
                truth_mask = masks / size_one_element_label
                truth_mask = truth_mask.clone().detach().float().to(device)

                truth_mask = truth_mask.unsqueeze(1)
                pred_mask = unet.forward(inputs)
                pred_mask = torch.sigmoid(pred_mask)
                pred_mask = torch.where(pred_mask >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
                pred_mask = torch.unsqueeze(pred_mask, 1)

                loss += torch.nn.functional.mse_loss(pred_mask, truth_mask)

                curr_num_input += inputs.shape[0]
                print(f'{loss} - Num inputs: {curr_num_input}', file=f)
        print('', file=f)
        print("--------------------------END--------------------------", file=f)


if __name__ == "__main__":
    test_and_show_1_input()
    #write_losses_of_model()

