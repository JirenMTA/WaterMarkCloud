import torch.nn.functional as F
import torchsummary
from torch.utils.data import DataLoader
import data
import torch.nn as nn
import torch
from torchvision.models import resnet34
import utils
import cv2
import os

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "..\\models\\pretrained_model2\\model")

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Modify first layer of ResNet34 to accept custom number of channels
        base_model = resnet34(pretrained=False)  # Change this line
        base_model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.upconv4 = self.expand_block(512, 256)
        self.upconv3 = self.expand_block(256 * 2, 128)
        self.upconv2 = self.expand_block(128 * 2, 64)
        self.upconv1 = self.expand_block(64 * 2, 64)
        self.upconv0 = self.expand_block(64 * 2, out_channels)

    def forward(self, x):
        # Contracting Path
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # Expansive Path
        upconv4 = self.upconv4(layer4)
        upconv3 = self.upconv3(torch.cat([upconv4, layer3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, layer2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, layer1], 1))
        upconv0 = self.upconv0(torch.cat([upconv1, layer0], 1))

        return upconv0

    def expand_block(self, in_channels, out_channels):
        expand = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )
        return expand

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

def test_and_show_1_input():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = UNET(4, 2).to(device)
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
        pred_mask = predb_to_mask(pred, 0)
        utils.show_images(
            [(input_gt, "Ground truth"),
            (pred_mask, "Ground predict")],
            "Model 2 (UNet base on ResNet)"
        )

def write_losses_of_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = data.Dataset38Cloud('train', 500)
    dl = DataLoader(dataset, batch_size=32, shuffle=False)

    unet = UNET(4, 2).to(device)
    unet.load_state_dict(torch.load(MODEL_PATH))
    unet.eval()

    size_one_element_input = 2 ** (dataset[0][0].itemsize*8)
    size_one_element_label = 2 ** (dataset[0][1].itemsize*8)

    with open(os.path.join(current_dir, "../result_loss.txt"), 'a') as f:
        print("-------------------------------------------------------------------", file=f)
        print("******************** UNET with base ResNet34 ********************", file=f)
        loss = 0
        curr_num_input = 0
        with torch.no_grad():
            for i, (inputs, masks) in enumerate(dl):
                inputs = inputs.clone().detach().float().to(device)
                truth_mask = masks / size_one_element_label
                truth_mask = truth_mask.clone().detach().float().to(device)

                pred_mask = unet.forward(inputs)
                pred_mask = torch.argmax(F.softmax(pred_mask, dim=1), dim=1)
                pred_mask = torch.unsqueeze(pred_mask, 1)

                loss += torch.nn.functional.mse_loss(pred_mask, truth_mask)
                curr_num_input += inputs.shape[0]
                print(f'{loss} - Num inputs: {curr_num_input}', file=f)
        print('', file=f)
        print("--------------------------END--------------------------", file=f)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = UNET(4, 2).to(device)
    torchsummary.summary(unet, input_size=(4, 384, 384))
    #test_and_show_1_input()
    #write_losses_of_model()


