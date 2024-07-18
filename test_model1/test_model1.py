import cv2
import torch
import torch.nn as nn
import numpy as np
import torchsummary
from torchvision.models import resnet18
from torch.nn.functional import interpolate
import torch.nn.functional as F
import utils
import data
from torch.utils.data import DataLoader
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "../models/pretrained_model1/fpn.pth")

class SemiBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(SemiBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class UpsampleModule(nn.Module):
    def __init__(self, in_chs, decoder_chs, norm_layer):
        super(UpsampleModule, self).__init__()

        self.down_conv = nn.Conv2d(in_chs, decoder_chs, kernel_size=1, bias=False)
        self.down_bn = norm_layer(decoder_chs)
        downsample = nn.Sequential(
            self.down_conv,
            self.down_bn,
        )
        self.conv_enc = SemiBasicBlock(in_chs, decoder_chs, downsample=downsample, norm_layer=norm_layer)
        self.conv_out = SemiBasicBlock(decoder_chs, decoder_chs, norm_layer=norm_layer)
        self.conv_up = nn.ConvTranspose2d(decoder_chs, decoder_chs, kernel_size=2, stride=2, bias=False)

    def forward(self, enc, prev):
        enc = self.conv_up(prev) + self.conv_enc(enc)
        dec = self.conv_out(enc)
        return dec

class FPNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(FPNHead, self).__init__()
        decoder_chs = out_channels
        layer_chs_list = [512, 256, 128, 64]  # Update the layer channels based on your input size

        self.conv_enc2dec = nn.Conv2d(layer_chs_list[0], decoder_chs, kernel_size=1, bias=False)
        self.bn_enc2dec = norm_layer(out_channels)
        self.relu_enc2dec = nn.ReLU(inplace=True)

        self.up3 = UpsampleModule(layer_chs_list[1], decoder_chs, norm_layer)
        self.up2 = UpsampleModule(layer_chs_list[2], decoder_chs, norm_layer)
        self.up1 = UpsampleModule(layer_chs_list[3], decoder_chs, norm_layer)

        self.conv_up0 = nn.ConvTranspose2d(decoder_chs, decoder_chs, kernel_size=2, stride=2, bias=False)
        self.conv_up1 = nn.ConvTranspose2d(decoder_chs, out_channels, kernel_size=2, stride=2, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, c1, c2, c3, c4):
        c4 = self.relu_enc2dec(self.bn_enc2dec(self.conv_enc2dec(c4)))
        c3 = self.up3(c3, c4)
        c2 = self.up2(c2, c3)
        c1 = self.up1(c1, c2)

        c1 = self.conv_up0(c1)
        c1 = self.conv_up1(c1)
        return c1

class FPN(nn.Module):
    def __init__(self, backbone, num_classes, pretrained=False, norm_layer=nn.BatchNorm2d):
        super(FPN, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        if backbone == 'resnet18':
            self.pretrained = resnet18(pretrained=pretrained)
            self.base_forward = self._resnet_base_forward

        in_chs_dict = {"resnet18": 512}  # Update the input channels based on your backbone
        in_chs = in_chs_dict[backbone]
        self.head = FPNHead(in_chs, num_classes, norm_layer)

        self._initialize_weights()

    def _resnet_base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, h, w = x.size()
        c1, c2, c3, c4 = self.base_forward(x)
        x = self.head(c1, c2, c3, c4)
        # x = torch.sigmoid(x)  # Apply sigmoid activation for binary classification
        x = interpolate(x, size=(h, w), mode='bilinear', align_corners=False)  # Resize output to input size
        return x

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

def test_and_show_1_input():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fpn = FPN(backbone='resnet18', num_classes=2, pretrained=False).to(device)
    fpn.load_state_dict(torch.load(MODEL_PATH))
    fpn.eval()

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
        input_gt = cv2.imread(os.path.join(current_dir, "../1_true_dataset/"
                             "gt_patch_192_10_by_12_LC08_L1TP_002053_20160520_20170324_01_T1.jpg"),
                             cv2.IMREAD_GRAYSCALE)

        merge_input = cv2.merge((input_r, input_g, input_b)) / np.iinfo(input_r.dtype).max
        merge_input = np.transpose(merge_input, (2, 0, 1))
        merge_input = np.stack([merge_input])
        tensor_mrg_input = torch.tensor(merge_input, dtype=torch.float32).to(device)

        pred = fpn.forward(tensor_mrg_input)
        pred_mask = predb_to_mask(pred, 0)
        utils.show_images(
            [(input_gt, "Ground truth"),
            (pred_mask, "Ground predict")],
            "Model 1 (FPN base on ResNet)"
        )

def write_losses_of_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fpn = FPN(backbone='resnet18', num_classes=2, pretrained=False).to(device)

    dataset = data.Dataset38Cloud('train', 500)
    size_one_element_input = 2 ** (dataset[0][0].itemsize * 8)
    size_one_element_label = 2 ** (dataset[0][1].itemsize * 8)
    dl = DataLoader(dataset, batch_size=32, shuffle=False)

    fpn.load_state_dict(torch.load(MODEL_PATH))
    fpn.eval()
    curr_num_input = 0
    with open(os.path.join(current_dir, "../result_loss.txt"), 'a') as f:
        print("-------------------------------------------------------------------", file=f)
        print("******************** FPN with base ResNet18 ********************", file=f)
        loss = 0
        with torch.no_grad():
            for i, (inputs, masks) in enumerate(dl):
                inputs = inputs[:,:3,:,:]
                inputs = inputs.clone().detach().float().to(device)
                truth_mask = masks / size_one_element_label
                truth_mask = truth_mask.clone().detach().float().to(device)

                pred_mask = fpn.forward(inputs)
                pred_mask = torch.argmax(F.softmax(pred_mask, dim=1), dim=1)
                pred_mask = torch.unsqueeze(pred_mask, 1)

                loss += torch.nn.functional.mse_loss(pred_mask, truth_mask)

                curr_num_input += inputs.shape[0]
                print(f'{loss} - Num inputs: {curr_num_input}', file=f)
        print('', file=f)
        print("--------------------------END--------------------------", file=f)

if __name__ == "__main__":
    #write_losses_of_model()
    # test_and_show_1_input()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fpn = FPN(backbone='resnet18', num_classes=2, pretrained=False).to(device)
    torchsummary.summary(fpn, input_size=(3,384,384))
