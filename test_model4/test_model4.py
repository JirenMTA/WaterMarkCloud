import numpy as np
from torch.utils.data import DataLoader
import data
import segmentation_models_pytorch as smp
import torch
import pytorch_lightning as pl
import os
import cv2
import utils

current_dir = os.path.dirname(os.path.abspath(__file__))

class SegmentationCloudModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        image = image.clone().detach().float()
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch[0]
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]
        mask = mask/255.
        assert mask.ndim == 4

        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")
    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

def test_and_show_1_input(arch):
    MODEL_PATH = os.path.join(current_dir, f"../models/trained_model4/model4_{arch}.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegmentationCloudModel(arch, "resnet34", in_channels=4, out_classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

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

        merge_input = cv2.merge((input_r, input_g, input_b, input_nir))/255.
        merge_input = np.transpose(merge_input, (2, 0, 1))
        merge_input = np.stack([merge_input])
        tensor_mrg_input = torch.tensor(merge_input, dtype=torch.float32).to(device)
        pred = model.forward(tensor_mrg_input).cpu()[0][0]
        pred = torch.nn.functional.sigmoid(pred)

        binary_mask = np.array((pred > 0.5).float())
        utils.show_images(
            [(input_gt, "Ground truth"),
             (binary_mask, "Ground predict")],
            f"Model 4 ({arch} base on ResNet34 using pre-trained model in pip)"
        )

def write_losses_of_model(arch):
    MODEL_PATH = os.path.join(current_dir, f"../models/trained_model4/model4_{arch}.pth")

    dataset = data.Dataset38Cloud('train', 500)
    dl = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegmentationCloudModel(arch, "resnet34", in_channels=4, out_classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    size_one_element_label = 2 ** (dataset[0][1].itemsize*8)

    with open(os.path.join(current_dir, "../result_loss.txt"), 'a') as f:
        print("-------------------------------------------------------------------", file=f)
        print(f"******************** {arch} with base ResNet34 using package pip ********************", file=f)
        loss = 0
        curr_num_input = 0
        with torch.no_grad():
            for i, (inputs, masks) in enumerate(dl):
                inputs = inputs.clone().detach().float().to(device)
                truth_mask = masks / size_one_element_label
                truth_mask = truth_mask.clone().detach().float().to(device)

                pred = model.forward(inputs)
                pred = torch.nn.functional.sigmoid(pred)
                pred_binary_mask = (pred > 0.5).float()

                loss += torch.nn.functional.mse_loss(pred_binary_mask, truth_mask)
                curr_num_input += inputs.shape[0]
                print(f'{loss} - Num inputs: {curr_num_input}', file=f)
        print('', file=f)
        print("--------------------------END--------------------------", file=f)

def train_model(arch):
    MODEL_PATH = os.path.join(current_dir, f"../models/trained_model4/model4_{arch}.pth")
    dataset = data.Dataset38Cloud('train', 8400)
    model = SegmentationCloudModel(arch, "resnet34", in_channels=4, out_classes=1)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=5,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    #train_model(arch='Unet')
    #test_and_show_1_input(arch = 'FPN')
    write_losses_of_model(arch='Unet')