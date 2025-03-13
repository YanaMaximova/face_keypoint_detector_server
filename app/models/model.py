import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class FaceModel(pl.LightningModule):
    def __init__(self, class_count=28):
        super().__init__()

        self.norm0 = nn.BatchNorm2d(3)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, padding = 'same')
        self.relu2 = nn.LeakyReLU(0.1)
        self.norm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, 3, padding = 'same')
        self.relu3 = nn.LeakyReLU(0.1)
        self.norm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, padding = 'same')
        self.relu4 = nn.LeakyReLU(0.1)
        self.norm4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 128, 3, padding = 'same')
        self.relu5 = nn.LeakyReLU(0.1)
        self.norm5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, 3, padding = 'same')
        self.relu6 = nn.LeakyReLU(0.1)
        self.norm6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 256, 3, padding = 'same')
        self.relu7 = nn.LeakyReLU(0.1)
        self.norm7 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv8 = nn.Conv2d(256, 256, 3, padding = 'same')
        self.relu8 = nn.LeakyReLU(0.1)
        self.norm8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(256, 256, 3, padding = 'same')
        self.relu9 = nn.LeakyReLU(0.1)
        self.norm9 = nn.BatchNorm2d(256)

        self.conv10 = nn.Conv2d(256, 512, 3, padding = 'same')
        self.relu10 = nn.LeakyReLU(0.1)
        self.norm10 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv11 = nn.Conv2d(512, 512, 3, padding = 'same')
        self.relu11 = nn.LeakyReLU(0.1)
        self.norm11 = nn.BatchNorm2d(512)

        self.conv12 = nn.Conv2d(512, 512, 3, padding = 'same')
        self.relu12 = nn.LeakyReLU(0.1)
        self.norm12 = nn.BatchNorm2d(512)

        self.conv13 = nn.Conv2d(512, 1024, 3, padding = 'same')
        self.relu13 = nn.LeakyReLU(0.1)
        self.norm13 = nn.BatchNorm2d(1024)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.LazyLinear(class_count)
        self.relu14 = nn.ReLU()

        self.loss = F.mse_loss

    def forward(self, x):

        x =  self.norm0(x)

        x = self.pool1(self.norm1(self.relu1(self.conv1(x))))

        x = x + self.norm3(self.relu3(self.conv3(self.norm2(self.relu2(self.conv2(x))))))
        x = self.pool2(self.norm4(self.relu4(self.conv4(x))))

        x = x + self.norm6(self.relu6(self.conv6(self.norm5(self.relu5(self.conv5(x))))))
        x = self.pool3(self.norm7(self.relu7(self.conv7(x))))

        x = x + self.norm9(self.relu9(self.conv9(self.norm8(self.relu8(self.conv8(x))))))
        x = self.pool4(self.norm10(self.relu10(self.conv10(x))))

        x = x + self.norm12(self.relu12(self.conv12(self.norm11(self.relu11(self.conv11(x))))))
        x = self.pool5(self.norm13(self.relu13(self.conv13(x))))

        x = self.relu14(self.fc1(self.flatten(x)))

        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        metrics = { "train_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=5e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode='min',
                                                                  factor=0.5,
                                                                  patience=10,
                                                                  verbose=True)
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        }
        return [optimizer], [lr_dict]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        metrics = {"val_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return metrics
