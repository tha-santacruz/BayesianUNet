""" Full assembly of the parts to form the complete network """

from .bayesian_unet_parts import *


class BayesianUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(BayesianUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, drop_channels=False)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # print("start size : {}".format(x.size()))
        # print("encoding...")
        x1 = self.inc(x)
        # print("step 1 size : {}".format(x1.size()))
        x2 = self.down1(x1)
        # print("step 2 size : {}".format(x2.size()))
        x3 = self.down2(x2)
        # print("step 3 size : {}".format(x3.size()))
        x4 = self.down3(x3)
        # print("step 4 size : {}".format(x4.size()))
        x5 = self.down4(x4)
        # print("step 5 size : {}".format(x5.size()))
        # print("decoding...")
        x = self.up1(x5, x4)
        # print("step 6 size : {}".format(x.size()))
        x = self.up2(x, x3)
        # print("step 7 size : {}".format(x.size()))
        x = self.up3(x, x2)
        # print("step 8 size : {}".format(x.size()))
        x = self.up4(x, x1)
        # print("step 9 size : {}".format(x.size()))
        logits = self.outc(x)
        # print("end size : {}".format(logits.size()))
        return logits
