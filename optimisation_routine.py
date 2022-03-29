import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch 
from torch import optim

from unet import UNet
from train import train_net
from utils.dataset import BBKDataset

zone = ("all",)
device = 'cuda'
##Hyperparameters

epochs= 10

batch_size= 32
#batch_size = [16, 32, 64, 128]

learning_rate_list= [1e-5]
#learning_rate_list =  [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

save_checkpoint = True

amp = False

bilinear = False

# Choose optimizer
optim_class_list  = [optim.RMSprop, optim.Adam, optim.SGD]
#optimizer = [Adam(weight_decay), SDGD(momentum), RMSprop]

#Defining the loading of pre_trained model  (if defined, it is the path for the *.pth file)
load = False

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Create datasets
    train_set = BBKDataset(zone = zone, split = "train", buildings = True, vegetation = True, random_seed = 1)
    val_set = BBKDataset(zone = zone, split = "val", buildings = True, vegetation = True, random_seed = 1)

    # Change here to adapt to your data
    net = UNet(n_channels=7, n_classes=9, bilinear=bilinear)

    logging.info(f'Network:\n'
                    f'\t{net.n_channels} input channels\n'
                    f'\t{net.n_classes} output channels (classes)\n'
                    f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if load:
        net.load_state_dict(torch.load(load, map_location=device))
        logging.info(f'Model loaded from {load}')

    net.to(device=device)

    for optim_class in optim_class_list:
        for learning_rate in learning_rate_list:

            try:
                train_net(net=net,
                            val_set=val_set,
                            train_set=train_set,
                            optim_class = optim_class,
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            device=device,
                            amp=amp)

            except KeyboardInterrupt:
                torch.save(net.state_dict(), 'INTERRUPTED.pth')
                logging.info('Saved interrupt')
                sys.exit(0)
