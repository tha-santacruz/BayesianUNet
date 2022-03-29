import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from train import train_net


##Hyperparameters

epochs= 5

batch_size= 32
#batch_size = [16, 32, 64, 128]

learning_rate= 1e-5
#learning_rate = [1e-4, 1e-5, 1e-6, 1e-7]

val_percent= 0.1

save_checkpoint = True

img_scale= 0.5

amp = False


#TODO : add optmizer in the wandb.config
#TODO : pass optimizer as parameter in train.py
optimizer = [Adam(weight_decay), SDGD(momentum), RMSprop]

#TODO : to iterate to 
#momentum = 

#TODO : defining the model load (if defined, it is the path for the *.pth file)
load = False


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')

# Change here to adapt to your data
net = UNet(n_channels=7, n_classes=9, bilinear=args.bilinear)

logging.info(f'Network:\n'
            f'\t{net.n_channels} input channels\n'
            f'\t{net.n_classes} output channels (classes)\n'
            f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

if load:
    net.load_state_dict(torch.load(load, map_location=device))
    logging.info(f'Model loaded from {load}')

# 1. Create dataset
dataset = BBKDataset(zone = ("genf", "goesch","jura"), split = "train", buildings = True, vegetation = True, random_seed = 1)

net.to(device=device)
try:
    train_net(net=net,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                val_percent=val / 100,
                amp=amp)
except KeyboardInterrupt:
    torch.save(net.state_dict(), 'INTERRUPTED.pth')
    logging.info('Saved interrupt')
    sys.exit(0)