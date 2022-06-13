import torch
from evaluate import evaluate
from utils.dataset import BBKDataset
from torch.utils.data import DataLoader
# from unet import UNet
from bayesian_unet import BayesianUNet

bbkd = BBKDataset(zone = ("genf", "goesch"))
dl = DataLoader(bbkd, batch_size=16, shuffle=True)
x = next(iter(dl))
print(x[0][0].size())
print(x[1][0].type())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = BayesianUNet(n_channels=7, n_classes=9, bilinear=False).to(device=device)
print(net)
evaluate(net,dl,device)