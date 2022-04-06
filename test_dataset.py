from utils.dataset import BBKDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

bbkd = BBKDataset(zone = ("goesch",),split="all", augment=False)
print(len(bbkd))
dl = DataLoader(bbkd, batch_size=64, shuffle=True)
x = next(iter(dl))
print(x[0][0].size())
print(x[1][0].type())

document = x[0][0]
rgbidsm = document[:5,:,:]*bbkd.std_vals_tiles+bbkd.mean_vals_tiles
image = rgbidsm[:3,:,:].div(torch.max(rgbidsm[:3,:,:])).permute(1,2,0).numpy()

plt.imshow(image)