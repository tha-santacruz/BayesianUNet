from utils.dataset import BBKDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

bbkd = BBKDataset(zone = ("genf",),split="test", augment=False)
dl_normal = DataLoader(bbkd, batch_size=64, shuffle=False)
bbkd = BBKDataset(zone = ("genf",),split="test", augment=True)
dl_augment = DataLoader(bbkd, batch_size=64, shuffle=False)

x = next(iter(dl_normal))
y = next(iter(dl_augment))

for j in range(64):
    document = x[0][j]
    rgbidsm = document[:5,:,:]*bbkd.std_vals_tiles+bbkd.mean_vals_tiles
    image_normal = rgbidsm[:3,:,:].div(torch.max(rgbidsm[:3,:,:])).permute(1,2,0)
    document = y[0][j]
    rgbidsm = document[:5,:,:]*bbkd.std_vals_tiles+bbkd.mean_vals_tiles
    image_augment = rgbidsm[:3,:,:].div(torch.max(rgbidsm[:3,:,:])).permute(1,2,0)
    diff = image_normal-image_augment
    diff = torch.abs(diff).mean()
    print(f"mean difference {diff}")
plt.savefig("test_image.png")