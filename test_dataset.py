import torch
from utils.potsdam_dataset import PotsdamDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ds = PotsdamDataset()
dl = DataLoader(ds, batch_size=10)

print(ds.split_tiles)
print(ds.__getitem__(1))
#x = next(iter(dl))

"""document = x[0][0]
rgbidsm = document[:5,:,:]*ds.std_vals_tiles+ds.mean_vals_tiles
image = rgbidsm[:3,:,:].div(torch.max(rgbidsm[:3,:,:])).permute(1,2,0).numpy()

# plt.imsave("test_image.png",image)

plt.figure(figsize=(8,4))
for j in range(8):
    document = x[0][j]
    rgbidsm = document[:5,:,:]*ds.std_vals_tiles+ds.mean_vals_tiles
    rgb = rgbidsm[:3,:,:].div(torch.max(rgbidsm[:3,:,:])).permute(1,2,0).numpy()
    ir = rgbidsm[3,:,:].numpy()
    dsm = rgbidsm[4,:,:].numpy()
    print(j)
    plt.subplot(8,4,1)
    plt.imshow(rgb)
    plt.subplot(8,4,2)
    plt.imshow(ir)
    plt.subplot(8,4,3)
    plt.imshow(dsm)
    plt.subplot(document = x[i][j].numpy())
plt.savefig("test_image.png")"""