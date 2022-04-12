from utils.dataset import BBKDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

bbkd = BBKDataset(zone = ("alles",),split="all", augment=True)
print(len(bbkd))
dl = DataLoader(bbkd, batch_size=64, shuffle=True)
# x = next(iter(dl))
# print(x[0][0].size())
# print(x[1][0].type())

# document = x[0][0]
# rgbidsm = document[:5,:,:]*bbkd.std_vals_tiles+bbkd.mean_vals_tiles
# image = rgbidsm[:3,:,:].div(torch.max(rgbidsm[:3,:,:])).permute(1,2,0).numpy()

# plt.imsave("test_image.png",image)

# j = 10
#     f, axarr = plt.subplots(4,2)
#     axarr[0,0].imshow(noisy_imgs_1[4*j].permute(1,2,0).numpy())
#     axarr[0,1].imshow(noisy_imgs_2[4*j].permute(1,2,0).numpy())
#     axarr[1,0].imshow(noisy_imgs_1[4*j+1].permute(1,2,0).numpy())
#     axarr[1,1].imshow(noisy_imgs_2[4*j+1].permute(1,2,0).numpy())
#     axarr[2,0].imshow(noisy_imgs_1[4*j+2].permute(1,2,0).numpy())
#     axarr[2,1].imshow(noisy_imgs_2[4*j+2].permute(1,2,0).numpy())
#     axarr[3,0].imshow(noisy_imgs_1[4*j+3].permute(1,2,0).numpy())
#     axarr[3,1].imshow(noisy_imgs_2[4*j+3].permute(1,2,0).numpy())

for i in dl:
    plt.figure(figsize=(8,8))
    for j in range(64):
        plt.subplot(8,8,j+1)
        document = i[0][j]
        rgbidsm = document[:5,:,:]*bbkd.std_vals_tiles+bbkd.mean_vals_tiles
        image = rgbidsm[:3,:,:].div(torch.max(rgbidsm[:3,:,:])).permute(1,2,0).numpy()
        plt.imshow(image)
    plt.show()