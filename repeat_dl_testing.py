import torch
from utils.dataset import BBKDataset
from torch.utils.data import DataLoader
# https://stackoverflow.com/questions/12007820/better-ways-to-get-nth-element-from-an-unsubscriptable-iterable

bbkd = BBKDataset(zone = ("genf",),split="test", augment=True)
dl = DataLoader(bbkd, batch_size=32, shuffle=False)

# for x, (image, target) in enumerate(dl):
#     print(image.size())
#     print(target.size())
# print(x)

x = 3
im1 = enumerate(dl)[x]
im2 = enumerate(dl)[x]

print(im1.size())
print(im2.size())
