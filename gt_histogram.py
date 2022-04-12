from utils.dataset import BBKDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

bbkd = BBKDataset(zone = ("goesch",),split="all", augment=True)
dl = DataLoader(bbkd, batch_size=64, shuffle=True)
x = next(iter(dl))
output, counts = torch.unique(x[1], return_counts=True)
print(output)
print(counts)
