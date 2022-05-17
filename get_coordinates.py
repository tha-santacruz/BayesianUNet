import torch
from utils.dataset import BBKDataset
import pandas as pd

bbkd = BBKDataset(zone = ("alles",),split="all", augment=True)
print(len(bbkd.coordinates))

df = pd.DataFrame(bbkd.coordinates)
print(df.head())
df.to_csv("coordinates.csv")