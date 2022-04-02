from utils.dataset import BBKDataset
from torch.utils.data import DataLoader

bbkd = BBKDataset(zone = ("genf", "goesch"),split="val")
dl = DataLoader(bbkd, batch_size=64, shuffle=True)
x = next(iter(dl))
print(x[0][0].size())
print(x[1][0].type())
