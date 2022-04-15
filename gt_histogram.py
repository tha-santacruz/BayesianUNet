from utils.dataset import BBKDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

bbkd = BBKDataset(zone = ("genf",),split="all", augment=True)
dl = DataLoader(bbkd, batch_size=64, shuffle=True)
counts_cumulated = torch.tensor([0,0,0,0,0,0,0,0,0])
for x in tqdm(dl):
    output, counts = torch.unique(x[1].argmax(dim=1), return_counts=True)
    for i in range(len(output)):
        counts_cumulated[output[i]] += counts[i]
counts_cumulated = counts_cumulated.div(counts_cumulated.sum())
print(counts_cumulated)
plt.bar(height=counts_cumulated.numpy()*100, x = bbkd.BBK_CLASSES_list)
plt.xticks(rotation=90)
plt.xlabel('Classe')
plt.ylabel('Pourcentage')
plt.grid(visible=True)
plt.tight_layout()
plt.savefig('example_data/classes_histogram.png')
plt.show()
