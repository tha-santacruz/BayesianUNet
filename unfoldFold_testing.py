import torch
import torch.nn as nn


patch_size = 2 # patch size in pixel
patch_num = 2 # sqrt of number of paches
doc_num = 4 # number of documents in the batch


batch= torch.randint(low=0,high=9,size=(doc_num,1,patch_size*patch_num,patch_size*patch_num)).to(torch.float32)
print("example batch")
print(batch)
print(batch.size())

unfold = nn.Unfold(kernel_size=(patch_size,patch_size),stride=patch_size)
unfolded = unfold(batch).mean(dim=1)
print("unfolded batch")
print(unfolded)
print(unfolded.size())

expanded = unfolded.view(unfolded.size(0),1,unfolded.size(1)).expand(unfolded.size(0),patch_size**2,unfolded.size(1))
print("expanded batch")
print(expanded)
print(expanded.size())

fold = nn.Fold(output_size=(batch.size(-2),batch.size(-1)), kernel_size=(patch_size, patch_size), stride=patch_size)
folded = fold(expanded)
print("folded batch")
print(folded)
print(folded.size())