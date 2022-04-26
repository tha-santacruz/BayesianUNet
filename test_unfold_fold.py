import torch
import torch.nn as nn


patch_size = 4 # patch size in pixel
patch_num = 2 # sqrt of number of paches 


batch= torch.randint(low=0,high=9,size=(1,1,patch_size*patch_num,patch_size*patch_num)).to(torch.float32)
print("example batch")
print(batch)
print(batch.size())

unfold = nn.Unfold(kernel_size=(patch_size,patch_size),stride=patch_size)
unfolded = unfold(batch).mean(dim=1)
print("unfolded batch")
print(unfolded)
print(unfolded.size())

#expanded = unfolded.view(unfolded.size(0),1,unfolded.size(1)).expand(unfolded.size(0),4,unfolded.size(1))
#unfolded.view(1,1,2500).expand(32,4,2500)

# fold = nn.Fold(output_size=(batch.size(-2),batch.size(-1)), kernel_size=(patch_size, patch_size), stride=patch_size)