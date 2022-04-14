from utils.dataset import BBKDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

bbkd = BBKDataset(zone = ("genf",),split="test", augment=False)
print(len(bbkd))
dl = DataLoader(bbkd, batch_size=16, shuffle=False)
# print(bbkd.split_counts)
# print(bbkd.split_coordinates)
x = next(iter(dl))
# print(x[0][0].size())
# print(x[1][0].type())

num = 1

# document = x[0][num]
# rgbidsm = document[:5,:,:]*bbkd.std_vals_tiles+bbkd.mean_vals_tiles
# rgb = rgbidsm[:3,:,:].div(torch.max(rgbidsm[:3,:,:])).permute(1,2,0).numpy()
# ir = rgbidsm[3,:,:].div(torch.max(rgbidsm[3,:,:])).numpy()
# dsm = rgbidsm[4,:,:].div(torch.max(rgbidsm[4,:,:])).numpy()
# build = document[5,:,:].numpy()
# hoe = document[6,:,:]*bbkd.std_vals_vegetation+bbkd.mean_vals_vegetation
# hoe = hoe.div(torch.max(hoe)).numpy()
# bbk = x[1][num].argmax(dim=0).numpy()

# print(np.shape(rgb))
# print(np.shape(ir))
# print(np.shape(dsm))
# print(np.shape(build))
# print(np.shape(hoe))
# print(np.shape(bbk))

# plt.figure()
# plt.subplot(321)
# plt.imshow(rgb)
# plt.subplot(322)
# plt.imshow(ir)
# plt.subplot(323)
# plt.imshow(dsm)
# plt.subplot(324)
# plt.imshow(build)
# plt.subplot(325)
# plt.imshow(hoe)
# plt.subplot(326)
# plt.imshow(bbk)
# plt.savefig("example_data.png")
# plt.show()

counter = 0
for i in dl:
	counter += 1
	document = i[0][num]
	rgbidsm = document[:5,:,:]*bbkd.std_vals_tiles+bbkd.mean_vals_tiles
	rgb = rgbidsm[:3,:,:].div(torch.max(rgbidsm[:3,:,:])).permute(1,2,0).numpy()
	ir = rgbidsm[3,:,:].div(torch.max(rgbidsm[3,:,:])).numpy()
	dsm = rgbidsm[4,:,:].div(torch.max(rgbidsm[4,:,:])).numpy()
	build = document[5,:,:].numpy()
	hoe = document[6,:,:]*bbkd.std_vals_vegetation+bbkd.mean_vals_vegetation
	hoe = hoe.div(torch.max(hoe)).numpy()
	bbk = i[1][num].argmax(dim=0).numpy()

	# print(np.shape(rgb))
	# print(np.shape(ir))
	# print(np.shape(dsm))
	# print(np.shape(build))
	# print(np.shape(hoe))
	# print(np.shape(bbk))

	plt.figure()
	plt.subplot(321)
	plt.imshow(rgb)
	plt.subplot(322)
	plt.imshow(ir)
	plt.subplot(323)
	plt.imshow(dsm)
	plt.subplot(324)
	plt.imshow(build)
	plt.subplot(325)
	plt.imshow(hoe)
	plt.subplot(326)
	plt.imshow(bbk)
	plt.savefig(f'example_data/example_data{counter}.png')


