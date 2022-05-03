from utils.dataset import BBKDataset
import albumentations as A
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import numpy as np
from bayesian_unet import BayesianUNet

# dropout activation
def enable_dropout(model):
	""" Function to enable the dropout layers during test-time """
	for m in model.modules():
		if m.__class__.__name__.startswith('Dropout'):
			m.train()
			print("activated dropout")


# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset
bbkd = BBKDataset(zone = ("genf",),split="test", augment=False)
print(len(bbkd))
dl = DataLoader(bbkd, batch_size=32, shuffle=False)
x = next(iter(dl))

# load model
net = BayesianUNet(n_channels=7, n_classes=9, bilinear=False).to(device=device)
checkpoint_path = 'checkpoints_bayesian/checkpoint_epoch60.pth'
net.load_state_dict(torch.load(checkpoint_path, map_location=device))
net.eval()
enable_dropout(net)

nb_forward = 20

num = 1

# Labels and colors
# classes : "null","wooded_area", "water", "bushes", "individual_tree", "no_woodland", "ruderal_area", "without_vegetation", "buildings"
# rgb : [0,0,0], [0,104,0], [0, 192, 210], 
# hex : '#000000', '#006800','#00c0d2', '#73fe8c', '#00d200', '#fffac7', '#d7b384', '#d4d3d4', '#ed0038'
bbk_cmap = colors.ListedColormap(['#000000', '#006800','#00c0d2', '#73fe8c', '#00d200', '#fffac7', '#d7b384', '#d4d3d4', '#ed0038'])
bbk_scale = [0,1,2,3,4,5,6,7,8,9]
bbk_scale = [-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5]
labels = ["null","wooded_area", "water", "bushes", "individual_tree", "no_woodland", "ruderal_area", "without_vegetation", "buildings"]
hex_colors = ['#000000', '#006800','#00c0d2', '#73fe8c', '#00d200', '#fffac7', '#d7b384', '#d4d3d4', '#ed0038']

# create a patch (proxy artist) for every color
patches = [ mpatches.Patch(color=hex_colors[i], label=labels[i]) for i in range(len(hex_colors)) ]
plt.figure()
plt.legend(handles=patches, loc='center', ncol = 1, markerscale=2, fontsize='xx-large')
plt.axis('off')
plt.savefig(f'example_data/bbk_legend.png')

# unfolding and folding
w_size = 4 #patch size
accuracy_tresh = 0.5
t = 0.4 # default 0.4
unfold = torch.nn.Unfold(kernel_size=(w_size, w_size),stride = w_size)
fold = torch.nn.Fold(output_size=(200,200), kernel_size=(w_size, w_size), stride=w_size)

counter = 0


for i in dl:
	for j in range(len(i[0])):
		bbk = i[1][j].argmax(dim=0)
		if torch.unique(bbk[bbk!=0], return_counts=False).size(0) >= 7:
			doc = i[0].clone().detach().to(device=device)
			print(doc.size())
			# to store n_forward predictions on the same batch
			dropout_predictions = torch.empty((0,i[1].size(0),i[1].size(1),i[1].size(2),i[1].size(3)))

			#to store n_forwad prediction (for the test time augmentation)
			aleatoric_predictions = torch.empty((0,i[1].size(0),i[1].size(1),i[1].size(2),i[1].size(3)))
			
			# bayesian inference
			for f_pass in range(nb_forward):
				image_aleatoric = doc.clone().detach()
             	#loop to add the noise
				for k in range(image_aleatoric.size()[0]):
					img = image_aleatoric[k,:,:,:].cpu().numpy()
					add_noise = A.augmentations.transforms.GaussNoise (var_limit=20, mean=0, per_channel=True, always_apply=False, p=1)
					image_aleatoric[k,:,:,:] = torch.tensor(add_noise(image=img)["image"]).to(device=device, dtype=torch.float32)

				with torch.no_grad():
					# predict
					mask_pred = net(doc)
					# concatenate prediction to the other made on the same batch
					dropout_predictions = torch.cat((dropout_predictions,mask_pred.cpu().softmax(dim=1).unsqueeze(dim=0)),dim=0)
					aleatoric_predictions = torch.cat((aleatoric_predictions,mask_pred.cpu().softmax(dim=1).unsqueeze(dim=0)),dim=0)

			# compute bayesian metrics
			batch_mean = dropout_predictions.mean(dim=0)
			batch_std = dropout_predictions.std(dim=0)
			batch_pred_entropy = -torch.sum(batch_mean*batch_mean.log(),dim=1)
			batch_mutual_info = batch_pred_entropy+torch.mean(torch.sum(dropout_predictions*dropout_predictions.log(),dim=-3), dim=0)
			entropy = batch_pred_entropy[j].cpu().numpy()
			mutual = batch_mutual_info[j].cpu().numpy()
			prediction = batch_mean[j].argmax(dim=0).cpu().numpy()

			#compute std to have aleatoric uncertainty
			aleatoric_std = aleatoric_predictions.std(dim=0)
			#compute the mean of std along the classes
			aleatoric = aleatoric_std.mean(dim=1)[j].cpu().numpy()

			# prepare uncertainity and accuracy maps
			mask_true = i[1]
			mask_pred_labels = batch_mean.argmax(dim=1)
			mask_true_labels = mask_true.argmax(dim=1)

			#compute the accuracy for each patch and check if it above the threshold
			masktrue_unfold = unfold(mask_true_labels.unsqueeze(dim=1).to(torch.float32))
			pred_unfold = unfold(mask_pred_labels.unsqueeze(dim=1).to(torch.float32))
			accuracy_matrix = torch.eq(pred_unfold, masktrue_unfold).to(torch.float32).mean(dim=1)
			bool_acc_matrix = torch.gt(accuracy_matrix, accuracy_tresh).to(torch.float32)

			# compute the mean uncertainty and if it is above the threshold
			uncertainty_matrix = unfold(batch_pred_entropy.unsqueeze(dim=1)).mean(dim=1)
			uncertainty_tresh = uncertainty_matrix.min()+t*(uncertainty_matrix.max()-uncertainty_matrix.min())
			bool_uncert_matrix = torch.gt(uncertainty_matrix, uncertainty_tresh).to(torch.float32)

			# fold uncertainity and accuracy matrices
			acc_expanded = bool_acc_matrix.view(bool_acc_matrix.size(0),1,bool_acc_matrix.size(1)).expand(bool_acc_matrix.size(0),w_size**2,bool_acc_matrix.size(1))
			uncert_expanded = bool_uncert_matrix.view(bool_uncert_matrix.size(0),1,bool_uncert_matrix.size(1)).expand(bool_uncert_matrix.size(0),w_size**2,bool_uncert_matrix.size(1))
			bin_acc_map = fold(acc_expanded)[j][0]
			bin_uncert_map = fold(uncert_expanded)[j][0]

			# create 
			bin_inacc_certain = (1-bin_acc_map)*(1-bin_uncert_map)


			# data visualization
			bbk = bbk.numpy()
			counter += 1
			document = i[0][j]
			rgbidsm = document[:5,:,:]*bbkd.std_vals_tiles+bbkd.mean_vals_tiles
			rgb = rgbidsm[:3,:,:].div(torch.max(rgbidsm[:3,:,:])).permute(1,2,0).numpy()
			ir = rgbidsm[3,:,:].numpy()
			dsm = rgbidsm[4,:,:].numpy()
			build = document[5,:,:].numpy()
			hoe = document[6,:,:]*bbkd.std_vals_vegetation+bbkd.mean_vals_vegetation
			hoe = hoe.numpy()

			# bayesian prediction image
			fig = plt.figure()
			plt.subplot(331)
			plt.imshow(rgb)
			# plt.imshow(bbk, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1), alpha=0.4)
			plt.axis('off')
			plt.title('RGB')
			plt.subplot(332)
			plt.imshow(bbk, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1), alpha=1)
			plt.axis('off')
			plt.title('BBK')
			plt.subplot(333)
			plt.imshow(prediction, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1), alpha=1)
			plt.axis('off')
			plt.title('Prediction')
			plt.subplot(334)
			# plt.imshow(entropy)
			plt.imshow(entropy, vmin=batch_pred_entropy.min().cpu().numpy(), vmax = batch_pred_entropy.max().cpu().numpy())
			plt.axis('off')
			plt.title('Entropy')
			plt.subplot(335)
			# plt.imshow(mutual)
			plt.imshow(mutual, vmin=batch_mutual_info.min().cpu().numpy(), vmax = batch_mutual_info.max().cpu().numpy())
			plt.axis('off')
			plt.title('Epistemic') # equal to mutual information

			plt.subplot(336)
			plt.imshow(aleatoric)
			plt.axis('off')
			plt.title('Aleatoric')

			plt.subplot(337)
			plt.imshow(bin_acc_map)
			plt.axis('off')
			plt.title('Accuracy (binary)')
			plt.subplot(338)
			plt.imshow(bin_uncert_map)
			plt.axis('off')
			plt.title('Uncertainity (binary)')

			plt.subplot(339)
			plt.imshow(bin_inacc_certain)
			plt.axis('off')
			plt.title('Inaccurate and certain') 

			plt.tight_layout()
			plt.savefig(f'example_data/example_pred_bayesian{counter}.png')
			plt.close()

			

			# # prediction image
			# fig = plt.figure()
			# plt.subplot(131)
			# plt.imshow(rgb)
			# # plt.imshow(bbk, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1), alpha=0.4)
			# plt.axis('off')
			# plt.title('RGB')
			# plt.subplot(132)
			# #plt.imshow(rgb)
			# plt.imshow(bbk, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1), alpha=1)
			# plt.axis('off')
			# plt.title('BBK')
			# plt.subplot(133)
			# #plt.imshow(rgb)
			# plt.imshow(prediction, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1), alpha=1)
			# plt.axis('off')
			# plt.title('Prediction')
			# plt.tight_layout()
			# plt.savefig(f'example_data/example_pred{counter}.png')
			# plt.close()

			# data image
			fig = plt.figure()
			plt.subplot(231)
			plt.imshow(rgb)
			# plt.imshow(bbk, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1), alpha=0.4)
			plt.axis('off')
			plt.title('RGB')
			plt.subplot(232)
			plt.imshow(ir, cmap='Reds', norm=colors.Normalize())
			plt.axis('off')
			plt.title('IR')
			plt.subplot(233)
			plt.imshow(dsm, norm=colors.Normalize())
			plt.axis('off')
			plt.title('DSM')
			plt.subplot(234)
			plt.imshow(build, cmap='Greys')
			plt.axis('off')
			plt.title('Build')
			plt.subplot(235)
			plt.imshow(hoe, cmap='Greens', norm=colors.Normalize())
			plt.axis('off')
			plt.title('HOE')
			plt.subplot(236)
			plt.imshow(bbk, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1))
			plt.axis('off')
			plt.title('BBK')
			plt.tight_layout()
			plt.savefig(f'example_data/example_data{counter}.png')
			plt.close()
print(f'number of resulting images : {counter}')
			