from utils.potsdam_dataset import PotsdamDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import numpy as np
from bayesian_unet import BayesianUNet
import os

# constants
W_SIZE = 4 # patch size to compute uncertainty performance metrics
NB_FORWARD = 20 # number of forward pass performed on the input
ACCURACY_THRESH = 0.5
UNCERTAINTY_THRESH = 0.4
CMAP = colors.ListedColormap(["b","r","y","g","m","w"])
CLASSES = ["impervious_surfaces", "buildings", "low_vegetaion", "tree", "car", "background"]
CMAPSCALE = [-0.5,0.5,1.5,2.5,3.5,4.5,5.5]

# dropout activation and deactivation
def enable_dropout(model):
	""" Function to enable the dropout layers during test-time """
	for m in model.modules():
		if m.__class__.__name__.startswith('Dropout'):
			m.train()

def disable_dropout(model):
	""" Function to enable the dropout layers during test-time """
	for m in model.modules():
		if m.__class__.__name__.startswith('Dropout'):
			m.eval()

def predict(net, inputs, targets=None, device="cpu", normalization_mean=0, normalization_std=1):
	if targets==None:
		usetargets = False
	else:
		usetargets = True
	doc = inputs.clone().detach().to(device=device)
	# to store n_forward predictions on the same batch
	dropout_predictions = torch.empty((0,targets.size(0),targets.size(1),targets.size(2),targets.size(3)))
	# bayesian inference
	for f_pass in range(NB_FORWARD):
		with torch.no_grad():
			# predict with dropout
			enable_dropout(net)
			mask_pred = net(doc)
			# concatenate prediction to the other made on the same batch
			dropout_predictions = torch.cat((dropout_predictions,mask_pred.cpu().softmax(dim=1).unsqueeze(dim=0)),dim=0)
	# compute bayesian metrics
	batch_mean = dropout_predictions.mean(dim=0)
	batch_std = dropout_predictions.std(dim=0)
	batch_pred_entropy = -torch.sum(batch_mean*batch_mean.log(),dim=1)
	batch_mutual_info = batch_pred_entropy+torch.mean(torch.sum(dropout_predictions*dropout_predictions.log(),dim=-3), dim=0)

	if usetargets:
		# prepare uncertainity and accuracy maps
		mask_true = targets.clone().detach()
		mask_pred_labels = batch_mean.argmax(dim=1)
		mask_true_labels = mask_true.argmax(dim=1)

		#compute the accuracy for each patch and check if it above the threshold
		masktrue_unfold = unfold(mask_true_labels.unsqueeze(dim=1).to(torch.float32))
		pred_unfold = unfold(mask_pred_labels.unsqueeze(dim=1).to(torch.float32))
		accuracy_matrix = torch.eq(pred_unfold, masktrue_unfold).to(torch.float32).mean(dim=1)
		bool_acc_matrix = torch.gt(accuracy_matrix, ACCURACY_THRESH).to(torch.float32)

		# compute the mean uncertainty and if it is above the threshold
		uncertainty_matrix = unfold(batch_pred_entropy.unsqueeze(dim=1)).mean(dim=1)
		uncertainty_tresh = uncertainty_matrix.min()+UNCERTAINTY_THRESH*(uncertainty_matrix.max()-uncertainty_matrix.min())
		bool_uncert_matrix = torch.gt(uncertainty_matrix, uncertainty_tresh).to(torch.float32)
		# fold uncertainity and accuracy matrices
		acc_expanded = bool_acc_matrix.view(bool_acc_matrix.size(0),1,bool_acc_matrix.size(1)).expand(bool_acc_matrix.size(0),W_SIZE**2,bool_acc_matrix.size(1))
		uncert_expanded = bool_uncert_matrix.view(bool_uncert_matrix.size(0),1,bool_uncert_matrix.size(1)).expand(bool_uncert_matrix.size(0),W_SIZE**2,bool_uncert_matrix.size(1))

	# create images of results for all processed inputs
	for j in range(targets.size(0)):
		entropy = batch_pred_entropy[j].cpu().numpy()
		mutual = batch_mutual_info[j].cpu().numpy()
		prediction = batch_mean[j].argmax(dim=0).cpu().numpy()

		bin_acc_map = fold(acc_expanded)[j][0]
		bin_uncert_map = fold(uncert_expanded)[j][0]

		# create binary coorective map
		bin_inacc_certain = (1-bin_acc_map)*(1-bin_uncert_map)

		# input visualization
		document = doc[j]
		rgbidsm = document[:5,:,:].cpu()*normalization_std+normalization_mean
		rgb = rgbidsm[:3,:,:].cpu().div(torch.max(rgbidsm[:3,:,:])).permute(1,2,0).numpy()
		ir = rgbidsm[3,:,:].cpu().numpy()
		dsm = rgbidsm[4,:,:].cpu().numpy()

		# data image
		fig = plt.figure()
		plt.subplot(131)
		plt.imshow(rgb)
		plt.axis('off')
		plt.title('RGB')
		plt.subplot(132)
		plt.imshow(ir, cmap='Reds', norm=colors.Normalize())
		plt.axis('off')
		plt.title('IR')
		plt.subplot(133)
		plt.imshow(dsm, norm=colors.Normalize())
		plt.axis('off')
		plt.title('DSM')
		plt.tight_layout()
		plt.savefig(f'predictions/prediction{j}_input.png', bbox_inches='tight', pad_inches=0)
		plt.close()

		# prediction image
		fig = plt.figure()
		plt.subplot(131)
		plt.imshow(prediction, cmap=CMAP, norm=colors.BoundaryNorm(CMAPSCALE, len(CMAPSCALE)-1), alpha=1)
		plt.axis('off')
		plt.title('Prediction')
		plt.subplot(132)
		plt.imshow(entropy, vmin=batch_pred_entropy.min().cpu().numpy(), vmax = batch_pred_entropy.max().cpu().numpy())
		plt.axis('off')
		plt.title('Predictive Entropy')
		plt.subplot(133)
		plt.imshow(mutual, vmin=batch_mutual_info.min().cpu().numpy(), vmax = batch_mutual_info.max().cpu().numpy())
		plt.axis('off')
		plt.title('Epistemic Uncertainty')
		plt.tight_layout()
		plt.savefig(f'predictions/prediction{j}_output.png', bbox_inches='tight', pad_inches=0)

		if usetargets:
			# binary maps
			fig = plt.figure()
			plt.subplot(231)
			plt.imshow(mask_true[j].argmax(dim=0).numpy(), cmap=CMAP, norm=colors.BoundaryNorm(CMAPSCALE, len(CMAPSCALE)-1), alpha=1)
			plt.axis('off')
			plt.title('Ground truth')
			plt.subplot(232)
			plt.imshow(prediction, cmap=CMAP, norm=colors.BoundaryNorm(CMAPSCALE, len(CMAPSCALE)-1), alpha=1)
			plt.axis('off')
			plt.title('Prediction')
			plt.subplot(234)
			plt.imshow(bin_acc_map)
			plt.axis('off')
			plt.title('Accuracy (binary)')
			plt.subplot(235)
			plt.imshow(bin_uncert_map)
			plt.axis('off')
			plt.title('Uncertainty (binary)')
			plt.subplot(236)
			plt.imshow(bin_inacc_certain)
			plt.axis('off')
			plt.title('Inaccurate and certain') 
			plt.tight_layout()
			plt.savefig(f'predictions/prediction{j}_binarymaps.png', bbox_inches='tight', pad_inches=0)
			plt.close()




if __name__ == "__main__":
	# device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# create repo for predictions if needed
	if not os.path.exists("./predictions"):
		os.mkdir("./predictions")

	# Get data to evaluate
	test_set = PotsdamDataset(split = "test", random_seed = 1, augment=False)
	dl = DataLoader(test_set, shuffle=True, batch_size=16)
	inputs, targets = next(iter(dl))

	# Declare model
	net = BayesianUNet(n_channels=test_set.N_CHANNELS, n_classes=test_set.N_CLASSES, bilinear=False).to(device=device)
	# Choose the trained parameters to load in the model
	checkpoint_path = 'checkpoints/checkpoint_epoch5.pth'
	net.load_state_dict(torch.load(checkpoint_path, map_location=device))
	net.eval()

	unfold = torch.nn.Unfold(kernel_size=(W_SIZE, W_SIZE),stride = W_SIZE)
	fold = torch.nn.Fold(output_size=(inputs.size(2),inputs.size(3)), kernel_size=(W_SIZE, W_SIZE), stride=W_SIZE)

	predict(net=net, inputs=inputs, targets=targets, device=device, normalization_mean=test_set.mean_vals_tiles, normalization_std=test_set.std_vals_tiles)



"""# Labels and colors
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
plt.savefig(f'example_data/bbk_legend.svg', bbox_inches='tight', pad_inches=0)"""

# unfolding and folding
"""W_SIZE = 4 #patch size
ACCURACY_THRESH = 0.5
t = 0.4 # default 0.4
unfold = torch.nn.Unfold(kernel_size=(W_SIZE, W_SIZE),stride = W_SIZE)
fold = torch.nn.Fold(output_size=(200,200), kernel_size=(W_SIZE, W_SIZE), stride=W_SIZE)

counter = 0
wanted = [1,2,4,6,10]


for i in dl:
	for j in range(len(i[0])):
		bbk = i[1][j].argmax(dim=0)
		if torch.unique(bbk[bbk!=0], return_counts=False).size(0) >= 7:
			counter +=1
			if counter in wanted:
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
						
						change_brightContrast = A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1)
						change_hueSaturation = A.augmentations.transforms.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, p=1)
						
						# Add augmentation changing contrast, saturation and brightness
						image_temp = change_hueSaturation(image=np.transpose(img[:3,:,:],axes=(1,2,0)))['image']
						image_aleatoric[k,:3,:,:] = torch.tensor(change_brightContrast(image=np.transpose(image_temp,axes=(2,1,0)))['image']).to(device=device, dtype=torch.float32)

					with torch.no_grad():
						# predict with dropout
						enable_dropout(net)
						mask_pred = net(doc)
						# concatenate prediction to the other made on the same batch
						dropout_predictions = torch.cat((dropout_predictions,mask_pred.cpu().softmax(dim=1).unsqueeze(dim=0)),dim=0)
						# predict noisy image
						disable_dropout(net)
						mask_pred = net(image_aleatoric)
						# concatenate prediction to the other made on the same batch
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
				batch_aletoric_std_mean = aleatoric_std.mean(dim=1)
				aleatoric = batch_aletoric_std_mean[j].cpu().numpy()

				# prepare uncertainity and accuracy maps
				mask_true = i[1]
				mask_pred_labels = batch_mean.argmax(dim=1)
				mask_true_labels = mask_true.argmax(dim=1)

				#compute the accuracy for each patch and check if it above the threshold
				masktrue_unfold = unfold(mask_true_labels.unsqueeze(dim=1).to(torch.float32))
				pred_unfold = unfold(mask_pred_labels.unsqueeze(dim=1).to(torch.float32))
				accuracy_matrix = torch.eq(pred_unfold, masktrue_unfold).to(torch.float32).mean(dim=1)
				bool_acc_matrix = torch.gt(accuracy_matrix, ACCURACY_THRESH).to(torch.float32)

				# compute the mean uncertainty and if it is above the threshold
				uncertainty_matrix = unfold(batch_pred_entropy.unsqueeze(dim=1)).mean(dim=1)
				uncertainty_tresh = uncertainty_matrix.min()+t*(uncertainty_matrix.max()-uncertainty_matrix.min())
				bool_uncert_matrix = torch.gt(uncertainty_matrix, uncertainty_tresh).to(torch.float32)

				# fold uncertainity and accuracy matrices
				acc_expanded = bool_acc_matrix.view(bool_acc_matrix.size(0),1,bool_acc_matrix.size(1)).expand(bool_acc_matrix.size(0),W_SIZE**2,bool_acc_matrix.size(1))
				uncert_expanded = bool_uncert_matrix.view(bool_uncert_matrix.size(0),1,bool_uncert_matrix.size(1)).expand(bool_uncert_matrix.size(0),W_SIZE**2,bool_uncert_matrix.size(1))
				bin_acc_map = fold(acc_expanded)[j][0]
				bin_uncert_map = fold(uncert_expanded)[j][0]

				# create binary coorective map
				bin_inacc_certain = (1-bin_acc_map)*(1-bin_uncert_map)

				# save maps as pngs
				plt.imsave(f"example_report/entropy{counter}.png",entropy)
				plt.imsave(f"example_report/mutual{counter}.png",mutual)
				plt.imsave(f"example_report/aleatoric{counter}.png",aleatoric)
				plt.imsave(f"example_report/bin_acc_map{counter}.png",bin_acc_map)
				plt.imsave(f"example_report/bin_uncert_map{counter}.png",bin_uncert_map)
				plt.imsave(f"example_report/bin_inacc_certain{counter}.png",bin_inacc_certain)


				# data visualization
				bbk = bbk.numpy()
				document = i[0][j]
				rgbidsm = document[:5,:,:]*bbkd.std_vals_tiles+bbkd.mean_vals_tiles
				rgb = rgbidsm[:3,:,:].div(torch.max(rgbidsm[:3,:,:])).permute(1,2,0).numpy()
				ir = rgbidsm[3,:,:].numpy()
				dsm = rgbidsm[4,:,:].numpy()
				build = document[5,:,:].numpy()
				hoe = document[6,:,:]*bbkd.std_vals_vegetation+bbkd.mean_vals_vegetation
				hoe = hoe.numpy()

				# prediction image
				fig = plt.figure()
				plt.subplot(131)
				plt.imshow(rgb)
				# plt.imshow(bbk, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1), alpha=0.4)
				plt.axis('off')
				plt.title('RGB')
				plt.subplot(132)
				plt.imshow(bbk, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1), alpha=1)
				plt.axis('off')
				plt.title('BBK')
				plt.subplot(133)
				plt.imshow(prediction, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1), alpha=1)
				plt.axis('off')
				plt.title('Prediction')
				plt.tight_layout()
				plt.savefig(f'example_report/example_pred{counter}.png',bbox_inches='tight', pad_inches=0)
				plt.close()


				# bayesian uncertainty
				fig = plt.figure()
				plt.subplot(121)
				plt.imshow(entropy, vmin=batch_pred_entropy.min().cpu().numpy(), vmax = batch_pred_entropy.max().cpu().numpy())
				plt.axis('off')
				plt.title('Entropy')
				plt.subplot(122)
				plt.imshow(mutual, vmin=batch_mutual_info.min().cpu().numpy(), vmax = batch_mutual_info.max().cpu().numpy())
				plt.axis('off')
				plt.title('Epistemic uncertainty')
				plt.tight_layout()
				plt.savefig(f'example_report/example_uncertainty{counter}.png', bbox_inches='tight', pad_inches=0)

				# aleatoric
				fig = plt.figure()
				plt.imshow(aleatoric, vmin=batch_aletoric_std_mean.min().cpu().numpy(), vmax = batch_aletoric_std_mean.max().cpu().numpy())
				plt.axis('off')
				plt.title('Aleatoric uncertainty')
				plt.tight_layout()
				plt.savefig(f'example_report/example_aleatoric{counter}.png', bbox_inches='tight', pad_inches=0)
				plt.close()

				# binary maps
				fig = plt.figure()
				plt.subplot(121)
				plt.imshow(bin_acc_map)
				plt.axis('off')
				plt.title('Accuracy (binary)')
				plt.subplot(122)
				plt.imshow(bin_uncert_map)
				plt.axis('off')
				plt.title('Uncertainty (binary)')
				plt.tight_layout()
				plt.savefig(f'example_report/example_binary_maps{counter}.png', bbox_inches='tight', pad_inches=0)
				plt.close()

				# correction map
				fig = plt.figure()
				plt.subplot(131)
				plt.imshow(bbk, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1))
				plt.axis('off')
				plt.title('BBK')
				plt.subplot(132)
				plt.imshow(prediction, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1), alpha=1)
				plt.axis('off')
				plt.title('Prediction')
				plt.subplot(133)
				plt.imshow(bin_inacc_certain)
				plt.axis('off')
				plt.title('Inaccurate and certain') 
				plt.tight_layout()
				plt.savefig(f'example_report/example_corrective{counter}.png', bbox_inches='tight', pad_inches=0)
				plt.close()

				# wrong prediction map
				fig = plt.figure()
				plt.subplot(131)
				plt.imshow(bbk, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1))
				plt.axis('off')
				plt.title('BBK')
				plt.subplot(132)
				plt.imshow(prediction, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1), alpha=1)
				plt.axis('off')
				plt.title('Prediction')
				plt.subplot(133)
				plt.imshow(mutual, vmin=batch_mutual_info.min().cpu().numpy(), vmax = batch_mutual_info.max().cpu().numpy())
				plt.axis('off')
				plt.title('Epistemic uncertainty')
				plt.tight_layout()
				plt.savefig(f'example_report/example_wrongpreds{counter}.png', bbox_inches='tight', pad_inches=0)
				plt.close()

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
				plt.title('Vegetation')
				plt.subplot(236)
				plt.imshow(bbk, cmap=bbk_cmap, norm=colors.BoundaryNorm(bbk_scale, len(bbk_scale)-1))
				plt.axis('off')
				plt.title('BBK')
				plt.tight_layout()
				plt.savefig(f'example_report/example_data{counter}.png', bbox_inches='tight', pad_inches=0)
				plt.close()
print(f'number of resulting images : {counter}')"""