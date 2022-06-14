## Importing packages
import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from bayesian_unet import BayesianUNet
from evaluate import evaluate
from utils.metrics import dice_loss
from utils.potsdam_dataset import PotsdamDataset

dir_checkpoint = Path('./checkpoints/')

def train_net(net,
			  train_set, 
			  val_set,
			  optim_class,
			  device,
			  epochs: int = 5,
			  batch_size: int = 32,
			  learning_rate: float = 1e-5,
			  weight_decay = 0,
			  momentum = 0.9,
			  patience: int = 2,
			  save_checkpoint: bool = True,
			  amp: bool = False,
			  ):

	# Create data loaders
	train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
	val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size)

	# Initialize logging 
	experiment = wandb.init(project='Bayesian UNet', resume='allow', anonymous='must')
	experiment.config.update(dict(
								epochs=epochs,
								optim_class=optim_class,
								batch_size=batch_size,
								learning_rate=learning_rate,
								patience = patience,
								weight_decay=weight_decay,
								momentum=momentum,
								save_checkpoint=save_checkpoint,
								amp=amp,
								allow_val_change=True)
								)
	n_val = len(val_set)
	n_train = len(train_set)

	logging.info(f'''Starting training:
		Epochs:		  {epochs}
		Optimizer:	   {optim_class}
		Batch size:	  {batch_size}
		Learning rate:   {learning_rate}
		Patience of learning rate: {patience}
		Weight  decay:   {weight_decay}
		Momentum:		{momentum}
		Training size:   {n_train}
		Validation size: {n_val}
		Checkpoints:	 {save_checkpoint}
		Device:		  {device.type}
		Mixed Precision: {amp}
	''')

	# Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
	try:
		optimizer = optim_class(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
	except:
		optimizer = optim_class(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
		
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience)  # goal: maximize Dice score
	grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
	criterion = nn.CrossEntropyLoss()
	global_step = 0

	# Begin training
	for epoch in range(epochs):
		net.train()
		epoch_loss = 0
		with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
			for batch in train_loader:
				images = batch[0]
				true_masks = batch[1]

				assert images.shape[1] == net.n_channels, \
					f'Network has been defined with {net.n_channels} input channels, ' \
					f'but loaded images have {images.shape[1]} channels. Please check that ' \
					'the images are loaded correctly.'

				images = images.to(device=device, dtype=torch.float32)
				true_masks = true_masks.to(device=device, dtype=torch.float32)

				with torch.cuda.amp.autocast(enabled=amp):
					masks_pred = net(images)
					loss = criterion(masks_pred, true_masks) \
						   + dice_loss(F.softmax(masks_pred, dim=1).float(),
									   true_masks,
									   multiclass=True)

				optimizer.zero_grad(set_to_none=True)
				grad_scaler.scale(loss).backward()
				grad_scaler.step(optimizer)
				grad_scaler.update()

				pbar.update(images.shape[0])
				global_step += 1
				epoch_loss += loss.item()
				experiment.log({
					'train loss': loss.item(),
					'step': global_step,
					'epoch': epoch
				})
				pbar.set_postfix(**{'loss (batch)': loss.item()})

				# Evaluation round
				division_step = (n_train // (10 * batch_size))
				if division_step > 0:
					if global_step % division_step == 0:

						val_score, accuracy_score, accuracy_per_class, F1_score, cf_matrix = evaluate(net, val_loader, device)
						
						#Update the scheduler
						scheduler.step(val_score)

						logging.info('Validation accuracy score : {}'.format(accuracy_score))
						logging.info('Validation Dice score: {}'.format(val_score))

						# create wandb objects for visualisation
						# confusion matrix
						plt.figure()
						sns.heatmap(cf_matrix, annot=True, annot_kws={"size":8}, fmt='.2%', cmap='Blues', cbar=True, xticklabels=val_set.CLASSES_list,yticklabels=val_set.CLASSES_list)
						plt.tight_layout()

						# scores table
						class_labels = {0 : "impervious_surfaces",
										1 : "buildings",
										2 : "low_vegetation",
										3 : "tree",
										4 : "car",
										5 : "background",}
						scores = {
								'Accuracy': accuracy_per_class,
								'F1 score' : F1_score
								}
						columns_table= list(class_labels.values())
						data_table = [accuracy_per_class, F1_score]
						score_table = wandb.Table(data = data_table, columns=columns_table)
						score_table.add_column(name='score',data=list(scores.keys()))
						
						#insert these metrics and objects in wandb
						experiment.log({
							'Learning rate': optimizer.param_groups[0]['lr'],
							'Validation Dice score': val_score,
							'Global accuracy score': accuracy_score,
							#'Metric per class':score_table, 
							'images': wandb.Image(images[0][:3].cpu()
													),
							'masks': {
								'true': wandb.Image(torch.softmax(true_masks, dim=1).argmax(dim=1)[0].float().cpu()),
								'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
							},
							'step': global_step,
							'epoch': epoch,
							'conf_mat' : wandb.Image(plt),
						})
						plt.close()

		if save_checkpoint:
			Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
			torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
			logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
	parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
	parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
	parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
	parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-2,
						help='Learning rate', dest='lr')
	parser.add_argument('--patience', '-p', metavar='P', type=int, default=50,
						help='LR Scheduler Patience', dest='patience')
	parser.add_argument('--weight-decay', '-wd', metavar='WD', type=float, default=0,
						help='Weight Decay of Optimizer', dest='weight_decay')
	parser.add_argument('--momentum', '-m', metavar='M', type=float, default=0,
						help='Momentum of Optimizer', dest='momentum')
	parser.add_argument('--optimizer', '-o', metavar='O', type=str, default="SGD",
						help='Optimizer : SGD or RMS', dest='optimizer')
	parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
	parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
	parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
	parser.add_argument('--augment', action='store_true', default=True, help='Add data augmentation')

	return parser.parse_args()


if __name__ == '__main__':
	args = get_args()

	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logging.info(f'Using device {device}')

	# Create datasets
	train_set = PotsdamDataset(split = "train", random_seed = 1, augment=True)
	val_set = PotsdamDataset(split = "val", random_seed = 1, augment=True)

	# Declare model
	net = BayesianUNet(n_channels=train_set.N_CHANNELS, n_classes=train_set.N_CLASSES, bilinear=args.bilinear)

	# Choose optimizer
	optims = {"SGD" : optim.SGD, "RMS" : optim.RMSprop}
	optim_class = optims.get(args.optimizer,"Invalid optimizer input")

	logging.info(f'Network:\n'
				 f'\t{net.n_channels} input channels\n'
				 f'\t{net.n_classes} output channels (classes)\n'
				 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

	if args.load:
		net.load_state_dict(torch.load(args.load, map_location=device))
		logging.info(f'Model loaded from {args.load}')

	net.to(device=device)
	try:
		logging.info('Training model')
		train_net(net=net,
				  val_set=val_set,
				  train_set=train_set,
				  optim_class = optim_class,
				  epochs=args.epochs,
				  batch_size=args.batch_size,
				  learning_rate=args.lr,
				  weight_decay=args.weight_decay,
				  momentum=args.momentum,
				  patience=args.patience,
				  device=device,
				  amp=args.amp)
		logging.info('Evaluating trained model')
		#scores = evaluate(net=net,dataloader=DataLoader(val_set, shuffle=False, batch_size=batch_size),device=device)
		#print(scores)
	except KeyboardInterrupt:
		torch.save(net.state_dict(), 'INTERRUPTED.pth')
		logging.info('Saved interrupt')
		sys.exit(0)
