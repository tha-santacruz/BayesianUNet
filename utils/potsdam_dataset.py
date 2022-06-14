## Importing packages
import os
import random

import torch
from skimage.io import imread
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


class PotsdamDataset:
	def __init__(self, split: str = "train", random_seed = 1, augment: bool = True):
		"""Initialization method"""
		# Constants
		self.ROOT = "./Potsdam_data/tiles/"
		self.SPLIT_SIZE = {"train": 0.7, "test": 0.2, "val": 0.1}
		self.CLASSES_list = [
			"impervious_surfaces",
			"buildings", 
			"low_vegetaion", 
			"tree", 
			"car", 
			"background"
			]
		self.CLASSES_dict = {
			0 : "impervious_surfaces", 
			1 : "buildings", 
			2 : "low_vegetaion", 
			3 : "tree", 
			4 : "car", 
			5 : "background"}
		self.N_CHANNELS = 5
		self.N_CLASSES = 6
		
		# Available tiles list
		self.tileslist = os.listdir(self.ROOT+"input/")
		self.valid_tiles = [] # to store valid tiles
		self.mean_vals_tiles = torch.zeros((5,0)) # to store stat values of images for normalization
		self.std_vals_tiles = torch.zeros((5,0))

		# For each tile
		for tile in tqdm(self.tileslist):
			valid = True
			file_content = torch.tensor(imread(self.ROOT+"input/"+tile)).permute(1,2,0)
			# checking shape
			if file_content.size()[2] != 5:
				valid = False
				#raise ValueError("Invalid data shape for target "+tile)
			# If valid, compute mean and std for normalization
			else:
				F.relu(file_content, inplace=True)
				m = file_content.mean(dim=[0,1])
				s = file_content.std(dim=[0,1])
				if any(torch.isnan(m)) | any(torch.isnan(s)):
					valid = False
					#raise ValueError("NAN values in target "+tile)
				else:
					self.mean_vals_tiles = torch.cat((self.mean_vals_tiles,m.unsqueeze(dim=1)),dim=1)
					self.std_vals_tiles = torch.cat((self.std_vals_tiles,s.unsqueeze(dim=1)),dim=1)
			if valid:
					self.valid_tiles.append(tile)

		# Average standardization vectors
		self.mean_vals_tiles = self.mean_vals_tiles.mean(dim=1).unsqueeze(dim=1).unsqueeze(dim=2)
		self.std_vals_tiles = self.std_vals_tiles.mean(dim=1).unsqueeze(dim=1).unsqueeze(dim=2)
		
		# Define how much samples will be available 
		self.split_counts = [int(len(self.valid_tiles)*self.SPLIT_SIZE["train"])]
		self.split_counts.append(int(len(self.valid_tiles)*self.SPLIT_SIZE["test"]))
		random.Random(random_seed).shuffle(self.valid_tiles)
		# Make split
		if split == "train":
			self.split_tiles = self.valid_tiles[:self.split_counts[0]]
		elif split == "test":
			self.split_tiles = self.valid_tiles[self.split_counts[0]:self.split_counts[0]+self.split_counts[1]]
		elif split == "val":
			self.split_tiles = self.valid_tiles[self.split_counts[0]+self.split_counts[1]:]
		elif split =="all":
			self.split_tiles = self.valid_tiles
		else:
			raise ValueError("Invalid split : values can be 'train', 'test', 'val' or 'all'")
		self.split = split

		# Augmentation methods
		self.augment = augment
		# Define transforms for augmentation of rgbir channels
		# Noise percent
		percent_noise = 0.1
		# Noise std for R-G-B-IR-DSM channels
		self.noise_std = self.std_vals_tiles[:5].squeeze()*percent_noise

	def __len__(self):
		"""returns length of valid tiles set"""
		return len(self.split_tiles)

	def __getitem__(self, idx):
		"""Generates multiband image and label for the coordinates of given indexes"""
		tile = self.split_tiles[idx]
		## Create document
		image= torch.tensor(imread(self.ROOT+"input/"+tile))
		# Padding if needed
		#if image.size()[-2:] != (200,200):
		#	image = nn.ConstantPad2d((0, 200-image.size()[2], 0, 200-image.size()[1]), 0)(image)
		# Replace negative values by 0 using ReLU
		F.relu(image, inplace=True)
		# Augment if requested
		if self.augment:
			for channel in range(len(self.noise_std)):
				image[channel] = image[channel] + torch.normal(mean=0, std=self.noise_std[channel], size=(200,200))
				F.relu(image, inplace=True)
		document = (image-self.mean_vals_tiles).div(self.std_vals_tiles)
		## Create label 
		image= torch.tensor(imread(self.ROOT+"target/"+tile))
		# Reshape to 3D
		if len(image.size())<3:
			image = image.unsqueeze(dim=2)
		image = image.permute(2,0,1)
		# Padding if needed
		if image.size()[-2:] != (200,200):
			image = nn.ConstantPad2d((0, 200-image.size()[2], 0, 200-image.size()[1]), 0)(image)
		# Replace negative values by 0 using ReLU
		F.relu(image, inplace=True)
		# To one hot encoding
		image = F.one_hot(image[0,:,:].to(torch.int64), num_classes = len(self.CLASSES_list))
		image = image.permute(2,0,1)
		label = image.to(torch.float32)

		return document, label
