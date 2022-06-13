## Importing packages
from skimage.io import imread, imsave
from pathlib import Path
import os
from os.path import exists
import torch
from tqdm import tqdm

ROOT = "./Potsdam_data/"
FOLDERS = ["5_Labels_all/","4_Ortho_RGBIR/","1_DSM/"]
TILESDIR = "tiles/"
LABELS = [
	"impervious_surfaces",
	"buildings", 
	"low_vegetaion", 
	"tree", 
	"car", 
	"background"
	]
LABELS_RGB = [
	torch.tensor((255,255,255)),
	torch.tensor((0,0,255)),
	torch.tensor((0,255,255)),
	torch.tensor((0,255,0)),
	torch.tensor((255,255,0)),
	torch.tensor((255,0,0))
	]
TSIZE = 400

if __name__ == "__main__":
	## Creating directory to store tiles
	if not exists(ROOT+TILESDIR):
		os.mkdir(ROOT+TILESDIR)
		os.mkdir(ROOT+TILESDIR+"input/")
		os.mkdir(ROOT+TILESDIR+"target/")
	## Creating a list of 6000 x 6000 large tiles available
	largetiles = []
	for elem in os.listdir(ROOT+FOLDERS[0]):
		largetiles.append(elem[:-10])
	## Making tiles
	tilenum = 0
	for lt in tqdm(largetiles):
		## Opening large tiles
		lt_label = torch.tensor(imread(ROOT+FOLDERS[0]+lt+"_label.tif"))
		lt_RGBIR = torch.tensor(imread(ROOT+FOLDERS[1]+lt+"_RGBIR.tif"))
		## to cope with inconsistent dsm filenames wrt to the other ones
		try:
			lt_DSM = torch.tensor(imread(ROOT+FOLDERS[2]+"dsm"+lt[3:-4]+"0"+lt[-4:]+".tif"))
		except:
			lt_DSM = torch.tensor(imread(ROOT+FOLDERS[2]+"dsm"+lt[3:-3]+"0"+lt[-3]+"_0"+lt[-1]+".tif"))
		## Encoding labels on one dimension
		lt_label_onedim = torch.zeros(lt_label.size(0),lt_label.size(1))
		for i in range(len(LABELS_RGB)):
			mask = torch.all(lt_label==LABELS_RGB[i],dim=2)
			lt_label_onedim[mask]=i
		## Making small tiles
		for i in range(0,lt_label.size(0),TSIZE):
			for j in range(0,lt_label.size(1),TSIZE):
				try:
					st_label = lt_label_onedim[i:i+TSIZE,j:j+TSIZE]
					imsave(ROOT+TILESDIR+"target/"+f"tile{tilenum}.tif",st_label.numpy(),check_contrast=False)
					st_document = torch.cat([lt_RGBIR[i:i+TSIZE,j:j+TSIZE],lt_DSM[i:i+TSIZE,j:j+TSIZE].unsqueeze(dim=2)],dim=2)
					imsave(ROOT+TILESDIR+"input/"+f"tile{tilenum}.tif",st_document.permute(2,0,1).numpy(),check_contrast=False)
					tilenum+=1
				except:
					pass