## Importing packages
from skimage.io import imread
import os
from os.path import exists
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import random

class BBKDataset:
    def __init__(self, zone: tuple = ("all",), split: str = "train", buildings: bool = True, vegetation: bool = True, random_seed = 1):
        """Initialization method"""
        # Constants
        self.ROOT = "./../data/"
        self.SPLIT_SIZE = {"train": 0.7, "test": 0.2, "val": 0.1}
        self.BBK_CLASSES = ["null","wooded_area", "water", "bushes", "individual_tree", "no_woodland", "ruderal_area", 
        "without_vegetation", "buildings"]

        # Handle data coverage zone
        self.zone = []
        if isinstance(zone, tuple):
            if any("ticino" in el for el in zone):
                self.zone.append("ticino/")
            if any("genf" in el for el in zone):
                self.zone.append("genf/")
            if any("goesch" in el for el in zone):
                self.zone.append("goesch/")
            if any("jura" in el for el in zone):
                self.zone.append("jura/")
            if any("wallis" in el for el in zone):
                self.zone.append("wallis/")
            if any("all" in el for el in zone):
                if len(zone)>1:
                    raise ValueError("zone cannot contain either 'all' or other zones")
                else:
                    self.zone = ["ticino/","genf/","goesch/","jura/","wallis/"]
        else:
            raise ValueError("zone argument must be a tuple if set")

        # Handle required data types   
        self.folders = ["BBK_50m/","tile_50m/"]
        self.appendix = ["_bbk.tif",".tif"]
        if buildings == True:
            self.folders.append("build_50m/")
            self.appendix.append("_build.tif")
        if vegetation == True:
            self.folders.append("hoe_50m/")
            self.appendix.append("_hoe.tif")
            self.mean_vals_vegetation = torch.zeros((0))
            self.std_vals_vegetation = torch.zeros((0))
        
        # Find tiles with all datatypes available, based on the bbk tiles coordinates
        self.coordinates = [] # to store valid coordinates
        self.rejected = [] # to store invalid coordinates and reported issue
        self.global_max = torch.tensor([0, 0, 0, 0]) # to store max values of images for normalization
        self.mean_vals_tiles = torch.zeros((5,0))
        self.std_vals_tiles = torch.zeros((5,0))

        # For each region of origin of data
        for z in self.zone:
            # Getting all candidate tiles (coordinates) from BBK (as it will be always used)
            candidates = [elem[:13] for elem in os.listdir(self.ROOT+z+self.folders[0]) if ".tif"==elem[-4:]]
            # Testing whether candidates are valid for training and testing
            for couple in tqdm(candidates, desc = "Processing data for zone " + z[:-1]+ " {}/{}".format(self.zone.index(z)+1,len(self.zone))):
                reason = [] # to store potentially reported issues
                valid = True
                for f in range(len(self.folders)):
                    # Check if data is available for each data source
                    if not exists(self.ROOT+z+self.folders[f]+couple+self.appendix[f]):
                        # Reject coordinates and append missing data type
                        valid = False
                        reason.append("missing data for "+self.folders[f])
                    # Check number of channels for multiband images
                    elif self.folders[f] == "tile_50m/":
                        file_content = torch.tensor(imread(self.ROOT+z+self.folders[f]+couple+self.appendix[f]))
                        if file_content.size()[2] != 5:
                            valid = False
                            reason.append("invalid data shape for tile_50m/")
                        # If valid, compute mean and std
                        else:
                            F.relu(file_content, inplace=True)
                            m = file_content.mean(dim=[0,1])
                            self.mean_vals_tiles = torch.cat((self.mean_vals_tiles,m.unsqueeze(dim=1)),dim=1)
                            s = file_content.std(dim=[0,1])
                            self.std_vals_tiles = torch.cat((self.std_vals_tiles,s.unsqueeze(dim=1)),dim=1)
                    elif self.folders[f] == "hoe_50m/":
                        file_content = torch.tensor(imread(self.ROOT+z+self.folders[f]+couple+self.appendix[f]))
                        F.relu(file_content, inplace=True)
                        m = file_content.mean()
                        self.mean_vals_vegetation = torch.cat((self.mean_vals_vegetation,m.unsqueeze(dim=0)),dim=0)
                        s = file_content.std()
                        self.std_vals_vegetation = torch.cat((self.std_vals_vegetation,s.unsqueeze(dim=0)),dim=0)
                            
                if valid:
                    self.coordinates.append([couple, z])
                else:
                    self.rejected.append([couple, z, reason])

        # Average standardization vectors
        self.mean_vals_tiles = self.mean_vals_tiles.mean(dim=1).unsqueeze(dim=1).unsqueeze(dim=2)
        self.std_vals_tiles = self.std_vals_tiles.mean(dim=1).unsqueeze(dim=1).unsqueeze(dim=2)
        if vegetation == True:
            self.mean_vals_vegetation = self.mean_vals_vegetation.mean()
            self.std_vals_vegetation = self.std_vals_vegetation.mean()
        
        
        # Define how much samples will be available 
        self.split_counts = [int(len(self.coordinates)*self.SPLIT_SIZE["train"])]
        self.split_counts.append(int(len(self.coordinates)*self.SPLIT_SIZE["test"]))
        random.Random(random_seed).shuffle(self.coordinates)
        # Make split
        if split == "train":
            self.split_coordinates = self.coordinates[:self.split_counts[0]]
        elif split == "test":
            self.split_coordinates = self.coordinates[self.split_counts[0]:self.split_counts[1]]
        elif split == "val":
            self.split_coordinates = self.coordinates[self.split_counts[0]+self.split_counts[1]:]
        else:
            raise ValueError("Invalid split : values can be 'train', 'test' or 'val'")
        

    def __len__(self):
        """returns length of valid tiles set"""
        return len(self.split_coordinates)

    def __getitem__(self, idx):
        """Generates multiband image and label for the coordinates of given indexes"""
        couple, z = self.coordinates[idx]
        doc= torch.empty((0,200,200))
        for f in range(len(self.folders)):
            # Open image
            image = torch.tensor(imread(self.ROOT+z+self.folders[f]+couple+self.appendix[f]))
            # Reshape to 3D
            if len(image.size())<3:
                image = image.unsqueeze(dim=2)
            image = image.permute(2,0,1)
            # Padding if needed
            if image.size()[-2:] != (200,200):
                image = nn.ConstantPad2d((0, 200-image.size()[2], 0, 200-image.size()[1]), 0)(image)
            # Replace negative values by 0 using ReLU
            F.relu(image, inplace=True)
            # Create label
            if self.folders[f] == "BBK_50m/":
                non_null = image.detach().clone()
                non_null[non_null>0] = 1
                # To one hot encoding
                image = F.one_hot(image[0,:,:].to(torch.int64), num_classes = len(self.BBK_CLASSES))
                image = image.permute(2,0,1)
                label = image.to(torch.float32)
            # Create document channel
            else:
                # Standardize using mean and std valiues
                if self.folders[f] == "tile_50m/":
                    image = (image-self.mean_vals_tiles).div(self.std_vals_tiles)
                elif self.folders[f] == "hoe_50m/":
                    image = (image-self.mean_vals_vegetation).div(self.std_vals_vegetation)
                    pass
                # Remove buildings in the "Null" class zones
                elif self.folders[f] == "build_50m/":
                    image = non_null*image
                doc = torch.cat((doc, image), dim=0)
                doc = doc.to(torch.float32)
        return doc, label

    def reportissues(self):
        """Provides information about discarded data sources"""
        for issue in self.rejected:
            print(issue)
