# Pytorch Bayesian UNet model for segmentation and uncertainty prediction 

This repository provides the well-known UNet model [[1]](#1) converted to a Bayesian UNet model.
This model has been coded using Pytorch. This code is a modified version of the original the original implementation of UNet model in Pytorch by milesial (https://github.com/milesial/Pytorch-UNet).
The model has been implemented using the Monte Carlo Dropout method [[2]](#2).
It consists of adding a dropout layer at the end of each convolution layer, which is used both during training and testing times.
This Bayesian model provides different scores (entropy and mutual information) that characterize uncertainty in predictions. 

## To set up the Python environment : 
1.	If you use conda, execute ```conda env create --name envname --file=environment.yml```. If you have a ResolvePackageNotFound error, edit the environment.yml file to place the mentionned packages under the pip section.
2.	If you use pip, execute ```pip3 install -r requirements.txt```
3.	Install pytorch by following the instructions of the Pytorch documentation : 		https://pytorch.org/get-started/locally/

## To create tiles : 
1.	Create a "Potsdam_data/" directory in the BayesianUNet directory.
2.	Download the Potsdam Dataset (International Society for Photogrammetry and Remote Sensing, 2022) from this URL:
	https://www.isprs.org/education/benchmarks/UrbanSemLab/Default.aspx
3.	Uncompress it, and place the folders "1_DSM/", "4_Ortho_RGBIR/" and "5_Labels_all/" 
	into the "Potsdam_data/" directory
4.	Run ```python make_tiles.py```. It will create input and target tiles in "Potsdam_data/tiles/"

## To train a model :
1.	Run ```python train.py``` with the desired parameters. 
	Refer to the arguments parser in the code to see the possible settings.
2.	Training can be monitored using the Weights and Biases tool (see https://docs.wandb.ai/quickstart).
	The URL to follow the training is provided in the console once validation occurs

## To test a model :
1.	Run ```python test.py``` with the desired parameters. 
2.	Metrics are printed in the console and the confusion matrix is saved as an image

## To predict an image :
1.	Run ```python predict.py``` by specifing the model and the image(s) to predict.
2.	If needed, try to expand the image(s) to create a batch of the size that has been used to train the model.
	This offers better results.
3.	If a ground truth is provided, "innacurate but certain" maps are produced
4.	Resulting predictions and maps are saved in the "predictions/" repository

**Update: 
A similar approach has been developed by Fisher et al [[3]](#3). This work was not inspired by it but further details can be found in this paper.

## References

<a id="1">[1]</a>  RONNEBERGER, Olaf, FISCHER, Philipp, et BROX, Thomas. U-net: Convolutional networks for biomedical image segmentation. In : International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015. p. 234-241. 

<a id="2">[2]</a> GAL, Yarin et GHAHRAMANI, Zoubin. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In : international conference on machine learning. PMLR, 2016. p. 1050-1059. [2]

<a id="3">[3]</a> Fisher, T.; Gibson, H.; Liu, Y.; Abdar, M.; Posa, M.; Salimi-Khorshidi, G.; Hassaine, A.; Cai, Y.; Rahimi, K.; Mamouei, M. Uncertainty-Aware Interpretable Deep Learning for Slum Mapping and Monitoring. Remote Sens. 2022, 14, 3072. https://doi.org/10.3390/rs14133072 
