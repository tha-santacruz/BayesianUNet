# Bayesian UNet model for uncertainty characterisation

This repository provides the well-known UNet model converted to a Bayesian UNet model [[1]](#1).
This model has been coded using Pytorch. This code was inspired by the implementation of the original UNet model in Pytorch (https://github.com/milesial/Pytorch-UNet).
The model has been implemented using the Monte Carlo Dropout method [[2]](#2).
It consists of adding a dropout layer (with a certain probability) at the end of each convolution layer during training and testing steps.
This Bayesian model provides different scores (entropy and mutual information) that could be used to characterize uncertainty in predictions. 

## To create tiles : 
1.	Download the Potsdam Dataset from this URL:
	https://www.isprs.org/education/benchmarks/UrbanSemLab/Default.aspx
2.	Uncompress it, and place the folders "1_DSM", "4_Ortho_RGBIR" and "5_Labels_all" 
	into the "Potsdam_data" directory
3.	Run the "make_tils.py" file that will create input and target tiles in "Potsdam_data/tiles"

## To train a model :
1.	Run train.py with the desired parameters. 
	Refer to the arguments parser in the code to see the possible settings
2.	Training can be monitored using the Weights and Biases tool.
	The URL to follow the training is provided in the console once validation occurs

## To test a model :
1.	Run test.py with the desired parameters. 
	Refer to the arguments parser in the code to see the possible settings
2.	Metrics are printed in the console and the confusion matrix is saved as an image

## To predict an image :
1.	Run predict.py by specifing the model and the image(s) to predict
2.	If needed, try to expand the image(s) to create a batch of the size that has been used to train the model.
	This offers better results.
3.	If a ground truth is provided, "innacurate but certain" maps are produced
4.	Resulting predictions and maps are saved in the "predictions" repository


## References

<a id="1">[1]</a>  RONNEBERGER, Olaf, FISCHER, Philipp, et BROX, Thomas. U-net: Convolutional networks for biomedical image segmentation. In : International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015. p. 234-241. 

<a id="2">[2]</a> GAL, Yarin et GHAHRAMANI, Zoubin. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In : international conference on machine learning. PMLR, 2016. p. 1050-1059. [2]
