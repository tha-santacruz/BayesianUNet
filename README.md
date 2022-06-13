# baseline_UNet
Baseline UNet model for bbk classification

to create tiles : 
1	download the Potsdam Dataset from this URL:
	https://www.isprs.org/education/benchmarks/UrbanSemLab/Default.aspx
2	Uncompress it, and place the folders "1_DSM", "4_Ortho_RGBIR" and "5_Labels_all" 
	into the "Potsdam_data" directory
3	Run the "make_tils.py" file that will create input and target tiles in "Potsdam_data/tiles"

to train a model :
1	Run train.py with the desired parameters. 
	Refer to the arguments parser in the code to see the possible settings
2	Training can be monitored using the Weights and Biases tool.
	Le URL to follow the training is provided in the console once validation occurs

to test a model :
1	Run test.py with the desired parameters. 
	Refer to the arguments parser in the code to see the possible settings
2	Metrics are printed in the console and the confusion matrix is saved as an image

to predict an image :
1	Run predict.py by specifing the model and the image(s) to predict
2	If needed, try to expand the image(s) to create a batch of the size that has been used to train the model.
	This offers better results
3	If a ground truth is provided, "innacurate but certain" maps are produced
4	Resulting predictions and maps are saved in the "predictions" repository
