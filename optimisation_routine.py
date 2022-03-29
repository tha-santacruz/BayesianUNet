from train import train_net


#TODO : pass optimizer as parameter in train.py
#Hyperparameters

epochs= 5,
batch_size= 32,
#batch_size = [16, 32, 64, 128]

learning_rate= 1e-5,
#learning_rate = [11e-4, 1e-5, 1e-6, 1e-7]
val_percent= 0.1,
save_checkpoint = True,
img_scale= 0.5,
amp = False

#TODO : add optmizer in the wandb.config
optimizer = [Adam(weight_decay), SDGD(momentum), RMSprop]

#TODO : to iterate to 
momentum = 