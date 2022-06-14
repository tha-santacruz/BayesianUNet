## Importing packages
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.metrics as metrics
from bayesian_unet import BayesianUNet
from utils.potsdam_dataset import PotsdamDataset


# Source to compute entropy and mutual information : 
# https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch
def evaluate_uncertainty(net,
                        dataloader,
                        device,
                        nb_forward):
    net.eval()
    #Activate the dropout layers
    enable_dropout(net)

    #Define the bash number
    num_val_batches = len(dataloader)

    #Initialization of scores
    dice_score = 0
    accuracy_score = 0
    accuracy_per_class = 0
    F1_coeff_per_class  = 0
    IOU_coeff = 0 
    IOU_coeff_per_class = 0
    cf_matrix = np.zeros(shape = (net.n_classes,net.n_classes))
    pa = 0
    pu = 0
    pavpu = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch[0], batch[1]

        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(dtype=torch.float32)

        #to store n_forward predictions on the same batch
        dropout_predictions = torch.empty((0,mask_true.size(0),mask_true.size(1),mask_true.size(2),mask_true.size(3)))
        
        # iterate over the forward passes
        for f_pass in range(nb_forward):
            with torch.no_grad():
                # predict the mask (pytorch tensor have the following structure : [batch_no, class, pixel_x, pixel_y])
                mask_pred = net(image)

                # concatenate prediction to the other made on the same batch
                dropout_predictions = torch.cat((dropout_predictions,mask_pred.cpu().softmax(dim=1).unsqueeze(dim=0)),dim=0)

                # compute confidence matrix
                cf_matrix = cf_matrix + confusion_matrix(mask_true.argmax(dim=1).view(-1).cpu(),mask_pred.argmax(dim=1).view(-1).cpu(), labels = np.arange(0,net.n_classes))


        #Compute uncertainty indicateurs
        batch_mean = dropout_predictions.mean(dim=0)
        batch_std = dropout_predictions.std(dim=0)
        batch_pred_entropy = -torch.sum(batch_mean*batch_mean.log(),dim=1)
        batch_mutual_info = batch_pred_entropy+torch.mean(torch.sum(dropout_predictions*dropout_predictions.log(),dim=-3),dim=0)


        #Transform predictions to float labels for others metrics
        mask_pred_labels = batch_mean.argmax(dim=1) 
        mask_true_labels = mask_true.argmax(dim=1)


        #compute the accuracy
        accuracy_score += metrics.accuracy_coeff(mask_pred_labels, mask_true_labels, num_classes = net.n_classes)
        accuracy_per_class += metrics.multiclass_accuracy(mask_pred_labels, mask_true_labels, num_classes = net.n_classes)
        #compute F1 score
        F1_coeff_per_class += metrics.F1_score(mask_pred_labels, mask_true_labels, num_classes= net.n_classes)
        
        #transform prediction in one-hot to compute dice score (ignoring background for dice score)
        mask_pred_onehot = F.one_hot(mask_pred_labels, net.n_classes).permute(0,3,1,2).float()
        # compute the Dice score per class 
        dice_score += metrics.multiclass_dice_coeff(mask_pred_onehot[:, 1:, ...],
                                                    mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)


        #Produce PAVPU score
        w_size = 4 #patch size
        accuracy_tresh = 0.5
        
        #Define the patch size
        unfold = torch.nn.Unfold(kernel_size=(w_size, w_size),stride = w_size)

        #compute the accuracy for each patch and check if it above the threshold
        masktrue_unfold = unfold(mask_true_labels.unsqueeze(dim=1).to(torch.float32))
        pred_unfold = unfold(mask_pred_labels.unsqueeze(dim=1).to(torch.float32))
        accuracy_matrix = torch.eq(pred_unfold, masktrue_unfold).to(torch.float32).mean(dim=1)
        bool_acc_matrix = torch.gt(accuracy_matrix, accuracy_tresh).to(torch.float32)
        # compute the mean uncertainty and if it is above the threshold
        uncertainty_matrix = unfold(batch_pred_entropy.unsqueeze(dim=1)).mean(dim=1)
        #Define the uncertainty treshold as in the litterature
        t = 0.4
        uncertainty_tresh = uncertainty_matrix.min()+t*(uncertainty_matrix.max()-uncertainty_matrix.min())
        bool_uncert_matrix = torch.gt(uncertainty_matrix, uncertainty_tresh).to(torch.float32)
        
        #compute the performances metrics 
        nac = (bool_acc_matrix*(1-bool_uncert_matrix)).sum()
        nac_plus_nic = (1-bool_uncert_matrix).sum()
        niu = (bool_uncert_matrix*(1-bool_acc_matrix)).sum()
        nic_plus_niu = (1-bool_acc_matrix).sum()
        pa += nac/nac_plus_nic
        pu += niu/nic_plus_niu
        pavpu += (nac+niu) / torch.ones_like(bool_acc_matrix).sum()

    #Compute the global confusion matrix    
    cf_matrix = cf_matrix/cf_matrix.sum(axis=1,keepdims=True)

    #Re-turn the model in train mode
    net.train()
    return (dice_score/num_val_batches,
            accuracy_score/num_val_batches,
            accuracy_per_class/num_val_batches,
            F1_coeff_per_class/num_val_batches,
            cf_matrix,
            pa/num_val_batches,
            pu/num_val_batches,
            pavpu/num_val_batches)


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            #logging.info("activated dropout")


if __name__ == '__main__':
    """ Evaluate samples of the test set to get performance metrics"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Create datasets
    test_set = PotsdamDataset(split = "test", random_seed = 1, augment=False)
    test_dl = DataLoader(test_set, batch_size=16, shuffle=False)

    # Declare model
    net = BayesianUNet(n_channels=test_set.N_CHANNELS, n_classes=test_set.N_CLASSES, bilinear=False).to(device=device)
    # Choose the trained parameters to load in the model
    checkpoint_path = 'checkpoints/checkpoint_epoch5.pth'
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()

    # evaluate test set using pretrained model
    (val_score,
    accuracy_score,
    accuracy_per_class,
    F1_score,
    cf_matrix,
    pa,
    pu,
    pavpu) = evaluate_uncertainty(net,test_dl,device, nb_forward=10)
    
    ##Confusion matrix 
    plt.figure()
    sns.heatmap(cf_matrix, annot=True, annot_kws={"size":8}, fmt='.1%', cmap='Blues', cbar=True, xticklabels=test_set.CLASSES_list,yticklabels=test_set.CLASSES_list)
    plt.tight_layout()
    plt.savefig("cf_matrix.png")
    plt.close()

    logging.info(f'Validation Dice score : {val_score}')
    logging.info(f'Global accuracy score : {accuracy_score}')
    logging.info(f'F1 score : {F1_score}')
    logging.info(f'Classwise accuracy score : {accuracy_per_class}')
    logging.info(f'Pa score : {pa}')
    logging.info(f'Pu score : {pu}')
    logging.info(f'PavPu score : {pavpu}')
