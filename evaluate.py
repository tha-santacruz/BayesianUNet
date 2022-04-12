import logging 

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import utils.metrics as metrics

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)

    #Initialization
    dice_score = 0

    accuracy_score = 0
    accuracy_per_class = 0

    F1_coeff_per_class  = 0

    IOU_coeff = 0 
    IOU_coeff_per_class = 0
    
    cf_matrix = np.zeros(shape = (net.n_classes,net.n_classes))

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch[0], batch[1]
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask (pytorch tensor have the following structure : [batch_no, class, pixel_x, pixel_y])
            mask_pred = net(image)

            # compute confidence matrix
            cf_matrix = cf_matrix + confusion_matrix(mask_true.argmax(dim=1).view(-1).cpu(),mask_pred.argmax(dim=1).view(-1).cpu(), labels = np.arange(0,9))

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += metrics.dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                
                #transform predictions to float labels for others metrics
                mask_pred_labels = mask_pred.argmax(dim=1) 
                mask_true_labels = mask_true.argmax(dim=1)

                #compute the accuracy
                accuracy_score += metrics.accuracy_coeff(mask_pred_labels, mask_true_labels, num_classes = net.n_classes)
                accuracy_per_class += metrics.multiclass_accuracy(mask_pred_labels, mask_true_labels, num_classes = net.n_classes)
                #compute F1 score
                F1_coeff_per_class += metrics.F1_score(mask_pred_labels, mask_true_labels, num_classes= net.n_classes)
                #compute IOU score 
                IOU_coeff += metrics.IOU_score(mask_pred_labels, mask_true_labels, num_classes= net.n_classes)
                IOU_coeff_per_class += metrics.IOU_score_per_class(mask_pred_labels, mask_true_labels, num_classes= net.n_classes)
               
                #transform prediction in one-hot to compute dice score (ignoring background for dice score)
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0,3,1,2).float()
                # compute the Dice score per class 
                dice_score += metrics.multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

    cf_matrix = cf_matrix/cf_matrix.sum(axis=1,keepdims=True)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score, accuracy_score, accuracy_per_class

    return dice_score / num_val_batches, accuracy_score/num_val_batches, accuracy_per_class/num_val_batches, F1_coeff_per_class/ num_val_batches, IOU_coeff/num_val_batches, IOU_coeff_per_class/num_val_batches,  cf_matrix
