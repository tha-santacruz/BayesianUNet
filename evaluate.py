import logging 

import torch
import torch.nn.functional as F
from tqdm import tqdm

import utils.metrics as metrics

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    accuracy_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch[0], batch[1]
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask (pytorch tensor have the following structure : [batch_no, class, pixel_x, pixel_y])
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += metrics.dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:

                #transform preidiction in one-hot to compute dice score (ignoring background for dice score)
                mask_pred_dice = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, 
                dice_score += metrics.multiclass_dice_coeff(mask_pred_dice[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

                #transform predictions to float labels for others metrics
                mask_preds = mask_pred.argmax(dim=1).to(torch.int64)[:, 1:, ...]
                mask_true = torch.softmax(mask_true, dim=1).argmax(dim=1).to(torch.int64)[:, 1:, ...]
                 #compute the accuracy
                accuracy_score += metrics.accuracy_coeff(mask_pred, mask_true)
                #compute accuracy per class
                accuracy_per_class = metrics.multiclass_accuracy(mask_preds, mask_true, num_classes = net.n_classes)

               #TODO: take the accuracy, dice score,  per classe and take it out the loop to compute them globally 
           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score, accuracy_score, accuracy_per_class

    return dice_score / num_val_batches, accuracy_score/num_val_batches, accuracy_per_class #/num_val_batches
