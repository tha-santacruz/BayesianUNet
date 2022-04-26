import logging
import torch
import torch.nn.functional as F
from utils.dataset import BBKDataset
from torch.utils.data import DataLoader
import utils.metrics as metrics
from unet import UNet
from bayesian_unet import BayesianUNet
from evaluate import evaluate
import seaborn as sns
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch
def evaluate_uncertainty(net, dataloader, device, nb_forward):
    net.eval()
    enable_dropout(net)
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
        mask_true = mask_true.to(#device=device,
                                 dtype=torch.float32)

        # to store n_forward predictions on the same batch
        dropout_predictions = torch.empty((0,mask_true.size(0),mask_true.size(1),mask_true.size(2),mask_true.size(3)))

        for f_pass in range(nb_forward):
            with torch.no_grad():
                # predict the mask (pytorch tensor have the following structure : [batch_no, class, pixel_x, pixel_y])
                mask_pred = net(image)

                # concatenate prediction to the other made on the same batch
                dropout_predictions = torch.cat((dropout_predictions,mask_pred.cpu().softmax(dim=1).unsqueeze(dim=0)),dim=0)

                # compute confidence matrix
                cf_matrix = cf_matrix + confusion_matrix(mask_true.argmax(dim=1).view(-1).cpu(),mask_pred.argmax(dim=1).view(-1).cpu(), labels = np.arange(0,9))

        
        
        #print(dropout_predictions.unique())
        # if torch.eq(dropout_predictions[-1].view(-1),dropout_predictions[-2].view(-1)).all():
        #     print("same result")
        batch_mean = dropout_predictions.mean(dim=0)
        batch_std = dropout_predictions.std(dim=0)
        print(batch_std.unique())
        #batch_mean = dropout_predictions.mean(dim=(0,-2,-1))
        #batch_std = dropout_predictions.std(dim=(0,-2,-1))
        print(f"mean size {batch_mean.size()}")
        batch_pred_entropy = -torch.sum(batch_mean*batch_mean.log(),dim=1)
        print(f"entropy size {batch_pred_entropy.size()}")
        batch_mutual_info = batch_pred_entropy+torch.mean(torch.sum(dropout_predictions*dropout_predictions.log(),dim=-3),dim=0)
        print(f"mutual info size {batch_mutual_info.size()}")

        #print(batch_pred_entropy.mean())
        #print(batch_pred_entropy.mean(dim=(-2,-1)).size())
        #batch_mutual_info = 

        #transform predictions to float labels for others metrics
        mask_pred = dropout_predictions[-1]
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

    score_div = num_val_batches*nb_forward
    return dice_score / score_div, accuracy_score/score_div, accuracy_per_class/score_div, F1_coeff_per_class/score_div, IOU_coeff/score_div, IOU_coeff_per_class/score_div,  cf_matrix


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            print("activated dropout")

def compute_scores(mask_pred, mask_true):

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

    
    return []

def compute_uncertainty():
    return []

if __name__ == '__main__':

    #wandb initilization
    experiment = wandb.init(project="Model Testing", entity="bbk_2022", resume='allow', name = datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    """ Evaluate samples of the test set with uncertainity  values"""
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # define test set
    test_set = BBKDataset(zone = ("alles",), split = "test", buildings = True, vegetation = True, random_seed = 1)
    test_dl = DataLoader(test_set, batch_size=32, shuffle=True)

    # declare model
    net = BayesianUNet(n_channels=7, n_classes=9, bilinear=False).to(device=device)
    checkpoint_path = 'checkpoints/checkpoint_epoch80.pth'
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()

    # evaluate test set using pretrained model
    val_score, accuracy_score, accuracy_per_class, F1_score, IOU_score, IOU_score_per_class, cf_matrix = evaluate_uncertainty(net,test_dl,device, nb_forward=10)
    
    # create wandb objects for visualisation
    plt.figure()
    sns.heatmap(cf_matrix, annot=True, annot_kws={"size":8}, fmt='.2%', cmap='Blues', cbar=True, xticklabels=test_set.BBK_CLASSES_list,yticklabels=test_set.BBK_CLASSES_list)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()

    class_labels = {0 : "null",
                    1 : "wooded_area",
                    2 : "water",
                    3 : "bushes",
                    4 : "individual_tree",
                    5 : "no_woodland",
                    6 : "ruderal_area",
                    7 : "without_vegetation", 
                    8 : "buildings"}
    scores = {
            'Accuracy': accuracy_per_class,
            'F1 score' : F1_score,
            'IOU': IOU_score_per_class
            }
    columns_table= list(class_labels.values())
    data_table = [accuracy_per_class, F1_score, IOU_score_per_class]
    score_table = wandb.Table(data = data_table, columns=columns_table)
    #Add a column for scores names
    score_table.add_column(name='score',data=list(scores.keys()))


    experiment.log({
                'Validation Dice score': val_score,
                'Global accuracy score': accuracy_score,
                'IOU score': IOU_score,
                'Metric per class':score_table, 
                'conf_mat' : wandb.Image(plt),
                #**histograms
            })
    plt.close()