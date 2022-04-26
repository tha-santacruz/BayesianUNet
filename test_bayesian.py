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
        #print(batch_std.unique())

        #print(f"mean size {batch_mean.size()}")
        batch_pred_entropy = -torch.sum(batch_mean*batch_mean.log(),dim=1)
        #print(f"entropy size {batch_pred_entropy.size()}")
        batch_mutual_info = batch_pred_entropy+torch.mean(torch.sum(dropout_predictions*dropout_predictions.log(),dim=-3),dim=0)
        #print(f"mutual info size {batch_mutual_info.size()}")

        #print(batch_pred_entropy.mean())
        #print(batch_pred_entropy.mean(dim=(-2,-1)).size())

        #transform predictions to float labels for others metrics
        mask_pred = batch_mean
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


        ## produce PAVPU score
        # hyperparameteters
        w_size = 4 #patch size
        accuracy_tresh = 0.5
        
        #traverse the matrix with a patch 
        unfold = torch.nn.Unfold(kernel_size=(w_size, w_size),stride = 4) #to have 1 meter block

        #compute the accuracy each pach and check if it above the threshold
        masktrue_unfold = unfold(mask_true_labels.unsqueeze(dim=1).to(torch.float32))
        pred_unfold = unfold(mask_pred_labels.unsqueeze(dim=1).to(torch.float32))
        #print('pred_unfold size: {}'.format(pred_unfold.size()))
        #print('masktrue_unfold size: {}'.format(masktrue_unfold.size()))
        accuracy_matrix = torch.eq(pred_unfold, masktrue_unfold).to(torch.float32).mean(dim=1)
        #print('accuracy_matrix size :{}'.format(accuracy_matrix.size()))
        #print('average accuracy : {}'.format(accuracy_matrix.mean()))
        

        # compute the mean uncertainty and if it above the threshold
        uncertainty_matrix = unfold(batch_pred_entropy.unsqueeze(dim=1)).mean(dim=1)
        #print('uncertainty_matrix size :{}'.format(uncertainty_matrix.size()))

        bool_acc_matrix = torch.gt(accuracy_matrix, accuracy_tresh).to(torch.float32)
        print('bool_acc_matrix size: {}'.format(bool_acc_matrix.size()))
        
        #print('average uncertainty score: {}'.format(uncertainty_matrix.mean()))
        #print('std uncertainty score: {}'.format(uncertainty_matrix.std()))
        #print('range uncertainty score: {}'.format(uncertainty_matrix.max()-uncertainty_matrix.min()))
        
        t = 0.4
        uncertainty_tresh = uncertainty_matrix.min()+t*(uncertainty_matrix.max()-uncertainty_matrix.min())

        bool_uncert_matrix = torch.gt(uncertainty_matrix, uncertainty_tresh).to(torch.float32)
        #print('bool_acc_matrix size: {}'.format(bool_uncert_matrix.size()))

        #compute cf matrix for both accuracy and uncertainity
        # compute confidence matrix
        
        #compute the PAVU score
        nac = (bool_acc_matrix*(1-bool_uncert_matrix)).sum()
        nac_plus_nic = (1-bool_uncert_matrix).sum()
        niu = (bool_uncert_matrix*(1-bool_acc_matrix)).sum()
        nic_plus_niu = (1-bool_acc_matrix).sum()

        pa += nac/nac_plus_nic
        pu += niu/nic_plus_niu
        pavpu += (nac+niu) / torch.ones_like(bool_acc_matrix).sum()
        
        '''
        print('pa = {}'.format(pa))
        print('pu = {}'.format(pu))
        print('pavpu = {}'.format(pavpu))

        print(torch.mean(bool_acc_matrix))
        print(torch.mean(bool_uncert_matrix))
        '''
    cf_matrix = cf_matrix/cf_matrix.sum(axis=1,keepdims=True)

    net.train()

    print('pa = {}'.format(pa/num_val_batches))
    print('pu = {}'.format(pu/num_val_batches))
    print('pavpu = {}'.format(pavpu/num_val_batches))


    return (dice_score / num_val_batches,
            accuracy_score/num_val_batches,
            accuracy_per_class/num_val_batches,
            F1_coeff_per_class/num_val_batches,
            IOU_coeff/num_val_batches,
            IOU_coeff_per_class/num_val_batches,
            cf_matrix,
            pa/num_val_batches,
            pu/num_val_batches,
            pavpu/num_val_batches)


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            print("activated dropout")


if __name__ == '__main__':

    #wandb initilization
    experiment = wandb.init(project="Model Testing", entity="bbk_2022", resume='allow', name = datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    """ Evaluate samples of the test set with uncertainity  values"""
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # define test set
    test_set = BBKDataset(zone = ("genf",), split = "test", buildings = True, vegetation = True, random_seed = 1)
    test_dl = DataLoader(test_set, batch_size=32, shuffle=True)

    # declare model
    net = BayesianUNet(n_channels=7, n_classes=9, bilinear=False).to(device=device)
    checkpoint_path = 'checkpoints_bayesian/checkpoint_epoch60.pth'
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()

    # evaluate test set using pretrained model
    (val_score,
    accuracy_score,
    accuracy_per_class,
    F1_score,
    IOU_score,
    IOU_score_per_class,
    cf_matrix,
    pa,
    pu,
    pavpu) = evaluate_uncertainty(net,test_dl,device, nb_forward=10)
    
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
                'PAvPU score':pavpu,
                'Metric per class':score_table, 
                'conf_mat' : wandb.Image(plt),
                #**histograms
            })
    plt.close()