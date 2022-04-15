import logging
import torch
from evaluate import evaluate
from utils.dataset import BBKDataset
from torch.utils.data import DataLoader
from unet import UNet
# from bayesian_unet import BayesianUNet
from evaluate import evaluate
import seaborn as sns
import wandb
from datetime import datetime
import matplotlib.pyplot as plt


if __name__ == "__main__":

    #wandb initilization
    experiment = wandb.init(project="Model Testing", entity="bbk_2022", resume='allow', name = datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    '''
    experiment.config.update(dict(
                                epochs=epochs,
                                optim_class=optim_class,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                patience = patience,
                                weight_decay=weight_decay,
                                momentum=momentum,
                                save_checkpoint=save_checkpoint,
                                amp=amp,
                                allow_val_change=True)
                                )
    '''

    # initialize logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # define test set
    test_set = BBKDataset(zone = ("alles",), split = "test", buildings = True, vegetation = True, random_seed = 1)
    test_dl = DataLoader(test_set, batch_size=16, shuffle=True)

    # define model
    net = UNet(n_channels=7, n_classes=9, bilinear=False).to(device=device)
    # net = BayesianUNet(n_channels=7, n_classes=9, bilinear=args.bilinear)

    # load pretrained model parameters
    checkpoint_path = 'checkpoints_final_model/checkpoint_epoch25.pth'
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logging.info(f'Model loaded from {checkpoint_path}')

    # evaluate test set using pretrained model
    val_score, accuracy_score, accuracy_per_class, F1_score, IOU_score, IOU_score_per_class, cf_matrix = evaluate(net,test_dl,device)
    
    #logging.info('Accuracy score per classe : {}'.format(accuracy_score))
    logging.info('Global accuracy score per class : {}'.format(accuracy_per_class))
    #logging.info('F1 score : {}'.format(F1_score))
    logging.info('Validation Dice score: {}'.format(val_score))
    logging.info('IOU score: {}'.format(IOU_score))

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