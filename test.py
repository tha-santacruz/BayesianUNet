import logging
import torch
from evaluate import evaluate
from utils.dataset import BBKDataset
from torch.utils.data import DataLoader
from unet import UNet
# from bayesian_unet import BayesianUNet
from evaluate import evaluate
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # initialize logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # define test set
    test_set = BBKDataset(zone = ("genf",), split = "test", buildings = True, vegetation = True, random_seed = 1)
    test_dl = DataLoader(test_set, batch_size=32, shuffle=True)

    # define model
    net = UNet(n_channels=7, n_classes=9, bilinear=False)
    # net = BayesianUNet(n_channels=7, n_classes=9, bilinear=args.bilinear)

    # load pretrained model parameters
    net.load_state_dict(torch.load('checkpoints/checkpoint_epoch50.pth', map_location=device))
    logging.info(f'Model loaded from checkpoints/checkpoint_epoch50.pth')

    # evaluate test set using pretrained model
    dice_score, accuracy, accuracy_class, F1_class, iou, iou_class, cf_matrix = evaluate(net,test_dl,device)
    print('overall dice score : {}'.format(dice_score))
    print('overall accuracy : {}'.format(accuracy))
    print('overall IOU : {}'.format(iou))
    print('per class F1 score : {}'.format(F1_class))
    print('per class accuracy : {}'.format(accuracy_class))
    print('per class IOU : {}'.format(iou_class))

    plt.figure()
    sns.heatmap(cf_matrix, annot=True, annot_kws={"size":8}, fmt='.2%', cmap='Blues', cbar=True, xticklabels=test_set.BBK_CLASSES_list,yticklabels=test_set.BBK_CLASSES_list)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()