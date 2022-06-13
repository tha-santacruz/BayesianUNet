import torch
from torch import Tensor
import torchmetrics

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def accuracy_coeff(preds, target, num_classes):
    #assert input.size() == target.size()
    return torchmetrics.functional.accuracy(preds = preds,
                                            target = target,
                                            average='macro',
                                            mdmc_average='global',
                                            threshold=0.5,
                                            top_k=None,
                                            subset_accuracy=False,
                                            num_classes=num_classes,
                                            multiclass=None,
                                            ignore_index=None)


def multiclass_accuracy(preds, target, num_classes):

    return torchmetrics.functional.accuracy(preds = preds,
                                            target = target,
                                            average=None,
                                            mdmc_average='global',
                                            threshold=0.5,
                                            top_k=None,
                                            subset_accuracy=False,
                                            num_classes=num_classes,
                                            multiclass=None,
                                            ignore_index=None)

def F1_score(preds, target, num_classes):

    
    return torchmetrics.functional.f1_score(preds,
                        target,
                        num_classes=num_classes,
                        threshold=0.5,
                        average=None,
                        mdmc_average='global',
                        ignore_index=None,
                        top_k=None,
                        multiclass=None,
                        )



def IOU_score(preds, target, num_classes):

    return torchmetrics.functional.jaccard_index(preds,
                                                target,
                                                num_classes = num_classes,
                                                reduction='elementwise_mean')
def IOU_score_per_class(preds, target, num_classes):

    return torchmetrics.functional.jaccard_index(preds,
                                                target,
                                                num_classes = num_classes,
                                                reduction='none')