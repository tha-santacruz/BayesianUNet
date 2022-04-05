import torchmetrics
#test branch
def accuracy_coeff(preds, target ):
    #assert input.size() == target.size()
    return torchmetrics.functional.accuracy(preds = preds,
                                            target = target,
                                            average='macro',
                                            mdmc_average='global',
                                            threshold=0.5,
                                            top_k=None,
                                            subset_accuracy=False,
                                            num_classes=7,
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


