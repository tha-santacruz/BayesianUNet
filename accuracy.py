import torchmetrics
#test branch
def compute_accuracy(preds, target ):

    return torchmetrics.functional.accuracy(preds = preds,
                                            target = preds,
                                            average='micro',
                                            mdmc_average='global',
                                            threshold=0.5,
                                            top_k=None,
                                            subset_accuracy=False,
                                            num_classes=None,
                                            multiclass=None,
                                            ignore_index=None)


def multiclass_accuracy(preds, target, num_classes):

    return torchmetrics.functional.accuracy(preds = preds,
                                            target = preds,
                                            average=None,
                                            mdmc_average='global',
                                            threshold=0.5,
                                            top_k=None,
                                            subset_accuracy=False,
                                            num_classes=num_classes,
                                            multiclass=None,
                                            ignore_index=None)


