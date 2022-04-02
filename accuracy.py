import torchmetrics

torchmetrics.functional.accuracy(preds,
                                target,
                                average='micro',
                                mdmc_average='global',
                                threshold=0.5,
                                top_k=None,
                                subset_accuracy=False,
                                num_classes=None,
                                multiclass=None,
                                ignore_index=None)
