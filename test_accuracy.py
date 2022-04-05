from utils.accuracy import multiclass_accuracy
import torch 
target = torch.tensor([0, 1, 2,2,3,4,2])

preds = torch.tensor([0, 4, 2,2,3,4,2])

test = multiclass_accuracy(target, preds, 5)
print(test[0])
#print('test {}'.format((test/5).to(float)))
#return a tensor 