import torch
import numpy as np
import datetime
from model.inceptionv4 import Inceptionv4
from torch.utils.data import DataLoader
from datasets import ImageNet


dataset = ImageNet('data/ILSVRC2012_img_val', 'data/imagenet_classes.txt', 'data/imagenet_2012_validation_synset_labels.txt')

## model
model = Inceptionv4()
model = model.cuda()
model.eval()
model.load_state_dict(torch.load('checkpoints/inceptionv4.pth'))

def topk_accuracy():
    val_loader = DataLoader(dataset=dataset,
                            batch_size=25,
                            shuffle=False,
                            num_workers=4)
    tp_1, tp_5 = 0, 0
    for i, data in enumerate(val_loader):
        input, label = data
        input, label = input.cuda(), label.cuda()
        pred = model(input)
        _, pred = torch.topk(pred, 5, dim=1)
        correct = pred.eq(label.view(-1, 1).expand_as(pred)).cpu().numpy()
        tp_1 += correct[:, 0].sum()
        tp_5 += correct.sum()
        print(i, "top1: ", tp_1, "top5:", tp_5)
    print("Top1 accuracy: ", tp_1 / 50000)
    print("Top5 accuracy: ", tp_5 / 50000)


def each_cls_topk_accuracy():
    batch_size = 25
    val_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)
    count = [0] * 1000
    tp_1 = [0] * 1000
    tp_5 =[0] * 1000
    for i, data in enumerate(val_loader):
        input, label = data
        input, label = input.cuda(), label.cuda()
        pred = model(input)
        _, pred = torch.topk(pred, 5, dim=1)
        correct = pred.eq(label.view(-1, 1).expand_as(pred)).cpu().numpy()
        for j in range(batch_size):
            tp_1[label.cpu()[j]] += correct[j, 0]
            tp_5[label.cpu()[j]] += correct[j].sum()
            count[label.cpu()[j]] += 1
        print("batch %d"%i)
    print(count)
    print(tp_1)
    print(tp_5)

    accuracy_1 = np.array(tp_1) / np.array(count)
    accuracy_5 = np.array(tp_5) / np.array(count)
    print(sorted(accuracy_1))
    print(sorted(accuracy_5))
    print(accuracy_1.argsort())
    print(accuracy_5.argsort())


def inference_time():
    val_loader = DataLoader(dataset=dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1)
    starttime = datetime.datetime.now()
    for i, data in enumerate(val_loader):
        input, label = data
        input, label = input.cuda(), label.cuda()
        pred = model(input)
        _, pred = torch.topk(pred, 1, dim=1)
        print(i, label, pred)
    endtime = datetime.datetime.now()
    total_time = (endtime - starttime).seconds
    single_time = total_time / 50000
    print("Total time is %f s"%total_time)
    print("Single average time is %f s" %single_time)



if __name__ == '__main__':
    # each_cls_topk_accuracy()
    # inference_time()
    topk_accuracy()