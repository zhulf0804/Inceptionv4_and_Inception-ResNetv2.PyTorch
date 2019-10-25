import torch
import numpy as np
import datetime
from collections import OrderedDict
from model.inceptionv4 import Inceptionv4
from torch.utils.data import DataLoader
from datasets import ImageNet


def parse_pth(checkpoint, model):
    model_dict = model.state_dict()
    state_dict = torch.load(checkpoint)
    last_linear_weight = state_dict.pop('last_linear.weight')
    last_linear_bias = state_dict.pop('last_linear.bias')

    def get_key(item):
        lists = ['conv.weight', 'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var']
        d = {item: v for v, item in enumerate(lists)}
        d2 = {'conv': 1, 'bn': 1, 'bias': 1, 'running_mean': 1, 'weight': 1, 'running_var': 1}
        if len(item.split('.')) == 6:
            return (int(item.split('.')[1]), item.split('.')[2], item.split('.')[3], d['.'.join(item.split('.')[-2:])])
        elif len(item.split('.')) == 5:
            return (
                int(item.split('.')[1]), item.split('.')[2], 1, d['.'.join(item.split('.')[-2:])])
        else:
            return (
            int(item.split('.')[1]), d2[item.split('.')[2]], d2[item.split('.')[3]], d['.'.join(item.split('.')[-2:])])

    sorted_keys = sorted(state_dict.keys(), key=lambda item: get_key(item))
    ordered_dict = OrderedDict()

    i = 0
    for k in model_dict.keys():
        if 'num_batches_tracked' not in k:
            if i == len(sorted_keys):
                ordered_dict[k] = last_linear_weight[1:, :]
            elif i == len(sorted_keys) + 1:
                ordered_dict[k] = last_linear_bias[1:]
            else:
                ordered_dict[k] = state_dict[sorted_keys[i]]
            i += 1
    return ordered_dict


dataset = ImageNet('data/ILSVRC2012_img_val', 'data/imagenet_classes.txt', 'data/imagenet_2012_validation_synset_labels.txt')
checkpoint = 'checkpoints/inceptionv4-8e4777a0.pth'

## model
model = Inceptionv4()
model = model.cuda()
model.eval()
ordered_dict = parse_pth(checkpoint, model)
model.load_state_dict(ordered_dict)

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
    inference_time()
