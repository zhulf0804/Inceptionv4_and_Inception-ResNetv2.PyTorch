import torch
import glob
import os
from model.inceptionv4 import Inceptionv4
from collections import OrderedDict
import pretrainedmodels.utils as utils


model = Inceptionv4()
model.eval()
model_dict = model.state_dict()

f = torch.load('/Users/zhulf/Downloads/inceptionv4-8e4777a0.pth')
last_linear_weight = f.pop('last_linear.weight')
last_linear_bias = f.pop('last_linear.bias')

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
        return (int(item.split('.')[1]), d2[item.split('.')[2]], d2[item.split('.')[3]], d['.'.join(item.split('.')[-2:])])
sorted_keys = sorted(f.keys(), key=lambda item: get_key(item))
ordered_dict = OrderedDict()

i = 0
for k in model_dict.keys():
    if 'num_batches_tracked' not in k:
        #print(k, "---", sorted_keys[i])
        if i == len(sorted_keys):
            ordered_dict[k] = last_linear_weight[1:, :]
        elif i == len(sorted_keys) + 1:
            ordered_dict[k] = last_linear_bias[1:]
        else:
            ordered_dict[k] = f[sorted_keys[i]]
        i += 1

#tmp = torch.load('/Users/zhulf/Downloads/inceptionv4-8e4777a0.pth')
model.load_state_dict(ordered_dict)
model = model.cuda()

load_img = utils.LoadImage()
# transformations depending on the model
# rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
tf_img = utils.TransformImage(model)
data_dir = '/root/data/zhulf/imagenet/raw-data/ILSVRC2012_img_val'
files = glob.glob(os.path.join(data_dir, '*', '*.JPEG'))
#files = ['/Users/zhulf/Downloads/ILSVRC2012_img_val/ILSVRC2012_val_00049857.JPEG']
tf_img = utils.TransformImage(model)
image_classes = '/root/data/zhulf/imagenet_classes.txt'
#image_classes = '../data/imagenet_classes.txt'
f = open(image_classes, 'r')
lines = f.readlines()
mmap = [line.strip() for line in lines]


count = 0
for i, file in enumerate(files):
    label = file.split('/')[-2]
    path_img = file
    input_img = load_img(path_img)
    input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
    input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
    input = torch.autograd.Variable(input_tensor,
        requires_grad=False)
    input = input.cuda()
    output_logits = model(input).data
    id = torch.argmax(output_logits)
    print(i, id, mmap[id])
    if label == mmap[id]:
        count += 1
print(count)

'''
data_dir = '/root/data/zhulf/imagenet/raw-data/ILSVRC2012_img_val'
files = glob.glob(os.path.join(data_dir, '*', '*.JPEG'))

path_img = '/Users/zhulf/Downloads/ILSVRC2012_img_val/ILSVRC2012_val_00049857.JPEG'

input_img = load_img(path_img)
input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_tensor, requires_grad=False)

output_logits = model(input)
print(torch.argmax(output_logits))
'''