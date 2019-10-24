import os
import glob
import torch
from collections import OrderedDict
from model.inceptionv4 import Inceptionv4
import pretrainedmodels.utils as utils


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

## data and labels
imagenet_classes_file = 'data/imagenet_classes.txt'
imagenet_2012_validation_synset_labels_file = 'data/imagenet_2012_validation_synset_labels.txt'
val_data_path = 'data/ILSVRC2012_img_val'
checkpoint = 'checkpoints/inceptionv4-8e4777a0.pth'
with open(imagenet_classes_file, 'r') as fr:
    imagenet_classes = fr.readlines()
imagenet_classes = [item.strip() for item in imagenet_classes]
with open(imagenet_2012_validation_synset_labels_file, 'r') as fr:
    validation_synset_labels = fr.readlines()
validation_synset_labels = [item.strip() for item in validation_synset_labels]


## model
model = Inceptionv4()
model.eval()
ordered_dict = parse_pth(checkpoint, model)
model.load_state_dict(ordered_dict)

def inference(model, jpeg_file):
    load_img = utils.LoadImage()
    # transformations depending on the model
    # rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
    tf_img = utils.TransformImage(model)
    input_img = load_img(jpeg_file)
    input_tensor = tf_img(input_img)  # 3x400x225 -> 3x299x299 size may differ
    input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
    input = torch.autograd.Variable(input_tensor, requires_grad=False)
    input = input
    output_logits = model(input).data
    id = torch.argmax(output_logits)
    label = validation_synset_labels[int(jpeg_file.split('/')[-1].split('.')[0].split('_')[-1]) - 1]
    return id, label

def inceptionv4_eval(model, val_data_path):
    jpeg_files = glob.glob(os.path.join(val_data_path, '*.JPEG'))
    tp = 0
    for i, jpeg_file in enumerate(jpeg_files):
        id, label = inference(model, jpeg_file)
        pred = imagenet_classes[id]
        if pred == label:
            tp += 1
        print(i, pred, label)
    return tp / len(jpeg_files)


if __name__ == '__main__':
    accuracy = inceptionv4_eval(model, val_data_path)
    print("The accuracy is %f" % accuracy)


