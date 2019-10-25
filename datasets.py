import glob
import math
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

class ImageNet(Dataset):
    def __init__(self, data_dir, imagenet_classes_file, imagenet_2012_validation_synset_labels_file):
        self.data_dir = data_dir
        self.imagenet_classes_file = imagenet_classes_file
        self.imagenet_2012_validation_synset_labels_file = imagenet_2012_validation_synset_labels_file
        self.input_size = (299, 299, 3)
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.input_range = [0, 1]
        self.scale = 0.875
        self.space = 'RGB'
        self.filenames = glob.glob(os.path.join(self.data_dir, "*.JPEG"))
        self.validation_synset_labels = self.get_label()
        self.synset2label = self.synset2label()
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img = Image.open(filename).convert(self.space)
        transforms = self.transform()
        tensor = transforms(img)
        basename = os.path.basename(filename).split('.')[0].split('_')[-1]
        label = self.synset2label[self.validation_synset_labels[int(basename) - 1]]
        return tensor, label

    def get_label(self):
        with open(self.imagenet_2012_validation_synset_labels_file, 'r') as fr:
            validation_synset_labels = fr.readlines()
        validation_synset_labels = [item.strip() for item in validation_synset_labels]
        return validation_synset_labels

    def synset2label(self):
        with open(self.imagenet_classes_file, 'r') as fr:
            imagenet_classes = fr.readlines()
        synset2label = {item.strip(): ind for ind, item in enumerate(imagenet_classes)}
        return synset2label

    def transform(self):
        tfs = []
        tfs.append(transforms.Resize(int(math.floor(max(self.input_size) / self.scale))))
        tfs.append(transforms.CenterCrop(max(self.input_size)))
        tfs.append(transforms.ToTensor())
        tfs.append(transforms.Normalize(self.mean, self.std))
        tf = transforms.Compose(tfs)
        return tf