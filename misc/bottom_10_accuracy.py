# These top_1 and top_5 data are obtained by each_cls_topk_accuracy() in inceptionv4_eval.py
#top_1 = [782, 818, 282, 620, 836, 899, 885, 482, 240, 638]
#top_5 = [633, 818, 885, 762, 600, 623, 584, 464, 415, 556]

top_1  = [782, 638, 282, 482, 744, 836, 623, 620, 818, 167,] # for inception_resnet_v2
top_5 = [818, 885, 633, 623, 762, 600, 906, 493, 799, 584] # for inception_resnet_v2

with open('../data/imagenet_classes.txt', 'r') as fr:
    id2synsets = fr.readlines()
with open('../data/imagenet_synsets.txt', 'r') as fr:
    synset2classes = fr.readlines()

id2synsets = [item.strip() for item in id2synsets]
synset2classes = {item.strip().split()[0]: " ".join(item.strip().split()[1:]) for item in synset2classes}

print("="*10, "Top-1", "="*10)
for item in top_1:
    print(id2synsets[item], ":", synset2classes[id2synsets[item]])

print("="*10, "Top5", "="*10)
for item in top_5:
    print(id2synsets[item], ":", synset2classes[id2synsets[item]])