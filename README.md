## Introduction
An inofficial PyTorch implementation of [Inception-v4, Inception-ResNet and
the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

## Models
+ Inception-v4
+ Inception-ResNet-v2(TBD)

## Analysis
All the results reported here are based on **this repo**, and 50000 ImageNet **validation** sets。

- [X] top-1 accuracy
- [ ] top-5 accuracy
- [ ] \# model parameters
- [ ] inference time (average)
- [ ] blacklists
- [ ] top10 accuracy
- [ ] bottom10 accuracy

+ top-1 and top-5 

|          | #top-1 | top-1 | #top-5 | top-5 |
| :------: | :------: | :------: | :------: | :------: |
| eps=0.001, count_include_pad=False | 40041 | 0.801 | 47445 | 0.949 |
| eps=0.001, count_include_pad=True | 39970 | 0.799 | 47395 | 0.948 |
| eps=1e-5, count_include_pad=False | 40036 | 0.801 | 47438 | 0.949 |


## Reference
+ [https://github.com/tensorflow/models/tree/master/research/slim/nets](https://github.com/tensorflow/models/tree/master/research/slim/nets)
+ [https://github.com/tensorflow/models/tree/master/research/inception/inception/data](https://github.com/tensorflow/models/tree/master/research/inception/inception/data)
+ [https://github.com/Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)
+ [https://github.com/kentsommer/keras-inceptionV4](https://github.com/kentsommer/keras-inceptionV4)