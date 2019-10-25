import torch
from thop import profile
from model.inceptionv4 import Inceptionv4

def get_inceptionv4_flops_params():
    model = Inceptionv4()
    input = torch.randn(1, 3, 229, 229)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)

get_inceptionv4_flops_params()