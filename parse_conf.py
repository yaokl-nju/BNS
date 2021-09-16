import hydra
from omegaconf.dictconfig import DictConfig
import argparse
import numpy as np
import torch
import sys

parser = argparse.ArgumentParser()
args = parser.parse_args()

@hydra.main(config_path='conf', config_name='config')
def get_args_global(conf):
    for key in conf.keys():
        args.__setattr__(key, conf[key])
@hydra.main(config_path='conf', config_name='config-dataset')
def get_args_dataset(conf):
    params = conf[args.dataset]
    for key in params.keys():
        args.__setattr__(key, params[key])
@hydra.main(config_path='conf', config_name='config-method')
def get_args_method(conf):
    params = conf[args.method][args.dataset]
    for key in params.keys():
        args.__setattr__(key, params[key])
@hydra.main(config_path='conf', config_name='config-model')
def get_args_model(conf):
    params = conf[args.model][args.dataset]
    for key in params.keys():
        args.__setattr__(key, params[key])
get_args_global()
get_args_dataset()
get_args_model()
get_args_method()

args.cuda = torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')
if args.cuda:
    torch.cuda.set_device(args.gpu_id)
args.device = device
seed = 12345
np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)
print(args)