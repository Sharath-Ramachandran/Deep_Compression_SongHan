import argparse

import torch

from net.huffmancoding import huffman_encode_model, huffman_decode_model
import util

#import argparse
import os

import numpy as np
#import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.nn.modules.module import Module
from net.models import LeNet_5
from net.quantization import apply_weight_sharing

parser = argparse.ArgumentParser(description='Huffman encode a quantized model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
args = parser.parse_args()


model = torch.load('saves/after-huffman.pth')
model = huffman_decode_model(model)
print()
print()
print()
print("----------------------- The model is -----------------------",model)

#model = torch.load(args.model)
#model = huffman_decode_model(model)
#print(model)
'''
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy
'''
use_cuda = not args.no_cuda and torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else 'cpu')

#model = model.to(device)

#kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
'''test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=2048, shuffle=False, **kwargs)
'''
accuracy = util.test(model, use_cuda)
util.log(args.log, f"accuracy_after_huffman  {accuracy}")
#print(" accuracy after huffman ", accuracy)
