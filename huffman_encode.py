'''
    This code takes as input by parameter the model after weight sharing and performs huffman encoding on it.
    It calls the huffman_encode__model() present in the net folder, which stores the encoded model in the encodings folder.
    We noted that the model reduced from 1.2MB to 94KB from huffman encoding 
'''


import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.module import Module
from torchvision import datasets, transforms

import util
from net.huffmancoding import huffman_decode_model, huffman_encode_model
from net.models import LeNet_5
from net.quantization import apply_weight_sharing
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Huffman encode a quantized model')
#parser.add_argument('model', type=str, help='saved quantized model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')


##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################

# Encoding Starts
print("\n\n\n")
print("In Encoding Function")
model = torch.load('saves/model_after_weight_sharing.ptmodel')
model = huffman_encode_model(model)				# Encoding the model function
torch.save(model,'saves/after-huffman.pth')


##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
'''
Decoding the model 
model = torch.load(args.model)
model = huffman_decode_model(model)
print(model)

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

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

model = model.to(device)

kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=2048, shuffle=False, **kwargs)

accuracy = test()
#util.log(args.log, f"accuracy_after_huffman  {accuracy}")
print(" accuracy after huffman ", accuracy)
'''
