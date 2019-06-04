''' This file will run all the 3 methods of Deep compression presented by Song Han in "https://arxiv.org/abs/1510.00149" . 
Our implementation would work perform deep compression on the sequential models.
'''
#TODO: Need to check for BatchNorm and DropOuts
#Future Work:Implement the same for residual models
import pruning
import weight_share
import huffman_encode
import huffman_decode

'''
from subprocess import Popen    
Popen('python pruning.py')
'''
