import os
from collections import defaultdict, namedtuple
from heapq import heappush, heappop, heapify
import struct
from pathlib import Path

import torch
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

Node = namedtuple('Node', 'freq value left right')
Node.__lt__ = lambda x, y: x.freq < y.freq

def huffman_encode(arr, prefix, save_dir='./'):
	"""
	Encodes numpy array 'arr' and saves to `save_dir`
	The names of binary files are prefixed with `prefix`
	returns the number of bytes for the tree and the data after the compression
	"""
	# Infer dtype
	dtype = str(arr.dtype)
	#print("in Encoded fun ",arr)
	#print("prefix" ,prefix)
	# Calculate frequency in arr
	freq_map = defaultdict(int)
	convert_map = {'float32':float, 'int32':int}
	for value in np.nditer(arr):
		value = convert_map[dtype](value)
		freq_map[value] += 1

	# Make heap
	heap = [Node(frequency, value, None, None) for value, frequency in freq_map.items()]
	heapify(heap)

	# Merge nodes
	while(len(heap) > 1):
		node1 = heappop(heap)
		node2 = heappop(heap)
		merged = Node(node1.freq + node2.freq, None, node1, node2)
		heappush(heap, merged)

	# Generate code value mapping
	value2code = {}

	def generate_code(node, code):
		if node is None:
			return
		if node.value is not None:
			value2code[node.value] = code
			return
		generate_code(node.left, code + '0')
		generate_code(node.right, code + '1')

	root = heappop(heap)
	generate_code(root, '')

	# Path to save location
	directory = Path(save_dir)

	# Dump data
	data_encoding = ''.join(value2code[convert_map[dtype](value)] for value in np.nditer(arr))
	#print("data Encode ",len(data_encoding))
	datasize = dump(data_encoding, directory/f'{prefix}.bin')

	# Dump codebook (huffman tree)
	codebook_encoding = encode_huffman_tree(root, dtype)
	treesize = dump(codebook_encoding, directory/f'{prefix}_codebook.bin')

	return treesize, datasize


def huffman_decode(directory, prefix, dtype):
    """
    Decodes binary files from directory
    """
    directory = Path(directory)

    # Read the codebook
    codebook_encoding = load(directory/f'{prefix}_codebook.bin')
    root = decode_huffman_tree(codebook_encoding, dtype)

    # Read the data
    data_encoding = load(directory/f'{prefix}.bin')

    # Decode
    data = []
    ptr = root
    for bit in data_encoding:
        ptr = ptr.left if bit == '0' else ptr.right
        if ptr.value is not None: # Leaf node
            data.append(ptr.value)
            ptr = root

    return np.array(data, dtype=dtype)


# Logics to encode / decode huffman tree
# Referenced the idea from https://stackoverflow.com/questions/759707/efficient-way-of-storing-huffman-tree
def encode_huffman_tree(root, dtype):
    """
    Encodes a huffman tree to string of '0's and '1's
    """
    converter = {'float32':float2bitstr, 'int32':int2bitstr}
    code_list = []
    def encode_node(node):
        if node.value is not None: # node is leaf node
            code_list.append('1')
            lst = list(converter[dtype](node.value))
            code_list.extend(lst)
        else:
            code_list.append('0')
            encode_node(node.left)
            encode_node(node.right)
    encode_node(root)
    return ''.join(code_list)


def decode_huffman_tree(code_str, dtype):
    """
    Decodes a string of '0's and '1's and costructs a huffman tree
    """
    converter = {'float32':bitstr2float, 'int32':bitstr2int}
    idx = 0
    def decode_node():
        nonlocal idx
        info = code_str[idx]
        idx += 1
        if info == '1': # Leaf node
            value = converter[dtype](code_str[idx:idx+32])
            idx += 32
            return Node(0, value, None, None)
        else:
            left = decode_node()
            right = decode_node()
            return Node(0, None, left, right)

    return decode_node()



# My own dump / load logics
def dump(code_str, filename):
    """
    code_str : string of either '0' and '1' characters
    this function dumps to a file
    returns how many bytes are written
    """
    # Make header (1 byte) and add padding to the end
    # Files need to be byte aligned.
    # Therefore we add 1 byte as a header which indicates how many bits are padded to the end
    # This introduces minimum of 8 bits, maximum of 15 bits overhead
    num_of_padding = -len(code_str) % 8
    header = f"{num_of_padding:08b}"
    code_str = header + code_str + '0' * num_of_padding

    # Convert string to integers and to real bytes
    byte_arr = bytearray(int(code_str[i:i+8], 2) for i in range(0, len(code_str), 8))

    # Dump to a file
    with open(filename, 'wb') as f:
        f.write(byte_arr)
    return len(byte_arr)


def load(filename):
    """
    This function reads a file and makes a string of '0's and '1's
    """
    with open(filename, 'rb') as f:
        header = f.read(1)
        rest = f.read() # bytes
        code_str = ''.join(f'{byte:08b}' for byte in rest)
        offset = ord(header)
        if offset != 0:
            code_str = code_str[:-offset] # string of '0's and '1's
    return code_str


# Helper functions for converting between bit string and (float or int)
def float2bitstr(f):
    four_bytes = struct.pack('>f', f) # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes) # string of '0's and '1's

def bitstr2float(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>f', byte_arr)[0]

def int2bitstr(integer):
    four_bytes = struct.pack('>I', integer) # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes) # string of '0's and '1's

def bitstr2int(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>I', byte_arr)[0]


# Functions for calculating / reconstructing index diff
def calc_index_diff(indptr):
    return indptr[1:] - indptr[:-1]

def reconstruct_indptr(diff):
	#print("File name",diff)
	#print("Cum Sum: ",np.cumsum(diff))
	return np.concatenate([[0], np.cumsum(diff)])


# Encode / Decode models
def huffman_encode_model(model, directory='encodings/'):
	print("Encoding Starts")
	os.makedirs(directory, exist_ok=True)
	original_total = 0
	compressed_total = 0
	print(f"{'Layer':<15} | {'original':>10} {'compressed':>10} {'improvement':>11} {'percent':>7}")
	print('-'*70)
	for name, param in model.named_parameters():
		if 'mask' in name:
			continue
		if 'weight' in name:
			weight = param.data.cpu().numpy()
			shape = weight.shape
			shape_x = shape
			print("shape------ :",shape)
			
			##########################################################################################################################################################
			##########################################################################################################################################################
			##########################################################################################################################################################
			##########################################################################################################################################################
			
			
			# Convert 4d/3d weight matrix,shape to 2D. 
			
			if len(shape) == 4:
				weight_x = np.reshape(weight,(weight.shape[0]*weight.shape[1],weight.shape[2]*weight.shape[3]))
				shape_x = (weight.shape[0]*weight.shape[1],weight.shape[2]*weight.shape[3])
			elif len(shape) == 3:
				weight_x = np.reshape(weight,(weight.shape[0],weight.shape[1]*weight.shape[2]))
				shape_x = (weight.shape[0],weight.shape[1]*weight.shape[2])
			else:
				weight_x = weight
				shape_x = shape
				
			
			print("Changed shape:",shape_x)
			
			
			form = 'csr' if shape_x[0] < shape_x[1] else 'csc'										#needs 2D weight/matrix
			
			
			##########################################################################################################################################################
			##########################################################################################################################################################
			##########################################################################################################################################################
			##########################################################################################################################################################
			
			
			mat = csr_matrix(weight_x) if shape_x[0] < shape_x[1] else csc_matrix(weight_x)				#needs 2D weight/matrix
			
			print("matrix indices shape: --------------------- ",mat.indices.shape)
			print("matrix indptr shape: --------------------- ",mat.indptr.shape)
			print("matrix indptr shape: --------------------- ",mat.data.shape)


			##########################################################################################################################################################
			##########################################################################################################################################################
			##########################################################################################################################################################
			##########################################################################################################################################################			
			
			# Encode
			t0, d0 = huffman_encode(mat.data, name+f'_{form}_data', directory)
			t1, d1 = huffman_encode(mat.indices, name+f'_{form}_indices', directory)
			t2, d2 = huffman_encode(calc_index_diff(mat.indptr), name+f'_{form}_indptr', directory)
			print("******************NameEncode**************",name+f'_{form}_indptr')
			print("t2,d2",t2,d2)
			# Print statistics
			original = mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
			compressed = t0 + t1 + t2 + d0 + d1 + d2
			
			print(f"{name:<15} | {original:10} {compressed:10} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}%")
		else: # bias
			# Note that we do not huffman encode bias
			bias = param.data.cpu().numpy()
			bias.dump(f'{directory}/{name}')
			
			# Print statistics
			original = bias.nbytes
			compressed = original
			
			print(f"{name:<15} | {original:10} {compressed:10} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}%")
		original_total += original
		compressed_total += compressed	
	print('-'*70)
	print(f"{'total':15} | {original_total:>10} {compressed_total:>10} {original_total / compressed_total:>10.2f}x {100 * compressed_total / original_total:>6.2f}%")
	return model
	
	
	

def huffman_decode_model(model, directory='encodings/'):
	#print(model)
	print("\n")
	print("\n")
	print("In huffman decode model")
	for name, param in model.named_parameters():
		if 'mask' in name:
			continue
		if 'weight' in name:
			#print("-------------------Name--------------------",name)
			dev = param.device
			weight = param.data.cpu().numpy()
			shape = weight.shape
			print("----------------------- Shape ----------------------- ",shape)
			
			##########################################################################################################################################################
			##########################################################################################################################################################
			##########################################################################################################################################################
			##########################################################################################################################################################
			#print("data shape after encode",data)
			
			new_shape = ()
			#Reshaping back to same format as original
			if len(shape) == 4:
				#shape = shape.reshape(weight.shape[0]*weight.shape[1],weight.shape[2]*weight.shape[3])
				new_shape = (weight.shape[0]*weight.shape[1],weight.shape[2]*weight.shape[3])
			elif len(shape) == 3:
				#shape = shape.reshape(weight.shape[0],weight.shape[1]*weight.shape[2])
				new_shape = (weight.shape[0],weight.shape[1]*weight.shape[2])
			else:
				new_shape = shape

			print("----------------------- New Shape ----------------------- ",new_shape)
			
			form = 'csr' if new_shape[0] < new_shape[1] else 'csc'
			
			min_shape = new_shape[0] if new_shape[0] < new_shape[1] else new_shape[1]
			max_shape = new_shape[0] if new_shape[0] > new_shape[1] else new_shape[1]
			
			matrix = csr_matrix if new_shape[0] < new_shape[1] else csc_matrix
			
			
			
			#print("----------------------------------shapeOfMatrix-------------------------",matrix.shape)
			
			
			
			
			
			
			# Decode data
			data = huffman_decode(directory, name+f'_{form}_data', dtype='float32')
			indices = huffman_decode(directory, name+f'_{form}_indices', dtype='int32')
			indptr = reconstruct_indptr(huffman_decode(directory, name+f'_{form}_indptr', dtype='int32'))
			print("******************Name**************",name+f'_{form}_indptr')
			
			
			##########################################################################################################################################################
			##########################################################################################################################################################
			##########################################################################################################################################################
			##########################################################################################################################################################
			
			
			'''
			print("matrix shape",matrix)
			print("data shape :",data.shape)
			print("shape of matrix :",weight.shape)'''
			

			##########################################################################################################################################################
			##########################################################################################################################################################
			##########################################################################################################################################################
			##########################################################################################################################################################
			
			
			
			print("----------------------- data shape ----------------------- ",data.shape)
			print("----------------------- changed shape ----------------------- ",new_shape)
			print("----------------------- indices shape ----------------------- ",indices.shape)
			print("----------------------- indptr ----------------------- ",indptr.shape)


			##########################################################################################################################################################
			##########################################################################################################################################################
			##########################################################################################################################################################
			##########################################################################################################################################################

			if 'conv' in name:
				indptr = np.arange(0,(min_shape*max_shape)+1,max_shape)
				print("----------------------- Chnagedindptr ----------------------- ",indptr.shape)
			# Construct matrix
			mat = matrix((data, indices, indptr), new_shape)
			mat = mat.toarray()
			print("----------------------- MatDecodeShape ----------------------- ",mat.shape)
			print("----------------------- mat type ----------------------- ",type(mat))
			print("----------------------- paramShape -----------------------",param.shape)
			print("----------------------- WeightShape -----------------------",weight.shape)
			
			#Reshaping back to same format as original
			if len(shape) == 4:
				mat = mat.reshape(weight.shape[0],weight.shape[1],weight.shape[2],weight.shape[3])
			elif len(shape) == 3:
				mat = mat.reshape(weight.shape[0],weight.shape[1],weight.shape[2])
			
			# Insert to model
			param.data = torch.from_numpy(mat).to(dev)			#toarray() will transpose the matrix in right format. in which it is needed.
		else:
			dev = param.device
			bias = np.load(directory+'/'+name)
			param.data = torch.from_numpy(bias).to(dev)
	return model
