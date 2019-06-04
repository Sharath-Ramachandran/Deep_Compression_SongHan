import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix


def apply_weight_sharing(model, bits=5):
	"""
	Applies weight sharing to the given model
	"""
	#print(model)
	for module in model.children():
		#print(module)
		dev = module.weight.device
		weight = module.weight.data.cpu().numpy()
		weight_x = weight
		shape = weight.shape
		
		# Convert 4d/3d weight matrix to 2D. 
		if len(shape) == 4:
			weight_x = np.reshape(weight,(weight.shape[0]*weight.shape[1],weight.shape[2]*weight.shape[3]))
		elif len(shape) == 3:
			weight_x = np.reshape(weight,(weight.shape[0],weight.shape[1]*weight.shape[2]))
		
		#print("shape of weight :",shape)
		#converting 2D weight matrix to CSR/CSC format. 
		mat = csr_matrix(weight_x) if shape[0] < shape[1] else csc_matrix(weight_x)
		min_ = min(mat.data)
		max_ = max(mat.data)
		space = np.linspace(min_, max_, num=2**bits)
		
		#Clustering of weight matrix
		kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
		kmeans.fit(mat.data.reshape(-1,1))
		new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
		mat.data = new_weight
		mat = mat.toarray()		# mat is in csc format so we need to convert it to numpy.
		#print(type(mat))
		
		#Reshaping back to same format as original
		if len(shape) == 4:
			mat = mat.reshape(weight.shape[0],weight.shape[1],weight.shape[2],weight.shape[3])
		elif len(shape) == 3:
			mat = mat.reshape(weight.shape[0],weight.shape[1],weight.shape[2])
		module.weight.data = torch.from_numpy(mat).to(dev)


