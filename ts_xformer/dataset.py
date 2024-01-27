from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE
import numpy as np
import torch

class KeplerDataset(Dataset):
	def __init__(self, path='/big-data/BrainProject/ryan/models/ts_xformer/data/'):
		raw = np.genfromtxt(path, delimiter=',')
		self.X = raw[1:, 1:-1]  # Remove the last datapoint to make things easy to divide
		self.y = raw[1:, 0] - 1  # Classes 1, 2 -> 0, 1
		if 'train' in path:
			over = SMOTE()
			print(self.X.shape, self.y.shape)
			self.X, self.y = over.fit_resample(self.X, self.y)
		self.X = torch.from_numpy(self.X).float()
		self.y = torch.from_numpy(self.y).float()
		print(self.X.shape, self.y.shape)
		
	def __len__(self):
		return self.y.shape[0]

	def __getitem__(self, index):
		# Convert the sequence into groups of 4 data points
		data = torch.reshape(self.X[index], (4, -1))
		data = torch.swapaxes(data, 0, 1)
		label = self.y[index]

		return data, label
		
		