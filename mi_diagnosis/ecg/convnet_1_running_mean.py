import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
import random

window_size = 10000  ##from each ecg reading

class ConvNet1(nn.Module):
    def __init__(self, num_features=32):
        super(ConvNet1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.Linear(32 * (window_size // 2), 1)  # Adjusted input size for the linear layer
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(32)
        # Set weight and bias to 27
        self.bn1.weight = nn.Parameter(torch.full((32,), 27, dtype=torch.float, requires_grad=True))
        self.bn1.bias = nn.Parameter(torch.full((32,), 27, dtype=torch.float, requires_grad=True))

    def forward(self, x):  
        x = self.bn1(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Changed torch.reshape to view to adapt to batch size
        x = self.linear1(x)
        x = self.sigmoid(x)
        return x



def get_batch(batch_size, split='train'):
    ##random sampling of batches
    ##divide batch number in half to improve speed of network
    if split == 'train':
        ##random samples from sorted per batch with size k = batch_size / 2
        ##mimic shuffling
        unhealthy_indices = random.sample(sorted(np.arange(len(data_unhealthy_train))), k=int(batch_size / 2))
        healthy_indices = random.sample(sorted(np.arange(len(data_healthy_train))), k=int(batch_size / 2))
        unhealthy_batch = []
        healthy_batch = []
        for i in unhealthy_indices:
            unhealthy_batch.append(data_unhealthy_train[i])
        for j in healthy_indices:
            healthy_batch.append(data_healthy_train[j])
    elif split == 'val': 
        unhealthy_indices = random.sample(sorted(np.arange(len(data_unhealthy_val))), k=int(batch_size / 2))
        healthy_indices = random.sample(sorted(np.arange(len(data_healthy_val))), k=int(batch_size / 2))
        unhealthy_batch = []
        healthy_batch = []
        for i in unhealthy_indices:
            unhealthy_batch.append(data_unhealthy_val[i])
        for j in healthy_indices:
            healthy_batch.append(data_healthy_val[j])

    
    batch_x = []  ##batch of mixed healthy and unhealthy data
    for sample in unhealthy_batch: ##if val or if train
        start = random.choice(np.arange(len(sample[0]) - window_size))  ##randomly sample window from ecg 
        # normalize ecg values 
        normalized_1 = minmax_scale(sample[0][start:start+window_size])
        normalized_2 = minmax_scale(sample[1][start:start+window_size])
        normalized_3 = minmax_scale(sample[2][start:start+window_size])
        normalized = np.array((normalized_1, normalized_2, normalized_3))
        batch_x.append(normalized)
        
    for sample in healthy_batch:
        start = random.choice(np.arange(len(sample[0]) - window_size))
        # normalize
        normalized_1 = minmax_scale(sample[0][start:start+window_size])
        normalized_2 = minmax_scale(sample[1][start:start+window_size])
        normalized_3 = minmax_scale(sample[2][start:start+window_size])
        normalized = np.array((normalized_1, normalized_2, normalized_3))
        batch_x.append(normalized)
    
    batch_y = [0.1 for _ in range(int(batch_size / 2))]
    for _ in range(int(batch_size / 2)):
        batch_y.append(0.9)
        
    indices = np.arange(len(batch_y))
    np.random.shuffle(indices)
    
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    
    batch_x = batch_x[indices]
    batch_y = batch_y[indices]
    
    batch_x = np.reshape(batch_x, (-1, 3, window_size))
    batch_x = torch.from_numpy(batch_x)
    batch_x = batch_x.float()#.cuda()
    
    batch_y = np.reshape(batch_y, (-1, 1))
    batch_y = torch.from_numpy(batch_y)  ##save tensors
    batch_y = batch_y.float()#.cuda()
    
    return batch_x, batch_y


model_1 = ConvNet1()
model_1 = nn.DataParallel(model_1, device_ids=[0])
optimizer = torch.optim.Adam(model_1.parameters(), lr=1.0e-4)
criterion = nn.BCELoss()
