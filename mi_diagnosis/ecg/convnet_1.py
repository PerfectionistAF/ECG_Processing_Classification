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

#optimizer = torch.optim.Adam(model_1.parameters(), lr=1.0e-4)
#criterion = nn.BCELoss()

##run the model now
#num_iters = 5000
#batch_size = 16

#acc_values = []
#acc_values_train = []
#avg_loss = []

#true_labels = []
#predicted_scores = []
#for iters in range(num_iters):
#    batch_x, batch_y = get_batch(batch_size, split='train')
#    y_pred = model_1(batch_x)
    
#    def closure():
        # Clear any grads from before the optimization step, since we will be changing the parameters
#        optimizer.zero_grad()  
#        return criterion(y_pred, batch_y)
    
#    loss = criterion(y_pred, batch_y)
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step(closure)
    
    # Validation
#    if iters%100 == 0 and iters != 0:
#        print('Loss/train', loss, iters)
#        avg_loss.append(loss)
        
#        with torch.no_grad():

            # test_set
#            iterations = 100
#            val_true_labels = []
#            val_predicted_scores = []
#            avg_acc = 0

#            for _ in range(iterations):
#                batch_x, batch_y = get_batch(batch_size, split='val')
#                cleaned = model(batch_x)

#                count = 0
#                acc = 0
#                for  num, true_label in zip(cleaned, batch_y):
#                    if int(torch.round(num)) == int(torch.round(true_label)):
#                        acc += 10
#                    count += 1
#                    val_true_labels.append(int(torch.round(true_label)))
#                    val_predicted_scores.append(float(num))
#                avg_acc += acc

#            acc_values.append((avg_acc / iterations))
#            print('Accuracy/val', (avg_acc / iterations), iters)
            
            ##accumulate true labels and predicted scores
                #for num, true_label in zip(cleaned, batch_y):
                #    val_true_labels.append(int(torch.round(true_label)))
                #    val_predicted_scores.append(float(num))

#            true_labels.extend(val_true_labels)
#            predicted_scores.extend(val_predicted_scores)
            
            # train_set
#            iterations = 100
#            avg_acc = 0

#            for _ in range(iterations):
#                batch_x, batch_y = get_batch(batch_size, split='train')
#                cleaned = model_1(batch_x)

#                count = 0
#                acc = 0
#                for num in cleaned:
#                    if int(torch.round(num)) == int(torch.round(batch_y[count])):
#                        acc += 10
#                    count += 1
 #               avg_acc += acc

            
#            acc_values_train.append((avg_acc / iterations))
#            print('Accuracy/train', (avg_acc / iterations), iters)

##convert lists to numpy arrays
#true_labels = np.array(true_labels)
#predicted_scores = np.array(predicted_scores)

##confusion matrix
#cm = confusion_matrix(true_labels, np.round(predicted_scores))
#print("Confusion Matrix:")
#print(cm)
