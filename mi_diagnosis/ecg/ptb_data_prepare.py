## Many neural networks we tested: AlexNet, ResNet50, Gradient Boosting CNN, ConvNetQuake
## the best one so far is the ConvQuakeNet then ResNet50
## Use this file to prepare data for a parallelized model
##start by processing the data
##then create the nn.Model subclass
##then call the Pso(forward function of model)
##then return loss score; ie optimize hyperparameters for search convergence
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import wfdb
import time
import random
from sklearn.preprocessing import minmax_scale
import sys
import torch

seed_num = 39
channel_1 = 'v6'
channel_2 = 'vz'
channel_3 = 'ii'


with open('./ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/RECORDS') as fp:  
    lines = fp.readlines()


files_myocardial = []
files_healthy = []

for file in lines:
    file_path = './ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/' + file[:-1] + '.hea'
    
    ##read header to determine class ##focus on myocardial infarction
    if 'Myocardial infarction' in open(file_path).read():
        files_myocardial.append(file)
        
    if 'Healthy control' in open(file_path).read():
        files_healthy.append(file)

np.random.seed(int(seed_num))
np.random.shuffle(files_myocardial) 
np.random.shuffle(files_healthy)

healthy_train = files_healthy[:int(0.8*len(files_healthy))]
healthy_val = files_healthy[int(0.8*len(files_healthy)):]
myocardial_train = files_myocardial[:int(0.8*len(files_myocardial))]
myocardial_val = files_myocardial[int(0.8*len(files_myocardial)):]


def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 


patient_ids_healthy_train = []
patient_ids_healthy_val = []
patient_ids_myocardial_train = []
patient_ids_myocardial_val = []
##extract first 10 letters in file path string, which is the patient id
for index in healthy_train:
    patient_ids_healthy_train.append(index[0:10])
for index in healthy_val:
    patient_ids_healthy_val.append(index[0:10])
for index in myocardial_train:
    patient_ids_myocardial_train.append(index[0:10])
for index in myocardial_val:
    patient_ids_myocardial_val.append(index[0:10])


intersection_myocardial = intersection(patient_ids_myocardial_train, patient_ids_myocardial_val)
intersection_healthy = intersection(patient_ids_healthy_train, patient_ids_healthy_val)

move_to_train = intersection_myocardial[:int(0.5*len(intersection_myocardial))]
move_to_val = intersection_myocardial[int(0.5*len(intersection_myocardial)):]

for patient_id in move_to_train:
    in_val = []
    ##find and remove all files in val ecg readings by id
    for file_ in myocardial_val:
        if file_[:10] == patient_id:
            in_val.append(file_)
            myocardial_val.remove(file_)
            
    ##add to train
    for file_ in in_val:
        myocardial_train.append(file_)

for patient_id in move_to_val:    
    in_train = []
    ##find and remove all files in train
    for file_ in myocardial_train:
        if file_[:10] == patient_id:
            in_train.append(file_)
            myocardial_train.remove(file_)
            
    ##add to val
    for file_ in in_train:
        myocardial_val.append(file_)

move_to_train = intersection_healthy[:int(0.5*len(intersection_healthy))]
move_to_val = intersection_healthy[int(0.5*len(intersection_healthy)):]

for patient_id in move_to_train:
    in_val = []
    ##find and remove all files in val ecg readings by id
    for file_ in healthy_val:
        if file_[:10] == patient_id:
            in_val.append(file_)
            healthy_val.remove(file_)
            
    ##add to train
    for file_ in in_val:
        healthy_train.append(file_)

for patient_id in move_to_val:    
    in_train = []
    ##find and remove all files in train
    for file_ in healthy_train:
        if file_[:10] == patient_id:
            in_train.append(file_)
            healthy_train.remove(file_)
            
    ##add to val
    for file_ in in_train:
        healthy_val.append(file_)

data_healthy_train = []
for file in healthy_train:
    ##records for each
    data_ii, _ = wfdb.rdsamp('./ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/' + file[:-1], channel_names=[str(channel_1)])
    data_v6, _ = wfdb.rdsamp('./ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/' + file[:-1], channel_names=[str(channel_2)])
    data_vz, _ = wfdb.rdsamp('./ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/' + file[:-1], channel_names=[str(channel_3)])
    data = [data_ii.flatten(), data_v6.flatten(), data_vz.flatten()]  ##flatten to input directly into keras model
    data_healthy_train.append(data)

data_healthy_val = []
for file in healthy_val:
    ##records for each
    data_ii, _ = wfdb.rdsamp('./ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/' + file[:-1], channel_names=[str(channel_1)])
    data_v6, _ = wfdb.rdsamp('./ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/' + file[:-1], channel_names=[str(channel_2)])
    data_vz, _ = wfdb.rdsamp('./ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/' + file[:-1], channel_names=[str(channel_3)])
    data = [data_ii.flatten(), data_v6.flatten(), data_vz.flatten()]  ##flatten to input directly into keras model
    data_healthy_val.append(data)

data_unhealthy_train = []
for file in myocardial_train:
    ##records for each
    data_ii, _ = wfdb.rdsamp('./ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/' + file[:-1], channel_names=[str(channel_1)])
    data_v6, _ = wfdb.rdsamp('./ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/' + file[:-1], channel_names=[str(channel_2)])
    data_vz, _ = wfdb.rdsamp('./ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/' + file[:-1], channel_names=[str(channel_3)])
    data = [data_ii.flatten(), data_v6.flatten(), data_vz.flatten()]  ##flatten to input directly into keras model
    data_unhealthy_train.append(data)

data_unhealthy_val = []
for file in myocardial_val:
    ##records for each
    data_ii, _ = wfdb.rdsamp('./ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/' + file[:-1], channel_names=[str(channel_1)])
    data_v6, _ = wfdb.rdsamp('./ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/' + file[:-1], channel_names=[str(channel_2)])
    data_vz, _ = wfdb.rdsamp('./ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/' + file[:-1], channel_names=[str(channel_3)])
    data = [data_ii.flatten(), data_v6.flatten(), data_vz.flatten()]  ##flatten to input directly into keras model
    data_unhealthy_val.append(data)

window_size = 10000  ##from each ecg reading
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
    batch_x = batch_x.float()
    
    batch_y = np.reshape(batch_y, (-1, 1))
    batch_y = torch.from_numpy(batch_y)  ##save tensors
    batch_y = batch_y.float()#.cuda()
    batch_y = batch_y.float()
    
    return batch_x, batch_y