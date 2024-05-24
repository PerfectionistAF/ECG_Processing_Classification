from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import wfdb
import time
import random
from sklearn.preprocessing import minmax_scale
import sys
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Sequential, Linear, MSELoss
from torch_pso import ParticleSwarmOptimizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import copy

from ptb_data_prepare import *

class ConvNetQuake(nn.Module):##issues with keras attributes, use PyTorch
    def __init__(self):
        super(ConvNetQuake, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.Linear(1280, 128)
        #self.linear1 = nn.Linear(4096, 10)
        self.linear2 = nn.Linear(128, 1)
        #self.linear2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(32)
        self.bn7 = nn.BatchNorm1d(32)
        self.bn8 = nn.BatchNorm1d(32)
        '''self.conv1 = keras.layers.Conv1D(kernel_size=3, stride=2, padding=1, data_format=(batch, features, steps))(x)
        self.dense2 = keras.layers.Dense(5, activation="softmax")
        self.dropout = keras.layers.Dropout(0.5)'''

    def forward(self, x):  ##x is preliminary input
        x = self.bn1(F.relu((self.conv1(x))))
        x = self.bn2(F.relu((self.conv2(x))))
        x = self.bn3(F.relu((self.conv3(x))))
        x = self.bn4(F.relu((self.conv4(x))))
        x = self.bn5(F.relu((self.conv5(x))))
        x = self.bn6(F.relu((self.conv6(x))))
        x = self.bn7(F.relu((self.conv7(x))))
        x = self.bn8(F.relu((self.conv8(x))))
        #x = torch.reshape(x, (10, -1))
        x = torch.reshape(x, (16, -1))
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        '''x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)'''
        return x



model = ConvNetQuake()
model = nn.DataParallel(model, device_ids=[0])

##UNCOMMENT TO TRAIN MODEL
#optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
#pso_optimizer = ParticleSwarmOptimizer(model.parameters(),
#                               inertial_weight=0.5,
#                               num_particles=100,
#                               max_param_value=1,
#                               min_param_value=-1)
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
#    y_pred = model(batch_x)
#    
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
#                    if int(torch.round(num)) == int(torch.round(batch_y[count])):
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
#                cleaned = model(batch_x)

#                count = 0
#                acc = 0
#                for num in cleaned:
#                    if int(torch.round(num)) == int(torch.round(batch_y[count])):
#                        acc += 10
#                    count += 1
#                avg_acc += acc

            
#            acc_values_train.append((avg_acc / iterations))
#            print('Accuracy/train', (avg_acc / iterations), iters)

##convert lists to numpy arrays
#true_labels = np.array(true_labels)
#predicted_scores = np.array(predicted_scores)

##confusion matrix
#cm = confusion_matrix(true_labels, np.round(predicted_scores))
#print("Confusion Matrix:")
#print(cm)

##ROC
#fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
#roc_auc = auc(fpr, tpr)
#print("AUC:", roc_auc)

#plt.figure()
#plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC)')
#plt.legend(loc="lower right")
#plt.show()

#torch.save(model.module.state_dict(), './models/convnet-pso_multiple_channel.pth')