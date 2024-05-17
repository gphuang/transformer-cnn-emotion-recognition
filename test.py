import os
import argparse
import numpy, random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

from src.model import parallel_all_you_want
from src.model import criterion, make_train_step
from src.config import emotions_dict

# set device to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device} selected')

X = torch.rand(100, 1, 40, 141)
y = torch.rand(100)
print(f'X_train:{X.shape}, y_train:{y.shape}')

# instantiate training tensors
X_tensor = X.to(device) 
Y_tensor = y.to(device)

# gp is here 
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (100x552 and 1064x8)

# model
model = parallel_all_you_want(len(emotions_dict)).to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)
train_step = make_train_step(model, criterion, optimizer=optimizer)
print('Number of trainable params: ',sum(p.numel() for p in model.parameters()) )

# Pass input tensors thru 1 training step (fwd+backwards pass)
loss, acc = train_step(X_tensor,Y_tensor) 

print(loss, acc)
        
