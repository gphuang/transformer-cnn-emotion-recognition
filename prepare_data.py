import os, sys, glob
import numpy as np

from src.config import sample_rate, data_path
from src.config import emotions_dict
from src.dataset import get_features, feature_mfcc
from src.dataset import load_data
    
# load data 
waveforms, emotions, intensities, genders, actors = load_data(data_path=data_path, 
                                                              duration=3, 
                                                              sample_rate=sample_rate)

## Check extracted audio waveforms and labels: 1440 labels, 1440 waveforms, each of length 3sec * 48k = 144k
print(f'Waveforms set: {len(waveforms)} samples')
print(f'Waveform signal length: {len(waveforms[0])}')
print(f'Emotions set: {len(emotions)} sample labels')

# convert waveforms to array for processing
waveforms = np.array(waveforms)
emotions = np.array(emotions)

# partion from MMEmotionRecognition and mmerr
# see: multimodal-emotion-recognition-ravdess/ravdess_preprocessing/create_annotations.py
folds = [[[1, 4, 9, 22],[2, 5, 14, 15, 16],[3, 6, 7, 13, 18, 10, 11, 12, 19, 20, 8, 17, 21, 23, 24]], ]
folds = [[[0,1,2,3],[4,5,6,7],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]], ]
test_ids, valid_ids, train_ids = folds[0]

# seed for reproducibility 
np.random.seed(69)
test_indices = [i for i, actor in enumerate(actors) if int(actor) in test_ids]
valid_indices = [i for i, actor in enumerate(actors) if int(actor) in valid_ids]
train_indices = [i for i, actor in enumerate(actors) if int(actor) in train_ids]

# create train waveforms/labels sets
X_test = waveforms[test_indices]
y_test = emotions[test_indices]
X_valid = waveforms[valid_indices]
y_valid = emotions[valid_indices]
X_train = waveforms[train_indices]
y_train = emotions[train_indices]

# check shape of each set
print(f'X_train:{X_train.shape}, y_train:{y_train.shape}')
print(f'X_valid:{X_valid.shape}, y_valid:{y_valid.shape}')
print(f'X_test:{X_test.shape}, y_test:{y_test.shape}')

# make sure train, validation, test sets have no overlap/are unique
# get all unique indices across all sets and how many times each index appears (count)
uniques, count = np.unique(np.concatenate([test_indices,valid_indices,train_indices],axis=0), return_counts=True)

# if each index appears just once, and we have 1440 such unique indices, then all sets are unique
if sum(count==1) == len(emotions):
    print(f'\nSets are unique: {sum(count==1)} samples out of {len(emotions)} are unique')
else:
    print(f'\nSets are NOT unique: {sum(count==1)} samples out of {len(emotions)} are unique')

## Extract Features
print('Train waveforms:') # get training set features 
features_train = get_features(X_train, sample_rate=sample_rate)

print('\n\nValidation waveforms:') # get validation set features
features_valid = get_features(X_valid, sample_rate=sample_rate)

print('\n\nTest waveforms:') # get test set features 
features_test = get_features(X_test, sample_rate=sample_rate)

print(f'\n\nFeatures set: {len(features_train)+len(features_test)+len(features_valid)} total, {len(features_train)} train, {len(features_valid)} validation, {len(features_test)} test samples')
print(f'Features (MFC coefficient matrix) shape: {len(features_train[0])} mel frequency coefficients x {len(features_train[0][1])} time steps')

# make dummy input channel for CNN input feature tensor
X_train = np.expand_dims(features_train,1)
X_valid = np.expand_dims(features_valid, 1)
X_test = np.expand_dims(features_test,1)

# convert emotion labels from list back to numpy arrays for PyTorch to work with 
y_train = np.array(y_train)
y_valid = np.array(y_valid)
y_test = np.array(y_test)

# tensor-ready 4D data array (batch, channel, width, height) == (4320, 1, 128, 282) when multiples==2
print(f'Shape of 4D input tensor: {X_train.shape} train, {X_valid.shape} validation, {X_test.shape} test')
print(f'Shape of emotion labels: {y_train.shape} train, {y_valid.shape} validation, {y_test.shape} test')

# free up some RAM - no longer need full feature set or any waveforms 
del features_train, features_valid, features_test, waveforms

## Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
N,C,H,W = X_train.shape
# Reshape to 1D because StandardScaler operates on a 1D array
# tell numpy to infer shape of 1D array with '-1' argument
X_train = np.reshape(X_train, (N,-1)) 
X_train = scaler.fit_transform(X_train)
# Transform back to NxCxHxW 4D tensor format
X_train = np.reshape(X_train, (N,C,H,W))

# Scale the validation set
N,C,H,W = X_valid.shape
X_valid = np.reshape(X_valid, (N,-1))
X_valid = scaler.transform(X_valid)
X_valid = np.reshape(X_valid, (N,C,H,W))

# Scale the test set
N,C,H,W = X_test.shape
X_test = np.reshape(X_test, (N,-1))
X_test = scaler.transform(X_test)
X_test = np.reshape(X_test, (N,C,H,W))

# check shape of each set again
print(f'X_train scaled:{X_train.shape}, y_train:{y_train.shape}')
print(f'X_valid scaled:{X_valid.shape}, y_valid:{y_valid.shape}')
print(f'X_test scaled:{X_test.shape}, y_test:{y_test.shape}')

###### SAVE #########
# choose save file name 
filename = 'features+labels.npy'

# open file in write mode and write data
with open(filename, 'wb') as f:
    np.save(f, X_train)
    np.save(f, X_valid)
    np.save(f, X_test)
    np.save(f, y_train)
    np.save(f, y_valid)
    np.save(f, y_test)

print(f'Features and labels saved to {filename}')
