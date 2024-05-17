import os, sys, glob
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import sample_rate, data_path, emotions_dict
from src.dataset import augment_waveforms, get_features, load_data

import argparse
def parse_args():

    parser = argparse.ArgumentParser(description='Prepare data.')

    parser.add_argument('--data_dir', type=str, default='./data/spkr-dep',
                        help=f'Specify the dir to save the features+labels file.')
    parser.add_argument('--agwn_augment', action='store_true',
                        help='Specify whether to augment data with Gaussian white noise (default: False).')
    parser.add_argument('--win_len', type=int, default=200,
                        help='Specify the window length for segmentation (default: 200 frames).')
    
    args = parser.parse_args()
    return args
    
def main(args):
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

    # partion w.r.t. emotion, speaker-dependant 
    # create storage for train, validation, test sets and their indices
    train_set,valid_set,test_set = [],[],[]
    X_train,X_valid,X_test = [],[],[]
    y_train,y_valid,y_test = [],[],[]

    # process each emotion separately to make sure we builf balanced train/valid/test sets 
    for emotion_num in range(len(emotions_dict)):
            
        # find all indices of a single unique emotion
        emotion_indices = [index for index, emotion in enumerate(emotions) if emotion==emotion_num]

        # seed for reproducibility 
        np.random.seed(69)
        # shuffle indicies 
        emotion_indices = np.random.permutation(emotion_indices)

        # store dim (length) of the emotion list to make indices
        dim = len(emotion_indices)

        # store indices of training, validation and test sets in 80/10/10 proportion
        # train set is first 80%
        train_indices = emotion_indices[:int(0.8*dim)]
        # validation set is next 10% (between 80% and 90%)
        valid_indices = emotion_indices[int(0.8*dim):int(0.9*dim)]
        # test set is last 10% (between 90% - end/100%)
        test_indices = emotion_indices[int(0.9*dim):]

        # create train waveforms/labels sets
        X_train.append(waveforms[train_indices,:])
        y_train.append(np.array([emotion_num]*len(train_indices),dtype=np.int32))
        # create validation waveforms/labels sets
        X_valid.append(waveforms[valid_indices,:])
        y_valid.append(np.array([emotion_num]*len(valid_indices),dtype=np.int32))
        # create test waveforms/labels sets
        X_test.append(waveforms[test_indices,:])
        y_test.append(np.array([emotion_num]*len(test_indices),dtype=np.int32))

        # store indices for each emotion set to verify uniqueness between sets 
        train_set.append(train_indices)
        valid_set.append(valid_indices)
        test_set.append(test_indices)

    # concatenate, in order, all waveforms back into one array 
    X_train = np.concatenate(X_train,axis=0)
    X_valid = np.concatenate(X_valid,axis=0)
    X_test = np.concatenate(X_test,axis=0)

    # concatenate, in order, all emotions back into one array 
    y_train = np.concatenate(y_train,axis=0)
    y_valid = np.concatenate(y_valid,axis=0)
    y_test = np.concatenate(y_test,axis=0)

    # combine and store indices for all emotions' train, validation, test sets to verify uniqueness of sets
    train_set = np.concatenate(train_set,axis=0)
    valid_set = np.concatenate(valid_set,axis=0)
    test_set = np.concatenate(test_set,axis=0)

    # check shape of each set
    print(f'X_train:{X_train.shape}, y_train:{y_train.shape}')
    print(f'X_valid:{X_valid.shape}, y_valid:{y_valid.shape}')
    print(f'X_test:{X_test.shape}, y_test:{y_test.shape}')

    # make sure train, validation, test sets have no overlap/are unique
    # get all unique indices across all sets and how many times each index appears (count)
    uniques, count = np.unique(np.concatenate([train_set,test_set,valid_set],axis=0), return_counts=True)

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

    if args.agwn_augment:
        # Augmented data
        multiples = 2

        print('Train waveforms:') # augment waveforms of training set
        features_train , y_train = augment_waveforms(X_train, features_train, y_train, multiples, sample_rate=sample_rate)

        print('\n\nValidation waveforms:') # augment waveforms of validation set
        features_valid, y_valid = augment_waveforms(X_valid, features_valid, y_valid, multiples, sample_rate=sample_rate)

        #print('\n\nTest waveforms:') # augment waveforms of test set 
        #features_test, y_test = augment_waveforms(X_test, features_test, y_test, multiples, sample_rate=sample_rate)

        # Check new shape of extracted features and data:
        print(f'\n\nNative + Augmented Features set: {len(features_train)+len(features_test)+len(features_valid)} total, {len(features_train)} train, {len(features_valid)} validation, {len(features_test)} test samples')
        print(f'{len(y_train)} training sample labels, {len(y_valid)} validation sample labels, {len(y_test)} test sample labels')
        print(f'Features (MFCC matrix) shape: {len(features_train[0])} mel frequency coefficients x {len(features_train[0][1])} time steps')

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

    # save file 
    filename = os.path.join(args.data_dir, 'features+labels.npy')

    # open file in write mode and write data
    with open(filename, 'wb') as f:
        np.save(f, X_train)
        np.save(f, X_valid)
        np.save(f, X_test)
        np.save(f, y_train)
        np.save(f, y_valid)
        np.save(f, y_test)

    print(f'Features and labels saved to {filename}')

if __name__ == '__main__':
    args = parse_args()

    import pathlib
    p = pathlib.Path(args.data_dir)
    p.mkdir(parents=True, exist_ok=True)

    main(args)