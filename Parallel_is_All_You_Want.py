# -*- coding: utf-8 -*-
"""Parallel_is_all_you_want.ipynb

# Parallel is All You Want: Combining Spatial and Temporal Feature Representions of Speech Emotion by Parallelizing CNNs and Transformer-Encoders
# Abstract
In this notebook, I'm going to build upon my [Intro to Speech Audio Classification repo](https://github.com/IliaZenkov/sklearn-audio-classification) and build two parallel convolutional neural networks (CNN) in parallel with a Transformer encoder network to classify audio data. We're working on the [RAVDESS dataset](https://smartlaboratory.org/ravdess/) to classify emotions from one of 8 classes. We combine the CNN for spatial feature representation and the Transformer for temporal feature representation. We augment the training data by increasing variation in the dataset to reduce overfitting; we use Additive White Gaussian Noise (AWGN) to augment the RAVDESS dataset three-fold for a total of 4320 audio samples.

We harness the image-classification and spatial feature representation power of the CNN by treating MFCC plots as grayscale images; their width is a time scale, their height is a frequency scale. The value of each pixel in the MFCC is the intensity of the audio signal at a particular range of mel frequencies at a time step. 

Because of the sequential nature of the data, we will also use the Transformer to try and model as accurately as possible the temporal relationships between pitch transitions in emotions.  

This notebook takes inspirations from a variety of recent advances in deep learning and network architectures; in particular, stacked and parallel CNN networks combined with multi-head self-attention layers from the Transformer Encoder. I hypothesize that the expansion of CNN filter channel dimensions and reduction of feature maps will provide the most expressive feature representation at the lowest computational cost, while the Transformer-Encoder is used with the hypothesis that the network will learn to predict frequency distributions of different emotions according to the global structure of the MFCC plot (and indirectly, mel spectrogram) of each emotion. **With the strength of the CNN in spatial feature representation and Transformer in sequence encoding, I manage to achieve a 97% accuracy on a hold-out set from the RAVDESS dataset.**

<!--TABLE OF CONTENTS-->
# Table of Contents
- [Introduction](#Introduction)
  - [Define features](#Define-features)
  - [Load Data and Extract Features](#Load-Data-and-Extract-Features)
  - [Augmenting the Data with AWGN: Additive White Gaussian Noise](#Augmenting-the-Data-with-AWGN---Additive-White-Gaussian-Noise)
  - [Format Data into Tensor-Ready 4D Arrays](#Format-Data-into-Tensor-Ready-4D-Arrays)
  - [Split into Train/Validation/Test Sets](#Split-into-Train/Validation/Test-Sets)
  - [Feature Scaling](#Feature-Scaling)
- [Architecture Overview](#Architecture-Overview)
- [CNN Motivation](#CNN-Motivation)
- [Transformer-Encoder Motivation](#Transformer-Encoder-Motivation)
- [Building Model Architecture and Forward Pass](#Build-Model-Architecture-and-Define-Forward-Pass)
  - [Analyzing The Flow of Tensors Through the Network](#Analyzing-The-Flow-of-Tensors-Through-the-Network)
  - [Choosing Loss Function](#Define-Loss/Criterion)
  - [Choosing Optimizer](#Choose-Optimizer)
  - [Build Training Step](#Define-Training-Step)
  - [Build Validation Step](#Define-Validation-Step)
  - [Make Checkpoint Functions](#Make-Checkpoint-Functions)
- [Build Training Loop](#Build-Training-Loop)
  - [Train Model](#Train-It)
- [Check the Loss Curve's Behaviour](#Check-the-Loss-Curve's-Behaviour)
- [Evaluate Performance on Test Set](#Evaluate-Performance-on-Test-Set)
- [Conclusion](#Conclusion)
- [References](#References)


# Introduction
From my previous notebook: "Long-Short-Term-Memory Recurrent Neural Networks (LSTM RNNs) and Convolutional Neural Networks (CNNs) are excellent DNN candidates for audio data classification: LSTM RNNs because of their excellent ability to interpret sequential data such as features of the audio waveform represented as a time series; CNNs because features engineered on audio data such as spectrograms have marked resemblance to images, in which CNNs excel at recognizing and discriminating between distinct patterns." - Me 

I'm going to build on that - CNNs are still the hallmark of image classification today, although even in this domain Transformers are beginning to take the main stage: A [2021 ICLR submission: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy) claims they've implemented a Transformer for image classification that outperforms a state of the art CNN, and at a much lower computational complexity.

In addition to taking inspiration from the above, it's also no longer 2015 - so instead of the LSTM-RNN I'm going to implement its successor the Transformer model in parallel with a CNN to try and get state-of-the-art performance on the RAVDESS dataset. 

Other motivations for the architecture of this model come from a variety of papers from the past few years. **The most notable inspirations are:**
- The Transformer: [Attention is All You Need](https://arxiv.org/abs/1706.03762) for the Transformer
- Inception and GoogLeNet: [Going Deeper with Convolutions](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf) for parallel, stacked CNNs
- AlexNet: [ImageNet Classification with Deep Convolutional
](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) for increasing the complexity of feature maps with deeper CNN networks, as well as data augmentation by adding modified versions of the training data to itself
- VGGNet: [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) for using fixed size kernels throughout stacked CNN layers
- LeNet: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) for the convolution>pool>convolution>pool paradigm
- Self-Attention: [Long Short-Term Memory-Networks for Machine Reading](https://arxiv.org/pdf/1601.06733.pdf) for understanding Transformer architecture
- Dropout regularization: [Dropout: A Simple Way to Prevent Neural Networks from
Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) speaks for itself
- Batch Norm: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) speaks for itself

Let's get to it.

#### Setup
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, glob
import librosa
import librosa.display
import IPython
from IPython.display import Audio
from IPython.display import Image
import warnings; warnings.filterwarnings('ignore') #matplot lib complains about librosa

prepare_data = False
agwn_augment = False
train_model = True
data_fname = './data/features+labels.npy'
data_fname = './data/spkr-dep/features+labels.npy'
# RAVDESS native sample rate is 48k
sample_rate = 48000

# Mel Spectrograms are not directly used as a feature in this model
# Mel Spectrograms are used in calculating MFCCs, which are a higher-level representation of pitch transition
# MFCCs work better - left the mel spectrogram function here in case anyone wants to experiment
def feature_melspectrogram(
    waveform, 
    sample_rate,
    fft = 1024,
    winlen = 512,
    window='hamming',
    hop=256,
    mels=128,
    ):
    
    # Produce the mel spectrogram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # Using 8khz as upper frequency bound should be enough for most speech classification tasks
    melspectrogram = librosa.feature.melspectrogram(
        y=waveform, 
        sr=sample_rate, 
        n_fft=fft, 
        win_length=winlen, 
        window=window, 
        hop_length=hop, 
        n_mels=mels, 
        fmax=sample_rate/2)
    
    # convert from power (amplitude**2) to decibels
    # necessary for network to learn - doesn't converge with raw power spectrograms 
    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
    
    return melspectrogram

def feature_mfcc(
    waveform, 
    sample_rate,
    n_mfcc = 40,
    fft = 1024,
    winlen = 512,
    window='hamming',
    #hop=256, # increases # of time steps; was not helpful
    mels=128
    ):

    # Compute the MFCCs for all STFT frames 
    # 40 mel filterbanks (n_mfcc) = 40 coefficients
    mfc_coefficients=librosa.feature.mfcc(
        y=waveform, 
        sr=sample_rate, 
        n_mfcc=n_mfcc,
        n_fft=fft, 
        win_length=winlen, 
        window=window, 
        #hop_length=hop, 
        n_mels=mels, 
        fmax=sample_rate/2
        ) 

    return mfc_coefficients

def get_features(waveforms, features, samplerate):

    # initialize counter to track progress
    file_count = 0

    # process each waveform individually to get its MFCCs
    for waveform in waveforms:
        mfccs = feature_mfcc(waveform, sample_rate)
        features.append(mfccs)
        file_count += 1
        # print progress 
        print('\r'+f' Processed {file_count}/{len(waveforms)} waveforms',end='')
    
    # return all features from list of waveforms
    return features

def get_waveforms(file):
    
    # load an individual sample audio file
    # read the full 3 seconds of the file, cut off the first 0.5s of silence; native sample rate = 48k
    # don't need to store the sample rate that librosa.load returns
    waveform, _ = librosa.load(file, duration=3, offset=0.5, sr=sample_rate)
    
    # make sure waveform vectors are homogenous by defining explicitly
    waveform_homo = np.zeros((int(sample_rate*3,)))
    waveform_homo[:len(waveform)] = waveform
    
    # return a single file's waveform                                      
    return waveform_homo
    
# RAVDESS dataset emotions
# shift emotions left to be 0 indexed for PyTorch
emotions_dict ={
    '0':'surprised',
    '1':'neutral',
    '2':'calm',
    '3':'happy',
    '4':'sad',
    '5':'angry',
    '6':'fearful',
    '7':'disgust'
}

# Additional attributes from RAVDESS to play with
emotion_attributes = {
    '01': 'normal',
    '02': 'strong'
}

# path to data for glob
data_path = 'RAVDESS dataset/Actor_*/*.wav'
data_path = '/scratch/work/huangg5/ravdess_ser/data/audio_speech/Actor_*/*.wav'

def load_data():
    # features and labels
    emotions = []
    # raw waveforms to augment later
    waveforms = []
    # extra labels
    intensities, genders = [],[]
    # progress counter
    file_count = 0
    for file in glob.glob(data_path):
        # get file name with labels
        file_name = os.path.basename(file)
        
        # get emotion label from the sample's file
        emotion = int(file_name.split("-")[2])

        #  move surprise to 0 for cleaner behaviour with PyTorch/0-indexing
        if emotion == 8: emotion = 0 # surprise is now at 0 index; other emotion indeces unchanged

        # can convert emotion label to emotion string if desired, but
        # training on number is better; better convert to emotion string after predictions are ready
        # emotion = emotions_dict[str(emotion)]
        
        # get other labels we might want
        intensity = emotion_attributes[file_name.split("-")[3]]
        # even actors are female, odd are male
        if (int((file_name.split("-")[6]).split(".")[0]))%2==0: 
            gender = 'female' 
        else: 
            gender = 'male'
            
        # get waveform from the sample
        waveform = get_waveforms(file)
        
        # store waveforms and labels
        waveforms.append(waveform)
        emotions.append(emotion)
        intensities.append(intensity) # store intensity in case we wish to predict
        genders.append(gender) # store gender in case we wish to predict 
        
        file_count += 1
        # keep track of data loader's progress
        print('\r'+f' Processed {file_count}/{1440} audio samples',end='')
        
    return waveforms, emotions, intensities, genders

if prepare_data:
    # load data 
    # init explicitly to prevent data leakage from past sessions, since load_data() appends
    waveforms, emotions, intensities, genders = [],[],[],[]
    waveforms, emotions, intensities, genders = load_data()

    print(f'Waveforms set: {len(waveforms)} samples')
    # we have 1440 waveforms but we need to know their length too; should be 3 sec * 48k = 144k
    print(f'Waveform signal length: {len(waveforms[0])}')
    print(f'Emotions set: {len(emotions)} sample labels')

    # create storage for train, validation, test sets and their indices
    train_set,valid_set,test_set = [],[],[]
    X_train,X_valid,X_test = [],[],[]
    y_train,y_valid,y_test = [],[],[]

    # convert waveforms to array for processing
    waveforms = np.array(waveforms)

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

    # initialize feature arrays
    # We extract MFCC features from waveforms and store in respective 'features' array
    features_train, features_valid, features_test = [],[],[]

    print('Train waveforms:') # get training set features 
    features_train = get_features(X_train, features_train, sample_rate)

    print('\n\nValidation waveforms:') # get validation set features
    features_valid = get_features(X_valid, features_valid, sample_rate)

    print('\n\nTest waveforms:') # get test set features 
    features_test = get_features(X_test, features_test, sample_rate)

    print(f'\n\nFeatures set: {len(features_train)+len(features_test)+len(features_valid)} total, {len(features_train)} train, {len(features_valid)} validation, {len(features_test)} test samples')
    print(f'Features (MFC coefficient matrix) shape: {len(features_train[0])} mel frequency coefficients x {len(features_train[0][1])} time steps')

    def awgn_augmentation(waveform, multiples=2, bits=16, snr_min=15, snr_max=30): 
        
        # get length of waveform (should be 3*48k = 144k)
        wave_len = len(waveform)
        
        # Generate normally distributed (Gaussian) noises
        # one for each waveform and multiple (i.e. wave_len*multiples noises)
        noise = np.random.normal(size=(multiples, wave_len))
        
        # Normalize waveform and noise
        norm_constant = 2.0**(bits-1)
        norm_wave = waveform / norm_constant
        norm_noise = noise / norm_constant
        
        # Compute power of waveform and power of noise
        signal_power = np.sum(norm_wave ** 2) / wave_len
        noise_power = np.sum(norm_noise ** 2, axis=1) / wave_len
        
        # Choose random SNR in decibels in range [15,30]
        snr = np.random.randint(snr_min, snr_max)
        
        # Apply whitening transformation: make the Gaussian noise into Gaussian white noise
        # Compute the covariance matrix used to whiten each noise 
        # actual SNR = signal/noise (power)
        # actual noise power = 10**(-snr/10)
        covariance = np.sqrt((signal_power / noise_power) * 10 ** (- snr / 10))
        # Get covariance matrix with dim: (144000, 2) so we can transform 2 noises: dim (2, 144000)
        covariance = np.ones((wave_len, multiples)) * covariance

        # Since covariance and noise are arrays, * is the haddamard product 
        # Take Haddamard product of covariance and noise to generate white noise
        multiple_augmented_waveforms = waveform + covariance.T * noise
        
        return multiple_augmented_waveforms

    def augment_waveforms(waveforms, features, emotions, multiples):
        # keep track of how many waveforms we've processed so we can add correct emotion label in the same order
        emotion_count = 0
        # keep track of how many augmented samples we've added
        added_count = 0
        # convert emotion array to list for more efficient appending
        emotions = emotions.tolist()

        for waveform in waveforms:

            # Generate 2 augmented multiples of the dataset, i.e. 1440 native + 1440*2 noisy = 4320 samples total
            augmented_waveforms = awgn_augmentation(waveform, multiples=multiples)

            # compute spectrogram for each of 2 augmented waveforms
            for augmented_waveform in augmented_waveforms:

                # Compute MFCCs over augmented waveforms
                augmented_mfcc = feature_mfcc(augmented_waveform, sample_rate=sample_rate)

                # append the augmented spectrogram to the rest of the native data
                features.append(augmented_mfcc)
                emotions.append(emotions[emotion_count])

                # keep track of new augmented samples
                added_count += 1

                # check progress
                print('\r'+f'Processed {emotion_count + 1}/{len(waveforms)} waveforms for {added_count}/{len(waveforms)*multiples} new augmented samples',end='')

            # keep track of the emotion labels to append in order
            emotion_count += 1
        
        return features, emotions

    if agwn_augment:
        """### Compute AWGN-augmented features and add to the rest of the dataset"""

        # specify multiples of our dataset to add as augmented data
        multiples = 2

        print('Train waveforms:') # augment waveforms of training set
        features_train , y_train = augment_waveforms(X_train, features_train, y_train, multiples)

        print('\n\nValidation waveforms:') # augment waveforms of validation set
        features_valid, y_valid = augment_waveforms(X_valid, features_valid, y_valid, multiples)

        # print('\n\nTest waveforms:') # augment waveforms of test set 
        # features_test, y_test = augment_waveforms(X_test, features_test, y_test, multiples)

        # Check new shape of extracted features and data:
        print(f'\n\nNative + Augmented Features set: {len(features_train)+len(features_test)+len(features_valid)} total, {len(features_train)} train, {len(features_valid)} validation, {len(features_test)} test samples')
        print(f'{len(y_train)} training sample labels, {len(y_valid)} validation sample labels, {len(y_test)} test sample labels')
        print(f'Features (MFCC matrix) shape: {len(features_train[0])} mel frequency coefficients x {len(features_train[0][1])} time steps')

    # need to make dummy input channel for CNN input feature tensor
    X_train = np.expand_dims(features_train,1)
    X_valid = np.expand_dims(features_valid, 1)
    X_test = np.expand_dims(features_test,1)

    # convert emotion labels from list back to numpy arrays for PyTorch to work with 
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)

    # confiorm that we have tensor-ready 4D data array
    # should print (batch, channel, width, height) == (4320, 1, 128, 282) when multiples==2
    print(f'Shape of 4D feature array for input tensor: {X_train.shape} train, {X_valid.shape} validation, {X_test.shape} test')
    print(f'Shape of emotion labels: {y_train.shape} train, {y_valid.shape} validation, {y_test.shape} test')

    # free up some RAM - no longer need full feature set or any waveforms 
    del features_train, features_valid, features_test, waveforms

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    #### Scale the training data ####
    # store shape so we can transform it back 
    N,C,H,W = X_train.shape
    # Reshape to 1D because StandardScaler operates on a 1D array
    # tell numpy to infer shape of 1D array with '-1' argument
    X_train = np.reshape(X_train, (N,-1)) 
    X_train = scaler.fit_transform(X_train)
    # Transform back to NxCxHxW 4D tensor format
    X_train = np.reshape(X_train, (N,C,H,W))

    ##### Scale the validation set ####
    N,C,H,W = X_valid.shape
    X_valid = np.reshape(X_valid, (N,-1))
    X_valid = scaler.transform(X_valid)
    X_valid = np.reshape(X_valid, (N,C,H,W))

    #### Scale the test set ####
    N,C,H,W = X_test.shape
    X_test = np.reshape(X_test, (N,-1))
    X_test = scaler.transform(X_test)
    X_test = np.reshape(X_test, (N,C,H,W))

    # check shape of each set again
    print(f'X_train scaled:{X_train.shape}, y_train:{y_train.shape}')
    print(f'X_valid scaled:{X_valid.shape}, y_valid:{y_valid.shape}')
    print(f'X_test scaled:{X_test.shape}, y_test:{y_test.shape}')

    ###### SAVE #########
    # open file in write mode and write data
    with open(data_fname, 'wb') as f:
        np.save(f, X_train)
        np.save(f, X_valid)
        np.save(f, X_test)
        np.save(f, y_train)
        np.save(f, y_valid)
        np.save(f, y_test)

    print(f'Features and labels saved to {data_fname}')

##### LOAD #########
"""# choose load file name 
filename = 'features+labels.npy'
"""
# open file in read mode and read data 
with open(data_fname, 'rb') as f:
    X_train = np.load(f)
    X_valid = np.load(f)
    X_test = np.load(f)
    y_train = np.load(f)
    y_valid = np.load(f)
    y_test = np.load(f)

# Check that we've recovered the right data
print(f'X_train:{X_train.shape}, y_train:{y_train.shape}')
print(f'X_valid:{X_valid.shape}, y_valid:{y_valid.shape}')
print(f'X_test:{X_test.shape}, y_test:{y_test.shape}')

class parallel_all_you_want(nn.Module):
    # Define all layers present in the network
    def __init__(self,num_emotions):
        super().__init__() 
        
        ################ TRANSFORMER BLOCK #############################
        # maxpool the input feature map/tensor to the transformer 
        # a rectangular kernel worked better here for the rectangular input spectrogram feature map/tensor
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1,4], stride=[1,4])
        
        # define single transformer encoder layer
        # self-attention + feedforward network from "Attention is All You Need" paper
        # 4 multi-head self-attention layers each with 64-->512--->64 feedforward network
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=40, # input feature (frequency) dim after maxpooling 128*563 -> 64*140 (freq*time)
            nhead=4, # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            dim_feedforward=512, # 2 linear layers in each encoder block's feedforward network: dim 64-->512--->64
            dropout=0.4, 
            activation='relu' # ReLU: avoid saturation/tame gradient/reduce compute time
        )
        
        # I'm using 4 instead of the 6 identical stacked encoder layrs used in Attention is All You Need paper
        # Complete transformer block contains 4 full transformer encoder layers (each w/ multihead self-attention+feedforward)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)
        
        ############### 1ST PARALLEL 2D CONVOLUTION BLOCK ############
        # 3 sequential conv2D layers: (1,40,282) --> (16, 20, 141) -> (32, 5, 35) -> (64, 1, 8)
        self.conv2Dblock1 = nn.Sequential(
            
            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1, # input volume depth == input channel dim == 1
                out_channels=16, # expand output feature map volume's depth to 16
                kernel_size=3, # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16), # batch normalize the output feature map before activation
            nn.ReLU(), # feature map --> activation map
            nn.MaxPool2d(kernel_size=2, stride=2), #typical maxpool kernel size
            nn.Dropout(p=0.3), #randomly zero 30% of 1st layer's output feature map in training
            
            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3), 
            
            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64, # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        ############### 2ND PARALLEL 2D CONVOLUTION BLOCK ############
        # 3 sequential conv2D layers: (1,40,282) --> (16, 20, 141) -> (32, 5, 35) -> (64, 1, 8)
        self.conv2Dblock2 = nn.Sequential(
            
            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1, # input volume depth == input channel dim == 1
                out_channels=16, # expand output feature map volume's depth to 16
                kernel_size=3, # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16), # batch normalize the output feature map before activation
            nn.ReLU(), # feature map --> activation map
            nn.MaxPool2d(kernel_size=2, stride=2), #typical maxpool kernel size
            nn.Dropout(p=0.3), #randomly zero 30% of 1st layer's output feature map in training
            
            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3), 
            
            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64, # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )

        ################# FINAL LINEAR BLOCK ####################
        # Linear softmax layer to take final concatenated embedding tensor 
        #    from parallel 2D convolutional and transformer blocks, output 8 logits 
        # Each full convolution block outputs (64*1*8) embedding flattened to dim 512 1D array 
        # Full transformer block outputs 40*70 feature map, which we time-avg to dim 40 1D array
        # 512*2+40 == 1064 input features --> 8 output emotions 
        self.fc1_linear = nn.Linear(512*2+40,num_emotions) 
        
        ### Softmax layer for the 8 output logits from final FC linear layer 
        self.softmax_out = nn.Softmax(dim=1) # dim==1 is the freq embedding
        
    # define one complete parallel fwd pass of input feature tensor thru 2*conv+1*transformer blocks
    def forward(self,x):
        
        ############ 1st parallel Conv2D block: 4 Convolutional layers ############################
        # create final feature embedding from 1st convolutional layer 
        # input features pased through 4 sequential 2D convolutional layers
        conv2d_embedding1 = self.conv2Dblock1(x) # x == N/batch * channel * freq * time
        
        # flatten final 64*1*4 feature map from convolutional layers to length 256 1D array 
        # skip the 1st (N/batch) dimension when flattening
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1) 
        
        ############ 2nd parallel Conv2D block: 4 Convolutional layers #############################
        # create final feature embedding from 2nd convolutional layer 
        # input features pased through 4 sequential 2D convolutional layers
        conv2d_embedding2 = self.conv2Dblock2(x) # x == N/batch * channel * freq * time
        
        # flatten final 64*1*4 feature map from convolutional layers to length 256 1D array 
        # skip the 1st (N/batch) dimension when flattening
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1) 
        
         
        ########## 4-encoder-layer Transformer block w/ 64-->512-->64 feedfwd network ##############
        # maxpool input feature map: 1*40*282 w/ 1*4 kernel --> 1*40*70
        x_maxpool = self.transformer_maxpool(x)

        # remove channel dim: 1*40*70 --> 40*70
        x_maxpool_reduced = torch.squeeze(x_maxpool,1)
        
        # convert maxpooled feature map format: batch * freq * time ---> time * batch * freq format
        # because transformer encoder layer requires tensor in format: time * batch * embedding (freq)
        x = x_maxpool_reduced.permute(2,0,1) 
        
        # finally, pass reduced input feature map x into transformer encoder layers
        transformer_output = self.transformer_encoder(x)
        
        # create final feature emedding from transformer layer by taking mean in the time dimension (now the 0th dim)
        # transformer outputs 64*140 (freq embedding*time) feature map, take mean of all columns i.e. take time average
        transformer_embedding = torch.mean(transformer_output, dim=0) # dim 40x70 --> 40
        
        ############# concatenate freq embeddings from convolutional and transformer blocks ######
        # concatenate embedding tensors output by parallel 2*conv and 1*transformer blocks
        complete_embedding = torch.cat([conv2d_embedding1, conv2d_embedding2,transformer_embedding], dim=1)  

        ######### final FC linear layer, need logits for loss #########################
        output_logits = self.fc1_linear(complete_embedding)  
        
        ######### Final Softmax layer: use logits from FC linear, get softmax for prediction ######
        output_softmax = self.softmax_out(output_logits)
        
        # need output logits to compute cross entropy loss, need softmax probabilities to predict class
        return output_logits, output_softmax

from torchsummary import summary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# instantiate model for 8 emotions and move to CPU for summary
model = parallel_all_you_want(len(emotions_dict)).to(device)

# include input feature map dims in call to summary()
summary(model, input_size=(1,40,282))

# define loss function; CrossEntropyLoss() fairly standard for multiclass problems 
def criterion(predictions, targets): 
    return nn.CrossEntropyLoss()(input=predictions, target=targets)

optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)

# define function to create a single step of the training phase
def make_train_step(model, criterion, optimizer):
    
    # define the training step of the training phase
    def train_step(X,Y):
        
        # forward pass
        output_logits, output_softmax = model(X)
        predictions = torch.argmax(output_softmax,dim=1)
        accuracy = torch.sum(Y==predictions)/float(len(Y))
        
        # compute loss on logits because nn.CrossEntropyLoss implements log softmax
        loss = criterion(output_logits, Y) 
        
        # compute gradients for the optimizer to use 
        loss.backward()
        
        # update network parameters based on gradient stored (by calling loss.backward())
        optimizer.step()
        
        # zero out gradients for next pass
        # pytorch accumulates gradients from backwards passes (convenient for RNNs)
        optimizer.zero_grad() 
        
        return loss.item(), accuracy*100
    return train_step

def make_validate_fnc(model,criterion):
    def validate(X,Y):
        
        # don't want to update any network parameters on validation passes: don't need gradient
        # wrap in torch.no_grad to save memory and compute in validation phase: 
        with torch.no_grad(): 
            
            # set model to validation phase i.e. turn off dropout and batchnorm layers 
            model.eval()
      
            # get the model's predictions on the validation set
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax,dim=1)

            # calculate the mean accuracy over the entire validation set
            accuracy = torch.sum(Y==predictions)/float(len(Y))
            
            # compute error from logits (nn.crossentropy implements softmax)
            loss = criterion(output_logits,Y)
            
        return loss.item(), accuracy*100, predictions
    return validate

def make_save_checkpoint(): 
    def save_checkpoint(optimizer, model, epoch, filename):
        checkpoint_dict = {
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint_dict, filename)
    return save_checkpoint

def load_checkpoint(optimizer, model, filename, device='cpu'):
    checkpoint_dict = torch.load(filename, map_location=torch.device(device))
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch

# get training set size to calculate # iterations and minibatch indices
train_size = X_train.shape[0]

# pick minibatch size (of 32... always)
minibatch = 32

# instantiate model and move to GPU for training
model = parallel_all_you_want(len(emotions_dict)).to(device) 
print('Number of trainable params: ',sum(p.numel() for p in model.parameters()) )

# encountered bugs in google colab only, unless I explicitly defined optimizer in this cell...
optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)

# instantiate the checkpoint save function
save_checkpoint = make_save_checkpoint()

# instantiate the training step function 
train_step = make_train_step(model, criterion, optimizer=optimizer)

# instantiate the validation loop function
validate_step = make_validate_fnc(model,criterion)

# instantiate lists to hold scalar performance metrics to plot later
train_losses=[]
valid_losses = []

# create training loop for one complete epoch (entire training set)
def train(optimizer, model, num_epochs, X_train, Y_train, X_valid, Y_valid):

    for epoch in range(num_epochs):
        
        # set model to train phase
        model.train()         
        
        # shuffle entire training set in each epoch to randomize minibatch order
        ind = np.random.permutation(train_size) 
        
        # shuffle the training set for each epoch:
        X_train = X_train[ind,:,:,:] 
        Y_train = Y_train[ind]

        # instantiate scalar values to keep track of progress after each epoch so we can stop training when appropriate 
        epoch_acc = 0 
        epoch_loss = 0
        num_iterations = int(train_size / minibatch)
        
        # create a loop for each minibatch of 32 samples:
        for i in range(num_iterations):
            
            # we have to track and update minibatch position for the current minibatch
            # if we take a random batch position from a set, we almost certainly will skip some of the data in that set
            # track minibatch position based on iteration number:
            batch_start = i * minibatch 
            # ensure we don't go out of the bounds of our training set:
            batch_end = min(batch_start + minibatch, train_size) 
            # ensure we don't have an index error
            actual_batch_size = batch_end-batch_start 
            
            # get training minibatch with all channnels and 2D feature dims
            X = X_train[batch_start:batch_end,:,:,:] 
            # get training minibatch labels 
            Y = Y_train[batch_start:batch_end] 

            # instantiate training tensors
            X_tensor = torch.tensor(X, device=device).float() 
            Y_tensor = torch.tensor(Y, dtype=torch.long,device=device)
            
            # Pass input tensors thru 1 training step (fwd+backwards pass)
            loss, acc = train_step(X_tensor,Y_tensor) 
            
            # aggregate batch accuracy to measure progress of entire epoch
            epoch_acc += acc * actual_batch_size / train_size
            epoch_loss += loss * actual_batch_size / train_size
            
            # keep track of the iteration to see if the model's too slow
            print('\r'+f'Epoch {epoch}: iteration {i}/{num_iterations}',end='')
        
        # create tensors from validation set
        X_valid_tensor = torch.tensor(X_valid,device=device).float()
        Y_valid_tensor = torch.tensor(Y_valid,dtype=torch.long,device=device)
        
        # calculate validation metrics to keep track of progress; don't need predictions now
        valid_loss, valid_acc, _ = validate_step(X_valid_tensor,Y_valid_tensor)
        
        # accumulate scalar performance metrics at each epoch to track and plot later
        train_losses.append(epoch_loss)
        valid_losses.append(valid_loss)
                  
        # Save checkpoint of the model
        checkpoint_filename = f'./models/checkpoints/EPOCH-{epoch}.pkl'
        save_checkpoint(optimizer, model, epoch, checkpoint_filename)
        
        # keep track of each epoch's progress
        print(f'\nEpoch {epoch} --- loss:{epoch_loss:.3f}, Epoch accuracy:{epoch_acc:.2f}%, Validation loss:{valid_loss:.3f}, Validation accuracy:{valid_acc:.2f}%')

if train_model:
    # choose number of epochs higher than reasonable so we can manually stop training 
    num_epochs = 500

    # train it!
    train(optimizer, model, num_epochs, X_train, y_train, X_valid, y_valid)

    plt.title('Loss Curve for Parallel is All You Want Model')
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.plot(train_losses[:],'b')
    plt.plot(valid_losses[:],'r')
    plt.legend(['Training loss','Validation loss'])
    plt.savefig('./results/loss.png')

# pick load folder  
load_folder = './models/checkpoints'  

# pick the epoch to load
epoch = '499'
model_name = f'EPOCH-{epoch}.pkl'

# make full load path
load_path = os.path.join(load_folder, model_name)

## instantiate model and populate with params from binary 
load_checkpoint(optimizer, model, load_path, device=device)

print(f'Loaded model from {load_path}')

# reinitialize validation function with model from chosen checkpoint
validate_step = make_validate_fnc(model,criterion)

# Convert 4D test feature set array to tensor and move to GPU
X_test_tensor = torch.tensor(X_test,device=device).float()
# Convert 4D test label set array to tensor and move to GPU
y_test_tensor = torch.tensor(y_test,dtype=torch.long,device=device)

# Get the model's performance metrics using the validation function we defined
test_loss, test_acc, predicted_emotions = validate_step(X_test_tensor,y_test_tensor)

print(f'Test accuracy is {test_acc:.2f}%')

from sklearn.metrics import confusion_matrix
import seaborn as sn

# because model tested on GPU, move prediction tensor to CPU then convert to array
predicted_emotions = predicted_emotions.cpu().numpy()
# use labels from test set
emotions_groundtruth = y_test

# build confusion matrix and normalized confusion matrix
conf_matrix = confusion_matrix(emotions_groundtruth, predicted_emotions)
conf_matrix_norm = confusion_matrix(emotions_groundtruth, predicted_emotions,normalize='true')

# set labels for matrix axes from emotions
emotion_names = [emotion for emotion in emotions_dict.values()]

# make a confusion matrix with labels using a DataFrame
confmatrix_df = pd.DataFrame(conf_matrix, index=emotion_names, columns=emotion_names)
confmatrix_df_norm = pd.DataFrame(conf_matrix_norm, index=emotion_names, columns=emotion_names)

# plot confusion matrices
plt.figure(figsize=(16,6))
sn.set(font_scale=1.8) # emotion label and title size
plt.subplot(1,2,1)
plt.title('Confusion Matrix')
sn.heatmap(confmatrix_df, annot=True, annot_kws={"size": 18}) #annot_kws is value font
plt.subplot(1,2,2)
plt.title('Normalized Confusion Matrix')
sn.heatmap(confmatrix_df_norm, annot=True, annot_kws={"size": 13}) #annot_kws is value font
plt.savefig('./results/cm.png')