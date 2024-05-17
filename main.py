# -*- coding: utf-8 -*-
"""Parallel_is_all_you_want.ipynb
[Intro to Speech Audio Classification repo](https://github.com/IliaZenkov/sklearn-audio-classification)
This notebook builds two parallel convolutional neural networks (CNN) in parallel with a Transformer encoder network to classify 8 emotions on the audio [RAVDESS dataset](https://smartlaboratory.org/ravdess/) 
It combines the CNN for spatial feature representation and the Transformer for temporal feature representation. 
It augments the training data with Additive White Gaussian Noise (AWGN) three-fold for a total of 4320 audio samples.
"""
import os, sys 
import argparse
import numpy, random
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchsummary import summary
from sklearn.metrics import confusion_matrix
import seaborn as sn

from src.config import emotions_dict
from src.model import parallel_all_you_want
from src.model import criterion, load_checkpoint
from src.model import make_save_checkpoint, make_validate_step, make_train_step

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

def parse_args():

    parser = argparse.ArgumentParser(description='Prepare data.')

    parser.add_argument('--data_dir', type=str, default='./data',
                        help=f'Specify the full path to the features+labels file.')
    parser.add_argument('--model_dir', type=str, default='./models/checkpoints',
                        help=f'Specify the output directory.')
    parser.add_argument('--out_dir', type=str, default='./results',
                        help=f'Specify the output directory.')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Specify the window length for segmentation (default: 200 frames).')
    parser.add_argument('--no_train', action='store_true',
                        help='Specify whether to train (default: False).')
    parser.add_argument('--no_test', action='store_true',
                        help='Specify whether to train (default: False).')
    
    args = parser.parse_args()
    return args
 
def main(args):
    numpy.random.seed(10)
    random.seed(10)

    # Data
    filename = os.path.join(args.data_dir, 'features+labels.npy')
    with open(filename, 'rb') as f:
        X_train = np.load(f)
        X_valid = np.load(f)
        X_test = np.load(f)
        y_train = np.load(f)
        y_valid = np.load(f)
        y_test = np.load(f)
    print(f'X_train:{X_train.shape}, y_train:{y_train.shape}')
    print(f'X_valid:{X_valid.shape}, y_valid:{y_valid.shape}')
    print(f'X_test:{X_test.shape}, y_test:{y_test.shape}')

    # model
    minibatch = 32
    input_size = X_train.shape[-3:]
    train_size = X_train.shape[0]
    model = parallel_all_you_want(len(emotions_dict)).to(device)
    summary(model, input_size=input_size) # (1,40,282)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)
    save_checkpoint = make_save_checkpoint()
    train_step = make_train_step(model, criterion, optimizer=optimizer)
    validate_step = make_validate_step(model, criterion)
    print('Number of trainable params: ',sum(p.numel() for p in model.parameters()) )

    # train
    if not args.no_train:
        num_epochs = args.num_epochs
        train_losses=[]
        valid_losses = []
        for epoch in range(num_epochs):
            
            # set model to train phase
            model.train()         
            
            # shuffle entire training set in each epoch to randomize minibatch order
            ind = np.random.permutation(train_size) 
            
            # shuffle the training set for each epoch:
            X_train = X_train[ind,:,:,:] 
            y_train = y_train[ind]

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
                Y = y_train[batch_start:batch_end] 

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
            Y_valid_tensor = torch.tensor(y_valid,dtype=torch.long,device=device)
            
            # calculate validation metrics to keep track of progress; don't need predictions now
            valid_loss, valid_acc, _ = validate_step(X_valid_tensor,Y_valid_tensor)
            
            # accumulate scalar performance metrics at each epoch to track and plot later
            train_losses.append(epoch_loss)
            valid_losses.append(valid_loss)
                    
            # Save checkpoint of the model
            checkpoint_filename = os.path.join(args.model_dir, f'EPOCH-{epoch}.pkl')
            save_checkpoint(optimizer, model, epoch, checkpoint_filename)
            
            # keep track of each epoch's progress
            print(f'\nEpoch {epoch} --- loss:{epoch_loss:.3f}, Epoch accuracy:{epoch_acc:.2f}%, Validation loss:{valid_loss:.3f}, Validation accuracy:{valid_acc:.2f}%')
    
        # Check the Loss Curve's Behaviour
        plt.title('Loss Curve for Parallel is All You Want Model')
        plt.ylabel('Loss', fontsize=16)
        plt.xlabel('Epoch', fontsize=16)
        plt.plot(train_losses[:],'b')
        plt.plot(valid_losses[:],'r')
        plt.legend(['Training loss','Validation loss'])
        plt.savefig(os.path.join(args.out_dir,'loss.png'))

    # test
    if not args.no_test:
        epoch = '499'
        model_name = f'EPOCH-{epoch}.pkl'
        load_path = os.path.join(args.model_dir, model_name)

        ## instantiate empty model and populate with params from binary 
        load_checkpoint(optimizer, model, load_path, device=device)

        print(f'Loaded model from {load_path}')

        # reinitialize validation function with model from chosen checkpoint
        validate_step = make_validate_step(model,criterion)
        X_test_tensor = torch.tensor(X_test,device=device).float()
        y_test_tensor = torch.tensor(y_test,dtype=torch.long,device=device)
        test_loss, test_acc, predicted_emotions = validate_step(X_test_tensor,y_test_tensor)

        print(f'Test accuracy is {test_acc:.2f}%')

        # analyze test results
        predicted_emotions = predicted_emotions.cpu().numpy()
        emotions_groundtruth = y_test
        conf_matrix = confusion_matrix(emotions_groundtruth, predicted_emotions)
        conf_matrix_norm = confusion_matrix(emotions_groundtruth, predicted_emotions,normalize='true')
        emotion_names = [emotion for emotion in emotions_dict.values()]
        confmatrix_df = pd.DataFrame(conf_matrix, index=emotion_names, columns=emotion_names)
        confmatrix_df_norm = pd.DataFrame(conf_matrix_norm, index=emotion_names, columns=emotion_names)

        # plot confusion matrices
        plt.figure(figsize=(16,6))
        sn.set_theme(font_scale=1.8) # emotion label and title size
        plt.subplot(1,2,1)
        plt.title(f'Confusion Matrix (Test acc. {test_acc:.2f}%)')
        sn.heatmap(confmatrix_df, annot=True, annot_kws={"size": 18}) #annot_kws is value font
        plt.subplot(1,2,2)
        plt.title('Normalized Confusion Matrix')
        sn.heatmap(confmatrix_df_norm, annot=True, annot_kws={"size": 13}) #annot_kws is value font
        plt.savefig(os.path.join(args.out_dir, 'cm.png'))

        plt.show()

if __name__ == '__main__':
    args = parse_args()

    import pathlib
    p = pathlib.Path(args.model_dir)
    p.mkdir(parents=True, exist_ok=True)

    p = pathlib.Path(args.out_dir)
    p.mkdir(parents=True, exist_ok=True)

    main(args)