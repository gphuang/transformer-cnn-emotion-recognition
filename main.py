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
from src.model import criterion
from src.model import make_save_checkpoint, make_validate_step, make_train_step

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

def parse_args():

    parser = argparse.ArgumentParser(description='Prepare data.')

    parser.add_argument('--in_file', type=str, default='./features+labels.npy',
                        help=f'Specify the full path to the features+labels file.')
    parser.add_argument('--model_dir', type=str, default='./models/checkpoints',
                        help=f'Specify the output directory.')
    parser.add_argument('--out_dir', type=str, default='./results/',
                        help=f'Specify the output directory.')
    parser.add_argument('--win_len', type=int, default=200,
                        help='Specify the window length for segmentation (default: 200 frames).')
    
    args = parser.parse_args()
    return args
    
def main(args):
    numpy.random.seed(10)
    random.seed(10)

    # Data
    filename = args.in_file
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
    model = parallel_all_you_want(num_emotions=len(emotions_dict)).to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)
    save_checkpoint = make_save_checkpoint()
    train_step = make_train_step(model, criterion, optimizer=optimizer)
    validate_step = make_validate_step(model, criterion)
    summary(model, input_size=input_size) # (1,40,469)
    print('Number of trainable params: ',sum(p.numel() for p in model.parameters()) )

    # train
    train_losses=[]
    valid_losses = []
    num_epochs = 500

    for epoch in range(num_epochs):
        
        # set model to train phase
        model.train()         
        
        # shuffle entire training set in each epoch to randomize minibatch order
        ind = np.random.permutation(train_size) 
        
        # shuffle the training set for each epoch:
        X_train = X_train[ind,:,:,:] 
        Y_train = y_train[ind]

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
        Y_valid_tensor = torch.tensor(y_valid,dtype=torch.long,device=device)
        
        # calculate validation metrics to keep track of progress; don't need predictions now
        valid_loss, valid_acc, _ = validate_step(X_valid_tensor,Y_valid_tensor)
        
        # accumulate scalar performance metrics at each epoch to track and plot later
        train_losses.append(epoch_loss)
        valid_losses.append(valid_loss)
                    
        # Save checkpoint of the model
        checkpoint_filename = os.path.join(args.model_dir, 'EPOCH-{:03d}.pkl'.format(epoch))
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

    # evaluate on test set
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
    plt.title('Confusion Matrix')
    sn.heatmap(confmatrix_df, annot=True, annot_kws={"size": 18}) #annot_kws is value font
    plt.subplot(1,2,2)
    plt.title('Normalized Confusion Matrix')
    sn.heatmap(confmatrix_df_norm, annot=True, annot_kws={"size": 13}) #annot_kws is value font
    plt.savefig(os.path.join(args.out_dir, 'cm.png'))

    plt.show()

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    main(args)

    """Unsurprising results - the model has trouble differentiating between 'neutral' and 'calm', and between 'disgust' and 'angry'. 

If a human were asked to differentiate disgust and anger, getting 49/60 correct and 11/60 wrong wouldn't be too bad. 

Other predictable results include confusion of 'sad' for 'disgust'. It perhaps surprising that 'fearful' is confused for 'happy' as often as it is for 'sad' or 'disgust' - although fear is indeed a multifaceted emotion. 

Based on this, I would compare in finer detail the features of confused emotions and see if there are any differences at all - and how to capture them. For real-world data, it would be much more productive to perform sentiment analysis on the spoken words translated to text, and consider that in our final evaluation.

# Conclusion
---------------
Advances of the last 5 years involving upgrades to the autoencoder scheme have lead to the RNN, the upgraded LSTM-RNN, bidirectional LSTM-RNNs, and eventually to LSTM-RNNs with Attention layers to give profound temporal expressivity to the latent space of sequentially encoded data. The Transformer has built on this by taking advantage of parallelized self-attention layers to provide an almost truly global temporal representation of sequential data. 

Today, carefully thought out architecture building on these blocks leads to reasonable training times and excellent generalizability. We combine the CNN for spatial feature representation and the Transformer for temporal feature representation, and  augment the training data by increasing variation in the training dataset to reduce overfitting.

CNNs are still the standard for encoding representations of spatial data. A CNN's filters' kernel sizes are important to both performance and accuracy, especially considering recent paradigms using smaller maxpool kernels such as those of the 3x3 strided 1 kernels in [VGGNet](https://arxiv.org/pdf/1409.1556.pdf), in contrast to the 11x11 stride 4 kernels as in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

When we added convolutional and transformer layers beyond what was used here, that actually decreased test accuracy. Only as much complexity is warranted as is needed for generalizability. Although CNNs are good for images and transformers for sequential data, recently emerging paradigms (such as in this notebook) show that these networks are perfectly cross-applicable given careful thought. 

CNNs are powerful. Transformers work beautifully. They're even better together. Gone are the days of the LSTM-RNN.

If you got this far, I sincerely appreciate your taking the time to do so. Feel free to drop me a line at ilzenkov@gmail.com with any feedback or questions you may have.

# References 
--------------
- Ba et al, 2016. Layer Normalization. https://arxiv.org/abs/1607.06450
- Bahdanau et al, 2015. https://arxiv.org/pdf/1409.0473.pdf
- Cheng et al, 2016. Long Short-Term Memory-Networks for Machine Reading. https://arxiv.org/pdf/1601.06733.pdf
- He et al, 2015. Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385
- Ioffe, Szegedy, 2015. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. https://arxiv.org/abs/1502.03167
- Krizhevsky et al, 2017. ImageNet Classification with Deep Convolutional Neural Networks. https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
- LeCunn et al, 1998. Gradient-Based Learning Applied to Document Recognition. http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
- Li et al, 2018. Visualizing the Loss Landscape of Neural Nets. https://papers.nips.cc/paper/7875-visualizing-the-loss-landscape-of-neural-nets.pdf
- Masters and Luschi, 2018. Revisiting Small Batch Training for Deep Neural Networks. https://arxiv.org/abs/1804.07612 
- Peng et al, 2017. Large Kernel Matters —— Improve Semantic Segmentation by Global Convolutional Network. https://arxiv.org/pdf/1703.02719.pdf
- Santurkar et al, 2019. How Does Batch Normalization Help Optimization? https://arxiv.org/pdf/1805.11604.pdf
- Simonyan and Zisserman, 2015. Very Deep Convolutional Networks for Large-Scale Image Recognition. https://arxiv.org/pdf/1409.1556.pdf
- Srivastava et al, 2014. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
- Ulyanov et al, 2017. Instance Normalization: The Missing Ingredient for Fast Stylization. https://arxiv.org/pdf/1607.08022.pdf
- Vaswani et al, 2017. Attention Is All You Need. https://arxiv.org/abs/1706.03762
- Wilson et al, 2017. The Marginal Value of Adaptive Gradient Methods in Machine Learning. https://arxiv.org/abs/1705.08292

- Christopher Olah's Blog; Neural Networks, Types, and Functional Programming: http://colah.github.io/posts/2015-09-NN-Types-FP/
- Lilian Weng's blog on Attention Mechanisms: https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
- Stanford Autoencoder tutorial: http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/
- Stanford CNN Tutorial: http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/
- Stanford's CS231n: https://cs231n.github.io/convolutional-networks/
- U of T CSC2515, Optimization: https://www.cs.toronto.edu/~hinton/csc2515/notes/lec6tutorial.pdf
"""