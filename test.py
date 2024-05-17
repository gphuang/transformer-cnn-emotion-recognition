import os
import argparse
import numpy, random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

from src.model import parallel_all_you_want, load_checkpoint
from src.model import criterion, make_validate_step
from src.config import emotions_dict

# set device to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device} selected')

def parse_args():

    parser = argparse.ArgumentParser(description='Prepare data.')

    parser.add_argument('--data_dir', type=str, default='./data',
                        help=f'Specify the feature directory.')
    parser.add_argument('--model_dir', type=str, default='./models/checkpoints',
                        help=f'Specify the model directory.')
    parser.add_argument('--out_dir', type=str, default='./results',
                        help=f'Specify the output directory.')
    parser.add_argument('--epoch_num', type=int, default=499,
                        help='Specify the window length for segmentation (default: 200 frames).')
    
    args = parser.parse_args()
    return args

def main(args):

    # pick the epoch to load
    epoch = args.epoch_num
    model_name = f'EPOCH-{epoch}.pkl'

    # make full load path
    load_path = os.path.join(args.model_dir, model_name)

    ## instantiate empty model and populate with params from binary 
    model = parallel_all_you_want(len(emotions_dict)).to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)
    load_checkpoint(optimizer, model, load_path, device=device)

    print(f'Loaded model from {load_path}')

    """# Evaluate the Model on Hold-Out Test Set
    Fingers crossed for generalizability.
    """
    ##### LOAD #########
    # choose load file name 
    filename = os.path.join(args.data_dir, 'features+labels.npy')

    # open file in read mode and read data 
    with open(filename, 'rb') as f:
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

    # reinitialize validation function with model from chosen checkpoint
    validate_step = make_validate_step(model,criterion)

    # Convert 4D test feature set array to tensor and move to GPU
    X_test_tensor = torch.tensor(X_test,device=device).float()
    # Convert 4D test label set array to tensor and move to GPU
    y_test_tensor = torch.tensor(y_test,dtype=torch.long,device=device)

    # Get the model's performance metrics using the validation function we defined
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
    sn.set(font_scale=1.8) # emotion label and title size
    plt.subplot(1,2,1)
    plt.title('Confusion Matrix (Test acc. {test_acc:.2f}%)')
    sn.heatmap(confmatrix_df, annot=True, annot_kws={"size": 18}) #annot_kws is value font
    plt.subplot(1,2,2)
    plt.title('Normalized Confusion Matrix')
    sn.heatmap(confmatrix_df_norm, annot=True, annot_kws={"size": 13}) #annot_kws is value font
    plt.savefig(os.path.join(args.out_dir, 'cm.png'))

if __name__ == '__main__':
    args = parse_args()

    import pathlib
    p = pathlib.Path(args.out_dir)
    p.mkdir(parents=True, exist_ok=True)

    main(args)