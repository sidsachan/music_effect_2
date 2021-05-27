from pathlib import Path
from data_process import pre_process, split
from dataset import DataFrameDataset
from network import Layer3Net
from learning import train, validate, load_model
from utils import plot_loss, cosine_sim, determine_similar_pairs, extract_activations, remove_neuron_pair, eva_model, get_train_val_eval, plot_all_evals
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import numpy as np

# Training settings
save_path = Path('Saved Models', 'h20_betagamma.pth')
parser = argparse.ArgumentParser(description='MusicEffectTrainingArguments')
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                    help='input batch size for testing (default: 8)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--hidden-units-l1', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                    help='learning rate (default: 0.003)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-path', type=Path, default=save_path,
                    help='path to save the learned model')
parser.add_argument('--do-training', type=bool, default=True,
                    help='path to save the learned model')
parser.add_argument('--do-eval', type=bool, default=True,
                    help='path to save the learned model')
parser.add_argument('--do-pruning', type=bool, default=False,
                    help='path to save the learned model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

'''Load, preprocess, split and make torch datasets'''
file_loc = Path('music-affect_v2','music-affect_v2-eeg-features','music-eeg-features.xlsx')

# loading the processed data
normalized_data_list = pre_process(file_loc, plot_data=False)

# splitting into train, validation and test sets (80,10,10)
aby = 3        # 0 for alpha, 1 for beta, 2 for gamma, 3 for beta-gamma combined
train_df, val_df, test_df = split(normalized_data_list[aby], train_frac=0.8, val_frac=0.1)
num_features = normalized_data_list[aby].shape[1] - 1  # last column is the target class
num_classes = 3

# define the custom datasets
train_dataset = DataFrameDataset(df=train_df)
val_dataset = DataFrameDataset(df=val_df)
test_dataset = DataFrameDataset(df=test_df)

# defining the data-loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)

model = Layer3Net(num_features, args.hidden_units_l1, num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.do_training:
    train_loss, val_loss = train(model, train_loader, val_loader, optimizer, args)
    plot_loss(train_loss, val_loss)
if args.do_eval:
    temp_model, optimizer = load_model(model, optimizer, save_path)
    print(save_path)
    eval_list = get_train_val_eval(temp_model, train_loader, val_loader, args)
    print('\nEvaluation through sk-learn metrics')
    print('Training F1 score: ', eval_list[2][0])
    print('Validation F1 score: ', eval_list[2][1])

if args.do_pruning:
    pass