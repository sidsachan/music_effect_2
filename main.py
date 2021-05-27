from pathlib import Path
from data_process import pre_process, split
from dataset import DataFrameDataset
from network import Layer3Net
from learning import train, validate, load_model
from utils import plot_loss, cosine_sim, determine_similar_pairs, extract_activations, remove_neuron_pair, eva_model, get_train_val_eval, plot_all_evals, strtoarray, getdataloaders
from genetic_algorithm import initialize_pop, select_top, select_probability, select_top_dual, cross_mutate
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import numpy as np

# Training settings
save_path = Path('Saved Models', 'h20_betagamma.pth')
parser = argparse.ArgumentParser(description='MusicEffectTrainingArguments')
parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=8, metavar='N', help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--hidden-units-l1', type=int, default=20, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='learning rate (default: 0.003)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-path', type=Path, default=save_path, help='path to save the learned model')
parser.add_argument('--do-training', type=bool, default=True, help='path to save the learned model')
parser.add_argument('--do-eval', type=bool, default=True, help='path to save the learned model')
parser.add_argument('--prune-distinct', type=bool, default=True, help='pruning using cosine distinctiveness')
parser.add_argument('--prune-gen-alg', type=bool, default=True, help='pruning using genetic algorithm')
parser.add_argument('--gen-alg-input', type=bool, default=False, help='genetic algorithm input space')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

'''Load, preprocess, split and make torch datasets'''
file_loc = Path('music-affect_v2', 'music-affect_v2-eeg-features', 'music-eeg-features.xlsx')

# loading the processed data
normalized_data_list = pre_process(file_loc, plot_data=False)

# splitting into train, validation and test sets (80,10,10)
aby = 4        # 0 for alpha, 1 for beta, 2 for gamma, 3 for beta-gamma combined, 4 for alpha-beta-gamma combined
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

if args.gen_alg_input:
    pop_size = 10
    num_generations = 20
    population = initialize_pop(dna_size=num_features, pop_size=pop_size)
    cross_rate = 0.8
    mutation_rate = 0.05
    len_factor = 0.1

    for gen in range(num_generations):
        print_verbose = False
        # evaluate the population
        acc_list = []
        for chromosome_num in range(pop_size):
            # all the features selected as represented by the chromosome
            column_list = list(np.argwhere(population[chromosome_num, :] == 1).reshape((-1)))
            num_features_temp = len(column_list)
            # last column is target, must be selected
            column_list.append(num_features)
            train_loader, val_loader, _ = getdataloaders(train_df, val_df, test_df, column_list, args)

            model = Layer3Net(num_features_temp, args.hidden_units_l1, num_classes=num_classes)
            print(model)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            _, _, best_val_acc = train(model, train_loader, val_loader, optimizer, args, verbose=print_verbose)
            acc_list.append(best_val_acc)
        # best validation accuracy, chromosome
        best_idx = acc_list.index(max(acc_list))
        print('Generation', gen+1)
        print('Length of Chromosome', len(np.argwhere(population[best_idx, :] == 1)))
        print('Average Length of the population', np.sum(population)/len(population))
        print('Best validation accuracy', max(acc_list))
        print('Corresponding chromosome', population[best_idx, :])
        print('[alpha, beta, gamma] contribution:', [sum(population[best_idx, :25]), sum(population[best_idx, 25:50]), sum(population[best_idx, 50:75])])
        # selecting new population, top 5
        population = select_top_dual(population, acc_list, top_count=3, len_factor=len_factor)
        # crossover and mutate
        population = cross_mutate(population, cross_rate, mutation_rate)


# if args.do_training:
#     train_loss, val_loss = train(model, train_loader, val_loader, optimizer, args, verbose=False)
#     plot_loss(train_loss, val_loss)
# if args.do_eval:
#     temp_model, optimizer = load_model(model, optimizer, save_path)
#     print(save_path)
#     eval_list = get_train_val_eval(temp_model, train_loader, val_loader, args)
#     print('\nEvaluation through sk-learn metrics')
#     print('Training F1 score: ', eval_list[2][0])
#     print('Validation F1 score: ', eval_list[2][1])


# distinctive pruning
if args.prune_distinct:
    # load model and run validation/test set through it
    chromosome = '[0 0 0 0 1 0 1 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 1 1 0 0 1 1 1 0 1 0 1 1 0 0 0 0 1 0 0 0 0 1 1 1 1 0 1 0 1 0 1 1 1 1 0 1 1 0]'
    column_list = strtoarray(chromosome=chromosome)
    num_features = len(column_list)-1
    train_loader, val_loader, _ = getdataloaders(train_df, val_df, test_df, column_list, args)
    model = Layer3Net(num_features, args.hidden_units_l1, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model, optimizer = load_model(model, optimizer, save_path)
    eval_list = get_train_val_eval(model, train_loader, val_loader, args)
    print('\nEvaluation through sk-learn metrics')
    # print('Test acc: ', eval_list[1][1])
    print('Training F1 score: ', eval_list[2][0])
    print('Validation F1 score: ', eval_list[2][1])
    # list to store evaluation measures after pruning
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    train_f1 = []
    val_f1 = []
    # adding the initial values before pruning
    num_neurons_pruned = [0]
    train_loss.append(eval_list[0][0])
    val_loss.append(eval_list[0][1])
    train_acc.append(eval_list[1][0])
    val_acc.append(eval_list[1][1])
    train_f1.append(eval_list[2][0])
    val_f1.append(eval_list[2][1])
    # we run pruning for a range of threshold
    for threshold in range(15, 60, 5):
        temp_model, optimizer = load_model(model, optimizer, save_path)
        prune = True
        h = args.hidden_units_l1
        idx = 0
        while (prune):
            activations = extract_activations(temp_model, train_loader, args)
            activations = activations - torch.mean(activations)  # angles between 0 and 180
            similarity_mat = cosine_sim(activations)
            similar_pairs, comp_pairs = determine_similar_pairs(angle_thresh=threshold, sim_mat=similarity_mat)
            if len(similar_pairs) != 0:
                idx = idx + 1
                print('Running pruning: ', idx)
                h = h - 1
                pruned_model = Layer3Net(num_features, h, num_classes)
                pruned_model = remove_neuron_pair(temp_model, pruned_model, similar_pairs[0])
                # eval_list = get_train_val_eval(pruned_model, train_loader, val_loader, args)
                # print('\nEvaluation through sk-learn metrics')
                # print('Training F1 score: ', eval_list[2][0])
                # print('Validation F1 score: ', eval_list[2][1])
                temp_model = Layer3Net(num_features, h, num_classes)
                temp_model.load_state_dict(pruned_model.state_dict())
            elif len(comp_pairs) != 0:
                idx = idx + 1
                print('Running pruning: ', idx)
                h = h - 1
                pruned_model = Layer3Net(num_features, h, num_classes)
                pruned_model = remove_neuron_pair(temp_model, pruned_model, comp_pairs[0], complement=True)
                # eval_list = get_train_val_eval(pruned_model, train_loader, val_loader, args)
                # print('\nEvaluation through sk-learn metrics')
                # print('Training F1 score: ', eval_list[2][0])
                # print('Validation F1 score: ', eval_list[2][1])
                temp_model = Layer3Net(num_features, h, num_classes)
                temp_model.load_state_dict(pruned_model.state_dict())
            else:
                prune = False
                print('Pruning done!!!')
                print(idx, 'neurons pruned')
                print('Final Evaluation measures:')
                eval_list = get_train_val_eval(temp_model, train_loader, val_loader, args)
                print('\nEvaluation through sk-learn metrics')
                print('Training F1 score: ', eval_list[2][0])
                print('Validation F1 score: ', eval_list[2][1])
                # storing the evaluation measures after pruning
                train_loss.append(eval_list[0][0])
                val_loss.append(eval_list[0][1])
                train_acc.append(eval_list[1][0])
                val_acc.append(eval_list[1][1])
                train_f1.append(eval_list[2][0])
                val_f1.append(eval_list[2][1])
                num_neurons_pruned.append(idx)
    # plotting the evaluation measures after pruning
    plot_all_evals(train_loss, val_loss, num_neurons_pruned, y_l='Average Loss')
    plot_all_evals(train_acc, val_acc, num_neurons_pruned, y_l='Accuracies')
    plot_all_evals(train_f1, val_f1, num_neurons_pruned, y_l='F1 score')


# genetic algorithm pruning
if args.prune_gen_alg:
    pass