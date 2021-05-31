from pathlib import Path
from data_process import pre_process, split
from dataset import DataFrameDataset
from network import Layer3Net
from learning import train, validate, load_model
from utils import *
from genetic_algorithm import initialize_pop, select_top, select_probability, select_top_dual, cross_mutate
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import numpy as np

# Training settings
save_path = Path('Saved Models', 'running_model.pth')
best_simple = Path('Saved Models', 'h20_best_alphabetagamma.pth')
best_path = Path('Saved Models', 'h20_best_chromosome_alphabetagamma.pth')
pruning_40thresh_distinct = Path('Saved Models', 'best_prune_distinct.pth')
best_after_pruning_genetic_alg = Path('Saved Models', 'best_prune_gen_alg.pth')
parser = argparse.ArgumentParser(description='MusicAffectTrainingArguments')
parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=8, metavar='N', help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--hidden-units-l1', type=int, default=20, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='learning rate (default: 0.003)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-path', type=Path, default=save_path, help='path to save the learned model')
parser.add_argument('--train_simple', type=bool, default=False, help='train model usual way')
parser.add_argument('--train_chromosome', type=bool, default=True, help='train model given a string of chromosome')
parser.add_argument('--prune-distinct', type=bool, default=True, help='pruning using cosine distinctiveness')
parser.add_argument('--prune-gen-alg', type=bool, default=True, help='pruning using genetic algorithm')
parser.add_argument('--gen-alg-input', type=bool, default=False, help='genetic algorithm input space selection')
parser.add_argument('--eval_paths', type=bool, default=True, help='evaluate models saved in the paths')
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

# training baseline models
if args.train_simple:
    num_trials = 20
    val_acc = []
    val_f1 = []
    train_acc = []
    train_f1 = []
    best_model = Layer3Net(num_features, args.hidden_units_l1, num_classes=num_classes)
    best_val_acc = 0
    for i in range(num_trials):
        model = Layer3Net(num_features, args.hidden_units_l1, num_classes=num_classes)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train_loss, val_loss, _ = train(model, train_loader, val_loader, optimizer, args, verbose=False)
        # uncomment to get the plots of losses
        # plot_loss(train_loss, val_loss)
        temp_model, optimizer = load_model(model, optimizer, save_path)
        print(save_path)
        eval_list = get_train_val_eval(temp_model, train_loader, val_loader, args)
        print(i+1,' th run')
        print('Training accuracy: ', eval_list[1][0])
        print('Validation accuracy: ', eval_list[1][1])
        print('\nEvaluation through sk-learn metrics')
        print('Training F1 score: ', eval_list[2][0])
        print('Validation F1 score: ', eval_list[2][1])
        train_acc.append(eval_list[1][0])
        val_acc.append(eval_list[1][1])
        train_f1.append(eval_list[2][0])
        val_f1.append(eval_list[2][1])
        if (eval_list[1][1] > best_val_acc):
            best_model = temp_model
            best_val_acc = eval_list[1][1]
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': eval_list[1][1],
            }, best_simple)
    print('Average training accuracy', np.mean(np.array(train_acc)))
    print('Average training F1-score', np.mean(np.array(train_f1)))
    print('Average validation accuracy', np.mean(np.array(val_acc)))
    print('Average validation F1-score', np.mean(np.array(val_f1)))

# genetic algorithm for input feature selection
if args.gen_alg_input:
    # hyper parameters for the genetic algorithm
    pop_size = 10
    num_generations = 10
    population = initialize_pop(dna_size=num_features, pop_size=pop_size)
    cross_rate = 0.8
    mutation_rate = 0.05
    len_factor = 1

    for gen in range(num_generations):
        print_verbose = False       # variable for printing during training
        # check if all the elements in a row are zero, make these rows random
        row_idx_zero = np.where(~population.any(axis=1))[0]
        if len(row_idx_zero) > 0:
            dna_len = population.shape[1]
            population[row_idx_zero] = np.random.randint(0, 2, size=dna_len)
        # evaluate the population
        acc_list = []
        for chromosome_num in range(pop_size):
            # all the features selected as represented by the chromosome
            column_list = list(np.argwhere(population[chromosome_num, :] == 1).reshape((-1)))
            num_input_features = len(column_list)
            # last column is target, must be selected
            column_list.append(num_features)
            train_loader, val_loader, _ = get_data_loaders(train_df, val_df, test_df, column_list, args)

            model = Layer3Net(num_input_features, args.hidden_units_l1, num_classes=num_classes)

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            _, _, best_val_acc = train(model, train_loader, val_loader, optimizer, args, verbose=print_verbose)
            acc_list.append(best_val_acc)
        # best validation accuracy, chromosome
        best_idx = acc_list.index(max(acc_list))
        print('Generation', gen+1)
        print('Length of Chromosome', len(np.argwhere(population[best_idx, :] == 1)))
        print('Average Length of the population', np.sum(population)/len(population))
        print('Best validation accuracy', max(acc_list))
        print('Mean Validation accuracy', np.mean(np.array(acc_list)))
        print('Variance in the accuracy', np.var(np.array(acc_list)))
        # acc = np.array(acc_list) - len_factor * np.sum(population, axis=1)
        # acc[np.where(acc < 0)] = 0 + 1e-5
        # fitness_prob = acc/np.sum(acc)
        # print('Probabilistic fitness', fitness_prob)
        print('Corresponding chromosome', population[best_idx, :])
        print('[alpha, beta, gamma] contribution:', [sum(population[best_idx, :25]), sum(population[best_idx, 25:50]), sum(population[best_idx, 50:75])])
        # selecting new population, top 5
        population = select_top_dual(population, acc_list, top_count=3, len_factor=len_factor)
        # select probabilistic
        # population = select_probability(population, acc_list, len_factor=len_factor)
        # crossover and mutate
        population = cross_mutate(population, cross_rate, mutation_rate)

# training model for a given chromosome
if args.train_chromosome:
    chromosome = '[1 0 1 0 0 1 0 0 1 0 0 0 1 1 1 1 1 1 0 0 0 1 0 0 1 1 0 1 1 0 0 0 0 1 1 1 0 1 0 0 1 1 1 0 1 1 0 1 0 0 1 1 0 1 1 1 0 1 1 0 1 1 0 1 0 1 1 1 1 0 1 0 0 0 0]'
    column_list = str_to_column_list(chromosome=chromosome)
    num_input_features = len(column_list) - 1
    train_loader, val_loader, _ = get_data_loaders(train_df, val_df, test_df, column_list, args)
    best_model = Layer3Net(num_input_features, args.hidden_units_l1, num_classes=num_classes)
    best_val_acc = 0
    num_trials = 20
    val_acc = []
    val_f1 = []
    train_acc = []
    train_f1 = []
    for i in range(num_trials):
        model = Layer3Net(num_input_features, args.hidden_units_l1, num_classes=num_classes)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train_loss, val_loss, _ = train(model, train_loader, val_loader, optimizer, args, verbose=False)
        # plot_loss(train_loss, val_loss)
        temp_model, optimizer = load_model(model, optimizer, save_path)
        print(save_path)
        eval_list = get_train_val_eval(temp_model, train_loader, val_loader, args)
        print(i+1, ' th run')
        print('Training accuracy: ', eval_list[1][0])
        print('Validation accuracy: ', eval_list[1][1])
        print('\nEvaluation through sk-learn metrics')
        print('Training F1 score: ', eval_list[2][0])
        print('Validation F1 score: ', eval_list[2][1])
        train_acc.append(eval_list[1][0])
        val_acc.append(eval_list[1][1])
        train_f1.append(eval_list[2][0])
        val_f1.append(eval_list[2][1])
        if(eval_list[1][1]>best_val_acc):
            best_model = temp_model
            best_val_acc = eval_list[1][1]
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': eval_list[1][1],
            }, best_path)
    print('Average training accuracy', np.mean(np.array(train_acc)))
    print('Average training F1-score', np.mean(np.array(train_f1)))
    print('Average validation accuracy', np.mean(np.array(val_acc)))
    print('Average validation F1-score', np.mean(np.array(val_f1)))

# distinctive pruning
if args.prune_distinct:
    # load model and run validation/test set through it
    chromosome = '[1 0 1 0 0 1 0 0 1 0 0 0 1 1 1 1 1 1 0 0 0 1 0 0 1 1 0 1 1 0 0 0 0 1 1 1 0 1 0 0 1 1 1 0 1 1 0 1 0 0 1 1 0 1 1 1 0 1 1 0 1 1 0 1 0 1 1 1 1 0 1 0 0 0 0]'
    column_list = str_to_column_list(chromosome=chromosome)
    num_features = len(column_list)-1
    train_loader, val_loader, _ = get_data_loaders(train_df, val_df, test_df, column_list, args)
    model = Layer3Net(num_features, args.hidden_units_l1, num_classes=num_classes)
    best_model = Layer3Net(num_features, args.hidden_units_l1, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model, optimizer = load_model(model, optimizer, best_path)
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
                temp_model = Layer3Net(num_features, h, num_classes)
                temp_model.load_state_dict(pruned_model.state_dict())
            elif len(comp_pairs) != 0:
                idx = idx + 1
                print('Running pruning: ', idx)
                h = h - 1
                pruned_model = Layer3Net(num_features, h, num_classes)
                pruned_model = remove_neuron_pair(temp_model, pruned_model, comp_pairs[0], complement=True)
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
                num_neurons_pruned.append(idx+1)
                # save the best model after pruning
                if threshold == 40:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_accuracy': eval_list[1][1],
                    }, pruning_40thresh_distinct)
    # plotting the evaluation measures after pruning
    plot_all_evals(train_loss, val_loss, num_neurons_pruned, y_l='Average Loss')
    plot_all_evals(train_acc, val_acc, num_neurons_pruned, y_l='Accuracies')
    plot_all_evals(train_f1, val_f1, num_neurons_pruned, y_l='F1 score')

# genetic algorithm pruning
if args.prune_gen_alg:
    # initializing the base model and data loaders according to the chromosome
    chromosome = '[1 0 1 0 0 1 0 0 1 0 0 0 1 1 1 1 1 1 0 0 0 1 0 0 1 1 0 1 1 0 0 0 0 1 1 1 0 1 0 0 1 1 1 0 1 1 0 1 0 0 1 1 0 1 1 1 0 1 1 0 1 1 0 1 0 1 1 1 1 0 1 0 0 0 0]'
    column_list = str_to_column_list(chromosome=chromosome)
    num_input_features = len(column_list) - 1
    train_loader, val_loader, _ = get_data_loaders(train_df, val_df, test_df, column_list, args)
    base_model = Layer3Net(num_input_features, args.hidden_units_l1, num_classes=num_classes)
    best_model = Layer3Net(num_input_features, args.hidden_units_l1, num_classes=num_classes)
    optimizer = optim.Adam(base_model.parameters(), lr=args.lr)
    # load model and run validation/test set through it
    base_model, _ = load_model(base_model, optimizer, best_path)
    eval_list = get_train_val_eval(base_model, train_loader, val_loader, args)
    print('Evaluation through sk-learn metrics')
    print('Training F1 score: ', eval_list[2][0])
    print('Validation F1 score: ', eval_list[2][1])

    # hyper parameters for the genetic algorithm
    pop_size = 100
    num_generations = 30
    num_hidden_neurons = args.hidden_units_l1
    population = initialize_pop(dna_size=num_hidden_neurons, pop_size=pop_size)
    cross_rate = 0.8
    mutation_rate = 0.1
    len_factor = 4
    # lists for performance parameters for plots
    best_val_acc_per_gen = []
    mean_val_acc_per_gen = []
    mean_train_acc_per_gen = []
    mean_length_per_gen = []
    mean_val_loss_per_gen = []
    mean_train_loss_per_gen = []
    mean_train_f1_per_gen = []
    mean_val_f1_per_gen = []
    # pruning algorithm through genetic algorithm
    for gen in range(num_generations):
        # check if all the elements in a row are zero, make these rows random
        row_idx_zero = np.where(~population.any(axis=1))[0]
        if len(row_idx_zero)>0:
            dna_len = population.shape[1]
            population[row_idx_zero] = np.random.randint(0, 2, size=dna_len)
        # evaluate the population
        val_acc_list = []
        val_loss_list =[]
        train_acc_list = []
        train_loss_list = []
        train_f1_list =[]
        val_f1_list = []
        best_val_acc = 0
        for chromosome in population:
            # new number of hidden neurons for the new model
            new_num_hidden_neurons = np.sum(chromosome)
            # copy parameters to new model
            m_temp = Layer3Net(num_input_features, new_num_hidden_neurons, num_classes=num_classes)
            new_model = get_new_model(base_model, m_temp, chromosome=chromosome)
            # run validation for the new model
            eval_list = get_train_val_eval(new_model, train_loader, val_loader, args)
            # evaluation parameters
            train_loss_list.append(eval_list[0][0])
            val_loss_list.append(eval_list[0][1])
            train_acc_list.append(eval_list[1][0])
            val_acc_list.append(eval_list[1][1])
            train_f1_list.append(eval_list[2][0])
            val_f1_list.append(eval_list[2][1])
            # saving best model after pruning
            if (eval_list[1][1] > best_val_acc):
                best_model = new_model
                best_val_acc = eval_list[1][1]
                torch.save({
                    'model_state_dict': new_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': eval_list[1][1],
                }, best_after_pruning_genetic_alg)
        # best validation accuracy, chromosome
        best_idx = val_acc_list.index(max(val_acc_list))
        m_length = np.sum(population) / len(population)
        # updating the lists for the plots
        best_val_acc_per_gen.append(max(val_acc_list))
        mean_val_acc_per_gen.append(np.mean(np.array(val_acc_list)))
        mean_val_loss_per_gen.append(np.mean(np.array(val_loss_list)))
        mean_length_per_gen.append(m_length)
        mean_train_acc_per_gen.append(np.mean(np.array(train_acc_list)))
        mean_train_loss_per_gen.append(np.mean(np.array(train_loss_list)))
        mean_train_f1_per_gen.append(np.mean(np.array(train_f1_list)))
        mean_val_f1_per_gen.append(np.mean(np.array(val_f1_list)))
        # information about pruning performance
        print('Generation', gen+1)
        print('Length of Chromosome', len(np.argwhere(population[best_idx, :] == 1)))
        print('Average size of hidden layer', m_length)
        print('Best validation accuracy', max(val_acc_list))
        print('Corresponding chromosome', population[best_idx, :])
        # selecting new population, top 25
        population = select_top_dual(population, val_acc_list, top_count=25, len_factor=len_factor)
        # select probabilistic
        # population = select_probability(population, val_acc_list, len_factor=len_factor)
        # crossover and mutate
        population = cross_mutate(population, cross_rate, mutation_rate)
    # plotting the accuracy and mean size
    plot_acc_size(best_acc_list=best_val_acc_per_gen, mean_acc_list=mean_val_acc_per_gen, mean_length=mean_length_per_gen, len_factor=len_factor)
    plot_eval_gen_pruning(mean_train_acc_per_gen, mean_val_acc_per_gen, y_l='Accuracy', len_factor=len_factor)
    plot_eval_gen_pruning(mean_train_loss_per_gen, mean_val_loss_per_gen, y_l='Loss', len_factor=len_factor)
    plot_eval_gen_pruning(mean_train_f1_per_gen, mean_val_f1_per_gen, y_l='F1 score', len_factor=len_factor)

# evaluate models saved in the paths
if args.eval_paths:
    chromosome = '[1 0 1 0 0 1 0 0 1 0 0 0 1 1 1 1 1 1 0 0 0 1 0 0 1 1 0 1 1 0 0 0 0 1 1 1 0 1 0 0 1 1 1 0 1 1 0 1 0 0 1 1 0 1 1 1 0 1 1 0 1 1 0 1 0 1 1 1 1 0 1 0 0 0 0]'
    column_list = str_to_column_list(chromosome=chromosome)
    num_features = len(column_list) - 1
    train_loader, val_loader, test_loader = get_data_loaders(train_df, val_df, test_df, column_list, args)
    model1 = Layer3Net(num_features, args.hidden_units_l1, num_classes=num_classes)
    optimizer = optim.Adam(model1.parameters(), lr=args.lr)

    model, optimizer = load_model(model1, optimizer, best_path)
    eval_list = get_train_val_eval(model, val_loader, test_loader, args)
    print('Baseline Model')
    print('Validation accuracy', eval_list[1][0])
    print('Validation F1 score', eval_list[2][0])
    print('Test accuracy', eval_list[1][1])
    print('Test F1 score', eval_list[2][1])

    model, optimizer = load_model(model1, optimizer, pruning_40thresh_distinct)
    eval_list = get_train_val_eval(model, val_loader, test_loader, args)
    print('Distinct pruned')
    print('Validation accuracy', eval_list[1][0])
    print('Validation F1 score', eval_list[2][0])
    print('Test accuracy', eval_list[1][1])
    print('Test F1 score', eval_list[2][1])

    model1 = Layer3Net(num_features, 7, num_classes=num_classes)
    model, optimizer = load_model(model1, optimizer, best_after_pruning_genetic_alg)
    eval_list = get_train_val_eval(model, val_loader, test_loader, args)
    print('Genetic Algorithm pruned')
    print('Validation accuracy', eval_list[1][0])
    print('Validation F1 score', eval_list[2][0])
    print('Test accuracy', eval_list[1][1])
    print('Test F1 score', eval_list[2][1])
