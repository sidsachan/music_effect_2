from pathlib import Path
from data_process import pre_process, split, split_participant_wise
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, as_completed, ProcessPoolExecutor
from concurrent import futures

# Training settings
save_path_t1 = Path('Saved Models', 'running_model_1.pth')
save_path_t2 = Path('Saved Models', 'running_model_2.pth')
save_path_t3 = Path('Saved Models', 'running_model_3.pth')
save_path_t4 = Path('Saved Models', 'running_model_4.pth')

best_simple = Path('Saved Models', 'h20_best_alphabetagamma.pth')
best_path = Path('Saved Models', 'h20_best_chromosome_alphabetagamma.pth')
pruning_40thresh_distinct = Path('Saved Models', 'best_prune_distinct.pth')
best_after_pruning_genetic_alg = Path('Saved Models', 'best_prune_gen_alg.pth')

parser = argparse.ArgumentParser(description='MusicAffectTrainingArguments')
parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=8, metavar='N', help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
parser.add_argument('--hidden-units-l1', type=int, default=20, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='learning rate (default: 0.003)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-path', type=Path, default=save_path_t1, help='path to save the learned model')
parser.add_argument('--train_simple', type=bool, default=False, help='train model usual way')
parser.add_argument('--train_chromosome', type=bool, default=False, help='train model given a string of chromosome')
parser.add_argument('--prune-distinct', type=bool, default=False, help='pruning using cosine distinctiveness')
parser.add_argument('--prune-gen-alg', type=bool, default=True, help='pruning using genetic algorithm')
parser.add_argument('--gen-alg-input', type=bool, default=False, help='genetic algorithm input space selection')
parser.add_argument('--eval_paths', type=bool, default=False, help='evaluate models saved in the paths')
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



def split_list(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def simple_train_thread(train_df_list, val_df_list, args):
    val_acc = []
    val_f1 = []
    train_acc = []
    train_f1 = []
    for i in range(len(train_df_list)):
        # define the custom datasets
        train_dataset = DataFrameDataset(df=train_df_list[i])
        val_dataset = DataFrameDataset(df=val_df_list[i])

        # defining the data-loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size)

        model = Layer3Net(num_features, args.hidden_units_l1, num_classes=num_classes)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train_loss, val_loss, _ = train(model, train_loader, val_loader, optimizer, args, verbose=False)
        temp_model, optimizer = load_model(model, optimizer, save_path_t1)
        eval_list = get_train_val_eval(temp_model, train_loader, val_loader, args)
        train_acc.append(eval_list[1][0])
        val_acc.append(eval_list[1][1])
        train_f1.append(eval_list[2][0])
        val_f1.append(eval_list[2][1])

    return [train_acc, train_f1, val_acc, val_f1]


def train_chromosome(df, chromosome, args, num_classes=3):
    column_list = str_to_column_list(chromosome=chromosome)
    num_input_features = len(column_list) - 1
    train_df_list, val_df_list = split_participant_wise(df, k=24)

    val_acc = []
    val_f1 = []
    train_acc = []
    train_f1 = []
    model_list = []
    for i in range(len(train_df_list)):
        train_loader, val_loader, _ = get_data_loaders(train_df_list[i], val_df_list[i], test_df, column_list, args)
        best_model = Layer3Net(num_input_features, args.hidden_units_l1, num_classes=num_classes)
        best_val_acc = 0
        for j in range(5):
            model = Layer3Net(num_input_features, args.hidden_units_l1, num_classes=num_classes)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            train_loss, val_loss, val_accy = train(model, train_loader, val_loader, optimizer, args, verbose=False)
            if val_accy > best_val_acc:
                best_model,_ = load_model(model, optimizer, save_path_t1)
                best_val_acc = val_accy
        eval_list = get_train_val_eval(best_model, train_loader, val_loader, args)
        print('\nParticipant ', i + 1, ' for validation')
        print('Training accuracy: ', eval_list[1][0])
        print('Validation accuracy: ', eval_list[1][1])
        # print('\nEvaluation through sk-learn metrics')
        print('Training F1 score: ', eval_list[2][0])
        print('Validation F1 score: ', eval_list[2][1])
        train_acc.append(eval_list[1][0])
        val_acc.append(eval_list[1][1])
        train_f1.append(eval_list[2][0])
        val_f1.append(eval_list[2][1])
        model_list.append(best_model)
    print('Average training accuracy', np.mean(np.array(train_acc)))
    print('Average training F1-score', np.mean(np.array(train_f1)))
    print('Average validation accuracy', np.mean(np.array(val_acc)))
    print('Average validation F1-score', np.mean(np.array(val_f1)))
    return model_list


# training baseline models
if args.train_simple:
    train_df_list, val_df_list = split_participant_wise(normalized_data_list[aby])
    num_threads = 4
    train_list_list = split_list(train_df_list, num_threads)
    val_list_list = split_list(val_df_list, num_threads)
    val_acc = []
    val_f1 = []
    train_acc = []
    train_f1 = []
    best_model = Layer3Net(num_features, args.hidden_units_l1, num_classes=num_classes)
    best_val_acc = 0

    # args2 = argparse.Namespace(**vars(args))
    # args3 = argparse.Namespace(**vars(args))
    # args4 = argparse.Namespace(**vars(args))
    # args2.save_path = save_path_t2
    # args3.save_path = save_path_t3
    # args4.save_path = save_path_t4
    # arg_list = [args, args2, args3, args4]
    #
    # pool = ProcessPoolExecutor(num_threads)
    # # futures = []
    # start = time.time()
    # with ThreadPoolExecutor(num_threads) as executor:
    #     # Use list jobs for concurrent futures
    #     # Use list scraped_results for results
    #     jobs = []
    #     results_done = []
    #     # Here you identify how many parallel tasks you want
    #     # and what value you'll send to them
    #     for x in range(num_threads):
    #         # Pass some keyword arguments if needed - per job
    #
    #         # Here we iterate 'number of dates' times, could be different
    #         # We're adding scrape_func, could be different function per call
    #         jobs.append(executor.submit(simple_train_thread, train_list_list[x], val_list_list[x], arg_list[x]))
    #
    #     # Once parallel processing is complete, iterate over results
    #     for job in as_completed(jobs):
    #         # Read result from future
    #         result_done = job.result()
    #         # Append to the list of results
    #         results_done.append(result_done)
    #
    #     # Iterate over results scraped and do whatever is needed
    #     for result in results_done:
    #         print((result))

    # for x in range(num_threads):
    #     futures.append(pool.submit(simple_train_thread, (train_list_list[x], val_list_list[x], arg_list[x])))
    # for x in as_completed(futures):
    #     print(x.result())

    # end = time.time()
    # print(end - start)

    # th1 = threading.Thread(target=simple_train_thread, args=(train_list_list[0], val_list_list[0], args))
    # th1.start()
    # args.save_path = save_path_t2
    # th2 = threading.Thread(target=simple_train_thread, args=(train_list_list[1], val_list_list[1], args))
    # th2.start()
    start = time.time()
    for i in range(len(train_df_list)):
        # define the custom datasets
        train_dataset = DataFrameDataset(df=train_df_list[i])
        val_dataset = DataFrameDataset(df=val_df_list[i])

        # defining the data-loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size)

        model = Layer3Net(num_features, args.hidden_units_l1, num_classes=num_classes)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train_loss, val_loss, _ = train(model, train_loader, val_loader, optimizer, args, verbose=False)
        # uncomment to get the plots of losses
        # plot_loss(train_loss, val_loss)
        temp_model, optimizer = load_model(model, optimizer, save_path_t1)
        eval_list = get_train_val_eval(temp_model, train_loader, val_loader, args)
        print('Participant ', i+1, ' for validation')
        print('Training accuracy: ', eval_list[1][0])
        print('Validation accuracy: ', eval_list[1][1])
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
    end = time.time()
    print(end - start)
    print('Average training accuracy', np.mean(np.array(train_acc)))
    print('Average training F1-score', np.mean(np.array(train_f1)))
    print('Average validation accuracy', np.mean(np.array(val_acc)))
    print('Average validation F1-score', np.mean(np.array(val_f1)))

# genetic algorithm for input feature selection
if args.gen_alg_input:
    # hyper parameters for the genetic algorithm
    pop_size = 20
    num_generations = 10
    population = initialize_pop(dna_size=num_features, pop_size=pop_size)
    cross_rate = 0.8
    mutation_rate = [0.05, 0.1, 0.2]
    len_factor = 1
    train_df_list, val_df_list = split_participant_wise(normalized_data_list[aby], k=6)

    for mr in mutation_rate:
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
                k_fold_acc_list = []
                for j in range(len(train_df_list)):
                    train_loader, val_loader, _ = get_data_loaders(train_df_list[j], val_df_list[j], test_df, column_list, args)

                    model = Layer3Net(num_input_features, args.hidden_units_l1, num_classes=num_classes)

                    optimizer = optim.Adam(model.parameters(), lr=args.lr)
                    _, _, best_val_acc = train(model, train_loader, val_loader, optimizer, args, verbose=print_verbose)
                    k_fold_acc_list.append(best_val_acc)
                # the mean accuracy after k_fold_cross validation represents the fitness of the chromosome
                acc_list.append(np.mean(np.array(k_fold_acc_list)))
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
            population = select_top_dual(population, acc_list, top_count=5, len_factor=len_factor)
            # select probabilistic
            # population = select_probability(population, acc_list, len_factor=lf)
            # crossover and mutate
            population = cross_mutate(population, cross_rate, mr)

# training model for a given chromosome
if args.train_chromosome:
    chromosome = '[1 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 1 1 0]'
    model_list = train_chromosome(normalized_data_list[aby], chromosome, args)
    for i, m in enumerate(model_list):
        torch.save(m.state_dict(), f"Saved Models/baseline_model{i}.pth")
# distinctive pruning
if args.prune_distinct:
    # load model and run validation/test set through it
    chromosome = '[1 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 1 1 0]'
    train_df_list, val_df_list = split_participant_wise(normalized_data_list[aby], k=24)
    column_list = str_to_column_list(chromosome=chromosome)
    num_input_features = len(column_list) - 1
    baseline_model_list = [Layer3Net(num_input_features, args.hidden_units_l1, num_classes) for i in range(24)]  # Note that you should adapt this to whatever method you use to create your models in the first place
    for i, m in enumerate(baseline_model_list):
        m.load_state_dict(torch.load(f"Saved Models/baseline_model{i}.pth"))
    # list to store evaluation measures after pruning
    train_loss = [[] for _ in range(24)]
    val_loss = [[] for _ in range(24)]
    train_acc = [[] for _ in range(24)]
    val_acc = [[] for _ in range(24)]
    train_f1 = [[] for _ in range(24)]
    val_f1 = [[] for _ in range(24)]
    num_neurons_pruned = [[] for _ in range(24)]

    for i in range(len(train_df_list)):
        print('Model ', i)
        train_loader, val_loader, _ = get_data_loaders(train_df_list[i], val_df_list[i], test_df, column_list, args)

        base_model = baseline_model_list[i]
        eval_list = get_train_val_eval(base_model, train_loader, val_loader, args)
        print('\nEvaluation through sk-learn metrics')
        # print('Test acc: ', eval_list[1][1])
        print('Training F1 score: ', eval_list[2][0])
        print('Validation F1 score: ', eval_list[2][1])

        # adding the initial values before pruning
        num_neurons_pruned[i].append(0)
        train_loss[i].append(eval_list[0][0])
        val_loss[i].append(eval_list[0][1])
        train_acc[i].append(eval_list[1][0])
        val_acc[i].append(eval_list[1][1])
        train_f1[i].append(eval_list[2][0])
        val_f1[i].append(eval_list[2][1])
        # we run pruning for a range of threshold
        for threshold in range(5, 60, 5):
            temp_model = base_model
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
                    # print('Running pruning: ', idx)
                    h = h - 1
                    pruned_model = Layer3Net(num_input_features, h, num_classes)
                    pruned_model = remove_neuron_pair(temp_model, pruned_model, similar_pairs[0])
                    temp_model = Layer3Net(num_input_features, h, num_classes)
                    temp_model.load_state_dict(pruned_model.state_dict())
                elif len(comp_pairs) != 0:
                    idx = idx + 1
                    # print('Running pruning: ', idx)
                    h = h - 1
                    pruned_model = Layer3Net(num_input_features, h, num_classes)
                    pruned_model = remove_neuron_pair(temp_model, pruned_model, comp_pairs[0], complement=True)
                    temp_model = Layer3Net(num_input_features, h, num_classes)
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
                    train_loss[i].append(eval_list[0][0])
                    val_loss[i].append(eval_list[0][1])
                    train_acc[i].append(eval_list[1][0])
                    val_acc[i].append(eval_list[1][1])
                    train_f1[i].append(eval_list[2][0])
                    val_f1[i].append(eval_list[2][1])
                    num_neurons_pruned[i].append(idx + 1)
                    # save the best model after pruning
                    # if threshold == 40:
                    #     torch.save({
                    #         'model_state_dict': model.state_dict(),
                    #         'optimizer_state_dict': optimizer.state_dict(),
                    #         'val_accuracy': eval_list[1][1],
                    #     }, pruning_40thresh_distinct)
    # plotting the evaluation measures after pruning
    train_loss_avg = np.round(np.mean(np.array(train_loss), axis=0), 2)
    val_loss_avg = np.round(np.mean(np.array(val_loss), axis=0), 2)
    train_acc_avg = np.round(np.mean(np.array(train_acc), axis=0), 2)
    val_acc_avg = np.round(np.mean(np.array(val_acc), axis=0), 2)
    train_f1_avg = np.round(np.mean(np.array(train_f1), axis=0), 2)
    val_f1_avg = np.round(np.mean(np.array(val_f1), axis=0), 2)
    num_neurons_pruned_avg = np.round(np.mean(np.array(num_neurons_pruned), axis=0), 2)

    plot_all_evals(train_loss_avg, val_loss_avg, num_neurons_pruned_avg, y_l='Average Loss')
    plot_all_evals(train_acc_avg, val_acc_avg, num_neurons_pruned_avg, y_l='Accuracies')
    plot_all_evals(train_f1_avg, val_f1_avg, num_neurons_pruned_avg, y_l='F1 score')

# genetic algorithm pruning
def gen_alg_pruning(train_df_list, val_df_list, baseline_model_list, num_input_features, cross_rate, mutation_rate, len_factor, pop_size = 50,num_generations = 10,num_hidden_neurons = args.hidden_units_l1):
    # lists for performance parameters for plots
    length_best_acc = [[] for _ in range(24)]
    best_val_acc_per_model = [[] for _ in range(24)]
    best_train_acc_per_model = [[] for _ in range(24)]
    mean_val_acc_per_model = [[] for _ in range(24)]
    mean_train_acc_per_model = [[] for _ in range(24)]
    mean_length_per_model = [[] for _ in range(24)]
    mean_val_loss_per_model = [[] for _ in range(24)]
    mean_train_loss_per_model = [[] for _ in range(24)]
    mean_train_f1_per_model = [[] for _ in range(24)]
    mean_val_f1_per_model = [[] for _ in range(24)]
    for i in range(len(train_df_list)):
        population = initialize_pop(dna_size=num_hidden_neurons, pop_size=pop_size)
        print('Model ', i)
        train_loader, val_loader, _ = get_data_loaders(train_df_list[i], val_df_list[i], test_df, column_list, args)

        base_model = baseline_model_list[i]
        best_model = Layer3Net(num_input_features, args.hidden_units_l1, num_classes=num_classes)
        optimizer = optim.Adam(base_model.parameters(), lr=args.lr)
        # load model and run validation/test set through it
        # base_model, _ = load_model(base_model, optimizer, best_path)
        eval_list = get_train_val_eval(base_model, train_loader, val_loader, args)

        best_train_acc_per_model[i].append(eval_list[1][0])
        best_val_acc_per_model[i].append(eval_list[1][1])
        mean_val_acc_per_model[i].append(eval_list[1][1])
        mean_val_loss_per_model[i].append(eval_list[0][1])
        mean_length_per_model[i].append(20)
        mean_train_acc_per_model[i].append(eval_list[1][0])
        mean_train_loss_per_model[i].append(eval_list[0][0])
        mean_train_f1_per_model[i].append(eval_list[2][0])
        mean_val_f1_per_model[i].append(eval_list[2][1])

        # print('Evaluation through sk-learn metrics')
        # print('Training F1 score: ', eval_list[2][0])
        # print('Validation F1 score: ', eval_list[2][1])

        # pruning algorithm through genetic algorithm
        for gen in range(num_generations):
            # check if all the elements in a row are zero, make these rows random
            row_idx_zero = np.where(~population.any(axis=1))[0]
            if len(row_idx_zero) > 0:
                dna_len = population.shape[1]
                population[row_idx_zero] = np.random.randint(0, 2, size=dna_len)
            # evaluate the population
            val_acc_list = []
            val_loss_list = []
            train_acc_list = []
            train_loss_list = []
            train_f1_list = []
            val_f1_list = []
            # best_val_acc = 0
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
                # if (eval_list[1][1] > best_val_acc):
                #     best_model = new_model
                #     best_val_acc = eval_list[1][1]
                #     torch.save({
                #         'model_state_dict': new_model.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         'val_accuracy': eval_list[1][1],
                #     }, best_after_pruning_genetic_alg)
            # best validation accuracy, chromosome
            best_idx = train_acc_list.index(max(train_acc_list))
            m_length = np.sum(population) / len(population)
            # updating the lists for the plots
            length_best_acc[i].append(np.sum(population[best_idx, :]))
            best_train_acc_per_model[i].append(max(train_acc_list))
            best_val_acc_per_model[i].append((val_acc_list[best_idx]))

            mean_val_acc_per_model[i].append(np.mean(np.array(val_acc_list)))
            mean_val_loss_per_model[i].append(val_loss_list[best_idx])
            mean_length_per_model[i].append(m_length)
            mean_train_acc_per_model[i].append(np.mean(np.array(train_acc_list)))
            mean_train_loss_per_model[i].append(train_loss_list[best_idx])
            mean_train_f1_per_model[i].append(train_f1_list[best_idx])
            mean_val_f1_per_model[i].append(val_f1_list[best_idx])

            # mean_val_loss_per_model[i].append(np.mean(np.array(val_loss_list)))
            # mean_train_loss_per_model[i].append(np.mean(np.array(train_loss_list)))
            # mean_train_f1_per_model[i].append(np.mean(np.array(train_f1_list)))
            # mean_val_f1_per_model[i].append(np.mean(np.array(val_f1_list)))

            # information about pruning performance
            # print('Generation', gen + 1)
            # print('Length of Chromosome', len(np.argwhere(population[best_idx, :] == 1)))
            # print('Average size of hidden layer', m_length)
            # print('Best training Accuracy', max(train_acc_list))
            # print('Corresponding chromosome', population[best_idx, :])
            # selecting new population, top 25
            population = select_top_dual(population, train_acc_list, top_count=10, len_factor=len_factor)
            # select probabilistic
            # population = select_probability(population, val_acc_list, len_factor=len_factor)
            # crossover and mutate
            population = cross_mutate(population, cross_rate, mutation_rate)

    # plotting the accuracy and mean size
    best_train_acc_per_gen = np.mean(np.array(best_train_acc_per_model), axis=0)
    best_val_acc_per_gen = np.mean(np.array(best_val_acc_per_model), axis=0)
    mean_val_acc_per_gen = np.mean(np.array(mean_val_acc_per_model), axis=0)
    mean_val_loss_per_gen = np.mean(np.array(mean_val_loss_per_model), axis=0)
    mean_length_per_gen = np.mean(np.array(mean_length_per_model), axis=0)
    mean_train_acc_per_gen = np.mean(np.array(mean_train_acc_per_model), axis=0)
    mean_train_loss_per_gen = np.mean(np.array(mean_train_loss_per_model), axis=0)
    mean_train_f1_per_gen = np.mean(np.array(mean_train_f1_per_model), axis=0)
    mean_val_f1_per_gen = np.mean(np.array(mean_val_f1_per_model), axis=0)

    idx_best_train = list(np.argmax(np.array(best_train_acc_per_model)[:, 1:], axis=1))
    len_best = np.array(length_best_acc)[list(np.arange(24)), idx_best_train]
    train_best = np.array(best_train_acc_per_model)[list(np.arange(24)), idx_best_train]
    val_best = np.array(best_val_acc_per_model)[list(np.arange(24)), idx_best_train]
    print('Best Average Training Accuracy: ', np.mean(train_best))
    print('Best Average Validation Accuracy: ', np.mean(val_best))
    print('Length of neuron: ', np.mean(len_best))

    plot_acc_size(best_acc_list=best_train_acc_per_gen, mean_acc_list=mean_train_acc_per_gen, mean_length=mean_length_per_gen, len_factor=len_factor, lbl="Training")
    plot_acc_size(best_acc_list=best_val_acc_per_gen, mean_acc_list=mean_val_acc_per_gen, mean_length=mean_length_per_gen, len_factor=len_factor, lbl="Validation")
    plot_eval_gen_pruning(mean_train_acc_per_gen, mean_val_acc_per_gen, y_l='Accuracy', len_factor=len_factor)
    plot_eval_gen_pruning(mean_train_loss_per_gen, mean_val_loss_per_gen, y_l='Loss', len_factor=len_factor)
    plot_eval_gen_pruning(mean_train_f1_per_gen, mean_val_f1_per_gen, y_l='F1 score', len_factor=len_factor)

if args.prune_gen_alg:
    # initializing the base model and data loaders according to the chromosome
    chromosome = '[1 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 1 1 0]'
    train_df_list, val_df_list = split_participant_wise(normalized_data_list[aby], k=24)
    column_list = str_to_column_list(chromosome=chromosome)
    num_input_features = len(column_list) - 1
    baseline_model_list = [Layer3Net(num_input_features, args.hidden_units_l1, num_classes) for i in range(24)]  # Note that you should adapt this to whatever method you use to create your models in the first place
    for i, m in enumerate(baseline_model_list):
        m.load_state_dict(torch.load(f"Saved Models/baseline_model{i}.pth"))
    # hyper parameters for the genetic algorithm
    pop_size = 50
    num_generations = 10
    num_hidden_neurons = args.hidden_units_l1
    cross_rate = [0.8]
    mutation_rate = [0.1]
    len_factor = [0]
    for cr in cross_rate:
        for mr in mutation_rate:
            for lf in len_factor:
                print("Cross Rate: ", cr, "\t", "Mutation Rate: ", mr, "\t", "Length factor: ", lf)
                gen_alg_pruning(train_df_list, val_df_list, baseline_model_list, num_input_features, cr, mr, lf, pop_size,num_generations, num_hidden_neurons)
    # lists for performance parameters for plots
    # best_val_acc_per_model = [[] for _ in range(24)]
    # best_train_acc_per_model = [[] for _ in range(24)]
    # mean_val_acc_per_model = [[] for _ in range(24)]
    # mean_train_acc_per_model = [[] for _ in range(24)]
    # mean_length_per_model = [[] for _ in range(24)]
    # mean_val_loss_per_model = [[] for _ in range(24)]
    # mean_train_loss_per_model = [[] for _ in range(24)]
    # mean_train_f1_per_model = [[] for _ in range(24)]
    # mean_val_f1_per_model = [[] for _ in range(24)]
    #
    # for i in range(len(train_df_list)):
    #     population = initialize_pop(dna_size=num_hidden_neurons, pop_size=pop_size)
    #     print('Model ', i)
    #     train_loader, val_loader, _ = get_data_loaders(train_df_list[i], val_df_list[i], test_df, column_list, args)
    #
    #     base_model = baseline_model_list[i]
    #     best_model = Layer3Net(num_input_features, args.hidden_units_l1, num_classes=num_classes)
    #     optimizer = optim.Adam(base_model.parameters(), lr=args.lr)
    #     # load model and run validation/test set through it
    #     # base_model, _ = load_model(base_model, optimizer, best_path)
    #     eval_list = get_train_val_eval(base_model, train_loader, val_loader, args)
    #
    #     best_train_acc_per_model[i].append(eval_list[1][0])
    #     best_val_acc_per_model[i].append(eval_list[1][1])
    #     mean_val_acc_per_model[i].append(eval_list[1][1])
    #     mean_val_loss_per_model[i].append(eval_list[0][1])
    #     mean_length_per_model[i].append(20)
    #     mean_train_acc_per_model[i].append(eval_list[1][0])
    #     mean_train_loss_per_model[i].append(eval_list[0][0])
    #     mean_train_f1_per_model[i].append(eval_list[2][0])
    #     mean_val_f1_per_model[i].append(eval_list[2][1])
    #
    #     # print('Evaluation through sk-learn metrics')
    #     # print('Training F1 score: ', eval_list[2][0])
    #     # print('Validation F1 score: ', eval_list[2][1])
    #
    #     # pruning algorithm through genetic algorithm
    #     for gen in range(num_generations):
    #         # check if all the elements in a row are zero, make these rows random
    #         row_idx_zero = np.where(~population.any(axis=1))[0]
    #         if len(row_idx_zero) > 0:
    #             dna_len = population.shape[1]
    #             population[row_idx_zero] = np.random.randint(0, 2, size=dna_len)
    #         # evaluate the population
    #         val_acc_list = []
    #         val_loss_list = []
    #         train_acc_list = []
    #         train_loss_list = []
    #         train_f1_list = []
    #         val_f1_list = []
    #         # best_val_acc = 0
    #         for chromosome in population:
    #             # new number of hidden neurons for the new model
    #             new_num_hidden_neurons = np.sum(chromosome)
    #             # copy parameters to new model
    #             m_temp = Layer3Net(num_input_features, new_num_hidden_neurons, num_classes=num_classes)
    #             new_model = get_new_model(base_model, m_temp, chromosome=chromosome)
    #             # run validation for the new model
    #             eval_list = get_train_val_eval(new_model, train_loader, val_loader, args)
    #             # evaluation parameters
    #             train_loss_list.append(eval_list[0][0])
    #             val_loss_list.append(eval_list[0][1])
    #             train_acc_list.append(eval_list[1][0])
    #             val_acc_list.append(eval_list[1][1])
    #             train_f1_list.append(eval_list[2][0])
    #             val_f1_list.append(eval_list[2][1])
    #             # saving best model after pruning
    #             # if (eval_list[1][1] > best_val_acc):
    #             #     best_model = new_model
    #             #     best_val_acc = eval_list[1][1]
    #             #     torch.save({
    #             #         'model_state_dict': new_model.state_dict(),
    #             #         'optimizer_state_dict': optimizer.state_dict(),
    #             #         'val_accuracy': eval_list[1][1],
    #             #     }, best_after_pruning_genetic_alg)
    #         # best validation accuracy, chromosome
    #         best_idx = train_acc_list.index(max(train_acc_list))
    #         m_length = np.sum(population) / len(population)
    #         # updating the lists for the plots
    #         best_train_acc_per_model[i].append(max(train_acc_list))
    #         best_val_acc_per_model[i].append(max(val_acc_list))
    #         mean_val_acc_per_model[i].append(np.mean(np.array(val_acc_list)))
    #         mean_val_loss_per_model[i].append(np.mean(np.array(val_loss_list)))
    #         mean_length_per_model[i].append(m_length)
    #         mean_train_acc_per_model[i].append(np.mean(np.array(train_acc_list)))
    #         mean_train_loss_per_model[i].append(np.mean(np.array(train_loss_list)))
    #         mean_train_f1_per_model[i].append(np.mean(np.array(train_f1_list)))
    #         mean_val_f1_per_model[i].append(np.mean(np.array(val_f1_list)))
    #         # information about pruning performance
    #         # print('Generation', gen + 1)
    #         # print('Length of Chromosome', len(np.argwhere(population[best_idx, :] == 1)))
    #         # print('Average size of hidden layer', m_length)
    #         # print('Best training Accuracy', max(train_acc_list))
    #         # print('Corresponding chromosome', population[best_idx, :])
    #         # selecting new population, top 25
    #         population = select_top_dual(population, train_acc_list, top_count=10, len_factor=len_factor)
    #         # select probabilistic
    #         # population = select_probability(population, val_acc_list, len_factor=len_factor)
    #         # crossover and mutate
    #         population = cross_mutate(population, cross_rate, mutation_rate)
    #
    # # plotting the accuracy and mean size
    # best_train_acc_per_gen = np.mean(np.array(best_train_acc_per_model), axis=0)
    # best_val_acc_per_gen = np.mean(np.array(best_val_acc_per_model), axis=0)
    # mean_val_acc_per_gen = np.mean(np.array(mean_val_acc_per_model), axis=0)
    # mean_val_loss_per_gen = np.mean(np.array(mean_val_loss_per_model), axis=0)
    # mean_length_per_gen = np.mean(np.array(mean_length_per_model), axis=0)
    # mean_train_acc_per_gen = np.mean(np.array(mean_train_acc_per_model), axis=0)
    # mean_train_loss_per_gen = np.mean(np.array(mean_train_loss_per_model), axis=0)
    # mean_train_f1_per_gen = np.mean(np.array(mean_train_f1_per_model), axis=0)
    # mean_val_f1_per_gen = np.mean(np.array(mean_val_f1_per_model), axis=0)
    #
    # print('Best Average Training Accuracy: ', np.mean(np.max(np.array(best_train_acc_per_model), axis=1)))
    # print('Best Average Validation Accuracy: ', np.mean(np.max(np.array(best_val_acc_per_model), axis=1)))
    #
    # plot_acc_size(best_acc_list=best_train_acc_per_gen, mean_acc_list=mean_train_acc_per_gen, mean_length=mean_length_per_gen, len_factor=len_factor, lbl="Training")
    # plot_acc_size(best_acc_list=best_val_acc_per_gen, mean_acc_list=mean_val_acc_per_gen, mean_length=mean_length_per_gen, len_factor=len_factor, lbl="Validation")
    # plot_eval_gen_pruning(mean_train_acc_per_gen, mean_val_acc_per_gen, y_l='Accuracy', len_factor=len_factor)
    # plot_eval_gen_pruning(mean_train_loss_per_gen, mean_val_loss_per_gen, y_l='Loss', len_factor=len_factor)
    # plot_eval_gen_pruning(mean_train_f1_per_gen, mean_val_f1_per_gen, y_l='F1 score', len_factor=len_factor)

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
