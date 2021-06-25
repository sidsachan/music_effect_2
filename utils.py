import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from sklearn.metrics import accuracy_score, f1_score
from learning import validate
from dataset import DataFrameDataset
from torch.utils.data import DataLoader


# function to plot losses
def plot_loss(train_loss_list, val_loss_list):
    x = np.arange(len(train_loss_list))
    plt.plot(x, train_loss_list, label = 'Training')
    plt.plot(x, val_loss_list, label = 'Validation')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Average loss')
    plt.title('Average Losses during training')
    plt.savefig('Plots/loss_plot.png', dpi=300)
    plt.show()


# function to return the cosine similarity between all the activations
def cosine_sim(activations):
    # activations is a tensor is of dim n x hidden-dim
    n, h = activations.shape
    activations = torch.transpose(activations,0,1)
    activations_norm = activations / activations.norm(dim=1)[:, None]
    res = torch.mm(activations_norm, activations_norm.transpose(0, 1))
    return res


# function to return pairs of similar or complementary neurons for a given threshold in degrees
def determine_similar_pairs(angle_thresh, sim_mat):
    n = len(sim_mat)
    similar = []
    complementary = []
    # thresholds for the cosine value given angle -> so for 15 degrees,
    # angles below 15 -> then add to similar pairs
    # angle > 175 -> add to complementary list
    complementary_t = - math.cos((angle_thresh * math.pi) / 180)
    similar_t = math.cos((angle_thresh * math.pi) / 180)
    for i in range(n):
        for j in range(i):
            if sim_mat[i, j] > similar_t:
                similar.append([i, j])
            elif sim_mat[i, j] < complementary_t:
                complementary.append([i, j])
    return similar, complementary


# function to adjust the weights of a model
# if complement=True then subtract the weights and take mean
# else add the weights and take mean
def remove_neuron_pair(model_in, model_out, n_pair, complement=False):
    low = min(n_pair)
    high = max(n_pair)
    fc1_w = model_in.state_dict()['fc1.weight']     # h x n
    fc1_b = model_in.state_dict()['fc1.bias']       # h
    fc2_w = model_in.state_dict()['fc2.weight']     # c x h
    fc2_b = model_in.state_dict()['fc2.bias']       # c

    with torch.no_grad():
        # assign the weights from model_in, skip the high index, this is the removed neuron
        model_out.fc1.weight = torch.nn.Parameter(torch.cat((fc1_w[:high, :], fc1_w[high+1:, :]), 0))
        model_out.fc1.bias = torch.nn.Parameter(torch.cat((fc1_b[:high], fc1_b[high + 1:]), 0))
        model_out.fc2.weight = torch.nn.Parameter(torch.cat((fc2_w[:, :high], fc2_w[:, high + 1:]), 1))
        model_out.fc2.bias = torch.nn.Parameter(fc2_b)
        # average the weight of low index according to the neurons being similar or complementary
        if complement:
            model_out.fc1.weight[low, :] = torch.nn.Parameter((fc1_w[low, :] - fc1_w[high, :])/2)
            model_out.fc1.bias[low] = torch.nn.Parameter((fc1_b[low] - fc1_b[high])/2)
            model_out.fc2.weight[:, low] = torch.nn.Parameter((fc2_w[:, low] - fc2_w[:, high])/2)
        else:
            model_out.fc1.weight[low, :] = torch.nn.Parameter((fc1_w[low, :] + fc1_w[high, :]) / 2)
            model_out.fc1.bias[low] = torch.nn.Parameter((fc1_b[low] + fc1_b[high]) / 2)
            model_out.fc2.weight[:, low] = torch.nn.Parameter((fc2_w[:, low] + fc2_w[:, high]) / 2)

    return model_out


# function to stack all the activations from the hidden layer, over the whole dataloader
def extract_activations(model, data_loader, args):
    stacked_activations = torch.Tensor()
    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        _, activations = model(data)
        stacked_activations = torch.cat((stacked_activations, activations), dim=0)
    return stacked_activations


# for box plots of the data
def box_plot(data, title, x_l, y_l):
    fig1, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.set_xlabel(x_l)
    ax1.set_ylabel(y_l)
    ax1.boxplot(data)
    plt.savefig('Plots/'+title, dpi=300)
    plt.show()


# to line plot the median statistics
def plot_participant_median(participant_stats, start):
    n, f = participant_stats.shape
    x = np.arange(f)
    for i in range(n):
        s = 'Participant '+ str(start+i+1)
        plt.plot(x, participant_stats[i,:], label=s)
    plt.legend(loc='upper right')
    plt.xlabel('Features')
    plt.ylabel('Median signal values over all frequency')
    plt.savefig('Plots/participants.png', dpi=300)
    plt.show()


# function to calculate accuracy and f1 score of the model
def eva_model(model, data_loader, args):
    model.eval()
    pred = []
    targ = []
    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output, _ = model(data)
        pred.append(output.data.max(1, keepdim=True)[1]) # get the index of the max log-probability
        targ.append(target)
    prediction = torch.cat(pred, dim=0)
    target = torch.cat(targ, dim=0)
    n = prediction.shape[0]
    y_pred = prediction.cpu().detach().numpy().reshape((n,))
    y_true = target.cpu().detach().numpy().reshape((n,))
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    return acc, f1


# to get train and validation losses, accuracy and f1 score
def get_train_val_eval(model, train_loader, val_loader, args, verbose=False):
    train_loss, train_acc = validate(model, train_loader, args)
    _, train_f1 = eva_model(model, train_loader, args)
    val_loss, val_acc = validate(model, val_loader, args)
    _, val_f1 = eva_model(model, val_loader, args)
    if verbose:
        print('Training accuracy = ', train_acc.item())
        print('Validation accuracy = ', val_acc.item())
    return [[train_loss, val_loss], [train_acc, val_acc], [train_f1, val_f1]]


# function to plot the evaluation measures after pruning
def plot_all_evals(train_eval, val_eval, num_removed, y_l):
    x = np.arange(0, 60, 5)
    plt.plot(x, train_eval, label='Training data', marker='o')
    plt.plot(x, val_eval, label='Validation data', marker='*')

    for i in range(len(x)):
        plt.annotate('('+str(num_removed[i])+')', (x[i], train_eval[i]), xytext=(-10, 20), textcoords='offset pixels')
        plt.annotate('('+str(num_removed[i])+')', (x[i], val_eval[i]), xytext=(-10, 20), textcoords='offset pixels')
    plt.legend(loc='upper right')
    plt.xlabel('Threshold angle')
    plt.ylabel(y_l)
    plt.savefig('./Plots/'+y_l, dpi=300)
    plt.show()


# function to convert string of column list
def str_to_column_list(chromosome):
    chromosome = chromosome.replace(" ", "")
    l = []
    for c in (chromosome[1:-1]):
        l.append(ord(c) - ord('0'))
    column_list = list(np.argwhere(np.array(l) == 1).reshape(-1))
    column_list.append(len(l))
    return column_list


# getting data loaders from a given list of column(a chromosome basically) and raw (already split) dataframes
def get_data_loaders(train_df, val_df, test_df, chromosome_list, args):
    train_ = train_df[chromosome_list]
    val_ = val_df[chromosome_list]
    test_ = test_df[chromosome_list]

    # define the custom datasets
    train_dataset = DataFrameDataset(df=train_)
    val_dataset = DataFrameDataset(df=val_)
    test_dataset = DataFrameDataset(df=test_)

    # defining the data-loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)

    return train_loader, val_loader, test_loader


# function to copy model parameters given a model and a chromosome for hidden layer
def get_new_model(base_model, new_model, chromosome):
    # the full weights of the base model
    fc1_w = base_model.state_dict()['fc1.weight']  # h x n
    fc1_b = base_model.state_dict()['fc1.bias']  # h
    fc2_w = base_model.state_dict()['fc2.weight']  # c x h
    fc2_b = base_model.state_dict()['fc2.bias']  # c
    neurons_to_include = list(np.argwhere(chromosome==1).reshape(-1))
    with torch.no_grad():
        # assign the weights from model_in, skip the high index, this is the removed neuron
        new_model.fc1.weight = torch.nn.Parameter(fc1_w[neurons_to_include, :])
        new_model.fc1.bias = torch.nn.Parameter(fc1_b[neurons_to_include])
        new_model.fc2.weight = torch.nn.Parameter(fc2_w[:, neurons_to_include])
        new_model.fc2.bias = torch.nn.Parameter(fc2_b)
    return new_model


# plotting mean and best accuracy after pruning, genetic algorithm
def plot_acc_size(best_acc_list, mean_acc_list, mean_length, len_factor):
    x = np.arange(len(best_acc_list))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Training Accuracy')
    lns1 = ax1.plot(x, best_acc_list, label='Best Accuracy', color='tab:red')
    lns2 = ax1.plot(x, mean_acc_list, label='Mean Accuracy', color='tab:green')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean size of the Hidden Layer')
    lns3 = ax2.plot(x, mean_length, label='Mean Size', color='tab:blue')
    # added these three lines
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    plt.title('Training accuracy after pruning')
    save_path = 'Plots/acc_len_pruning_' + str(len_factor) + '.png'
    plt.savefig(save_path, dpi=300)
    plt.show()


# plot train and validation performance after pruning
def plot_eval_gen_pruning(train_eval, val_eval, y_l, len_factor):
    x = np.arange(len(train_eval))
    plt.plot(x, train_eval, label='Training')
    plt.plot(x, val_eval, label='Validation')
    plt.legend(loc='upper right')
    plt.xlabel('Generation')
    plt.ylabel('Average ' + y_l)
    plt.title('Average '+ y_l + ' after pruning')
    save_path = 'Plots/perf_prune_' + y_l + '_' + str(len_factor) + '.png'
    plt.savefig(save_path, dpi=300)
    plt.show()


# to plot the mutation rate impact graph
# the mean accuracy values are copied from the separate log kept during the input feature selection experiments
def plot_mutation_rate_impact():
    a = [43.15, 48.94, 54.73, 60, 60, 57.36, 67.37, 65.26, 60.52, 60.52]
    b = [42.63, 44.21, 50.52, 61.58, 59.47, 64.21, 56.84, 62.11, 58.94, 56.84]
    c = [42.11, 43.16, 46.82, 49.47, 48.42, 46.82, 48.94, 50.52, 52.63, 51.58]
    x = np.arange(10)
    plt.plot(x + 1, np.array(a) / 100, label='Mutation rate = 0.05')
    plt.plot(x + 1, np.array(b) / 100, label='Mutation rate = 0.1')
    plt.plot(x + 1, np.array(c) / 100, label='Mutation rate = 0.2')
    plt.xlabel('Generation')
    plt.ylabel('Mean validation accuracy')
    plt.legend()
    plt.savefig('./Plots/mutation_rate.png', dpi=300)
    plt.show()


# plot_mutation_rate_impact()