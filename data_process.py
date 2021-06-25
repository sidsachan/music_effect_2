import pandas as pd
import numpy as np
from utils import box_plot, plot_participant_median

# function to remove useless data, median filter and then normalize
def pre_process(file_loc, plot_data = False):
    raw_df = pd.read_excel(file_loc)
    num_participants = 24
    num_features = 25

    # dropping the participant id
    raw_df.drop(raw_df.columns[0], axis=1, inplace=True)

    # dropping the column labels
    raw_df.drop([0], inplace=True)

    # dividing into alpha, beta and gamma channels, index start from 1
    df_aby =[]
    df_aby.append(raw_df.iloc[raw_df.index % 3 == 1])
    df_aby.append(raw_df.iloc[raw_df.index % 3 == 2])
    df_aby.append(raw_df.iloc[raw_df.index % 3 == 0])
    normalized_df_aby = []

    titles = ['alpha', 'beta', 'gamma']
    if plot_data:
        # finding and plotting median signals for each participant
        participant_stats = np.zeros((num_participants, raw_df.shape[1]))
        for i in range(1, 25):
            participant_stats[i - 1, :] = raw_df[num_participants * (i - 1):num_participants * i].median()
        plot_participant_median(participant_stats[14:20,:], start =14)
        # box-plot the data to understand the spread
        for i in range(3):
            title_str = 'Box Plot of signals in ' + titles[i] + ' frequency range'
            box_plot(df_aby[i].values[:, :-1], title=title_str, x_l='Features', y_l='Values')

    # adjusting the outlier (65535) in the alpha frequency range feature 21 to be the median value
    max_outlier_value = np.amax(df_aby[0].values[:, 20])
    idx_outlier = np.argwhere(df_aby[0].values[:, 20]==max_outlier_value)
    df_aby[0].values[list(idx_outlier), 20] = np.median(df_aby[0].values[:, 20])

    # column-wise normalize between 0 and 1, except last column
    # for df in df_aby:
    #     result = df.copy()
    #     for column in df.iloc[:, :-1]:
    #         min_value = df[column].min()      # minimum
    #         max_value = df[column].max()      # maximum
    #         mean_value = df[column].mean()     # mean
    #         std_value = df[column].std()      # standard deviation
    #         # min max normalization
    #         result[column] = (df[column] - min_value) / (max_value - min_value)
    #         # mean std normalization
    #         # result[column] = (df[column] - mean_value) / (std_value)
    #     normalized_df_aby.append(result)

    # column-wise and participant - wise normalize between 0 and 1, except last column
    eps = 1e-8
    for df in df_aby:
        result = df.copy()
        for column in range(num_features):
            for p in range(1, num_participants+1):
                min_value = min(df.iloc[(p-1)*8:p*8, column])      # minimum
                max_value = max(df.iloc[(p-1)*8:p*8, column])      # maximum
                # mean_value = df[column][(p-1)*24:p*24].mean()     # mean
                # std_value = df[column][(p-1)*24:p*24].std()      # standard deviation
                # min max normalization
                result.iloc[(p-1)*8:p*8, column] = (df.iloc[(p-1)*8:p*8, column] - min_value) / (max_value - min_value+eps)
                # mean std normalization
                # result[column][(p-1)*24:p*24] = (df[column][(p-1)*24:p*24] - mean_value) / (std_value)
        normalized_df_aby.append(result)
    # combining the beta and gamma wave data
    by_combined = pd.DataFrame(np.hstack((normalized_df_aby[1].iloc[:, :-1].values, normalized_df_aby[2].iloc[:, :].values)))
    aby_combined = pd.DataFrame(np.hstack((normalized_df_aby[0].iloc[:, :-1].values,normalized_df_aby[1].iloc[:, :-1].values, normalized_df_aby[2].iloc[:, :].values)))
    normalized_df_aby.append(by_combined)
    normalized_df_aby.append(aby_combined)
    return normalized_df_aby


# function to split the data into train validation and test data
def split(df, train_frac=0.8, val_frac=0.1):
    train, val, test = np.split(df.sample(frac=1, random_state=96),
                                [int(train_frac * len(df)), int((train_frac + val_frac) * len(df))])
    return train, val, test


# Function to split the data into train and validation according to leave one out cross-validation
# return a list of train and validation sets
def split_participant_wise(df, num_participants=24, observations_per_participant=8, k=6):
    train_list = []
    val_list = []
    # for i in range(num_participants):
    #     val_index_list = list(range(i*observations_per_participant, (i+1)*observations_per_participant))
    #     val_df = df.iloc[val_index_list, :]
    #     before = list(range(i*observations_per_participant))
    #     after = list(range((i+1)*observations_per_participant, num_participants*observations_per_participant))
    #     train_index_list = before+after
    #     train_df = df.iloc[train_index_list, :]
    #     train_df = train_df.sample(frac=1)
    #     train_list.append(train_df)
    #     val_list.append(val_df)
    num_participant_per_set = int(num_participants/k)
    num_observation_per_set = num_participant_per_set * observations_per_participant
    total_observation = num_participants * observations_per_participant
    for i in range(k):
        val_index_list = list(range(i * num_observation_per_set, (i + 1)*num_observation_per_set))
        val_df = df.iloc[val_index_list, :]
        before = list(range(i *num_observation_per_set))
        after = list(range((i + 1) * num_observation_per_set, total_observation))
        train_index_list = before + after
        train_df = df.iloc[train_index_list, :]
        train_df = train_df.sample(frac=1)
        train_list.append(train_df)
        val_list.append(val_df)
    return train_list, val_list
