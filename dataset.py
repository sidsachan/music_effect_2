import torch
import torch.utils.data


# define a customise torch dataset
class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data_tensor = torch.Tensor(df.values.astype(float))

    # a function to get items by index
    def __getitem__(self, index):
        input = self.data_tensor[index][0:-1]
        # target 0,1,2 to be compatible with nn.CrossEntropyLoss
        target = (self.data_tensor[index][-1] - 1).long()

        return input, target

    # a function to count samples
    def __len__(self):
        n, _ = self.data_tensor.shape
        return n
