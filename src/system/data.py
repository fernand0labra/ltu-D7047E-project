import ast
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import src.utils as utils
import pandas as pd


# Define the dynamic system dataset
class DynamicSystemDataset(Dataset):

    def __init__(self):
        self.inputs = None
        self.outputs = None

    @classmethod
    def init_create_dataset(cls, system, data_count, num_steps, step_width, max_step_height, max_signal_val):
        # Create empty input and outputs tensors
        # with shape (data_count, num_steps * step_width) and data type float32
        tensor_shape = (data_count, num_steps * step_width)

        dataset = DynamicSystemDataset()
        dataset.inputs = torch.empty(tensor_shape, dtype=torch.float32)
        dataset.outputs = torch.empty(tensor_shape, dtype=torch.float32)

        for i in tqdm(range(data_count)):
            u = utils.generate_random_steps(num_steps, max_step_height, max_signal_val, step_width)
            _, y = system.run(u, 0)
            dataset.inputs[i, :] = torch.from_numpy(u)
            dataset.outputs[i, :] = torch.from_numpy(y)

        return dataset

    @classmethod
    def init_load_dataset(cls, path):
        dataset_file = pd.read_csv(path)
        inputs = list(map(ast.literal_eval, dataset_file["input"]))
        outputs = list(map(ast.literal_eval, dataset_file["output"]))

        dataset = DynamicSystemDataset()

        dataset.inputs = torch.tensor(inputs, dtype=torch.float32)
        dataset.outputs = torch.tensor(outputs, dtype=torch.float32)

        return dataset

    def save_dataset(self, path):
        dataloader = torch.utils.data.DataLoader(self, batch_size=1, shuffle=True, num_workers=0)
        utils.save_signal_data(path, dataloader)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]