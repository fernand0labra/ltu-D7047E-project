import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from src.utils import generate_random_steps
from src.var import *


# Define the dynamic system dataset
class DynamicSystemDataset(Dataset):

    def __init__(self):
        self.inputs = None
        self.outputs = None

    @classmethod
    def init_create_dataset(cls, system):
        # Create empty input and outputs tensors
        # with shape (data_count, num_steps * step_width) and data type float32
        tensor_shape = (DATA_COUNT, NUM_STEPS * STEP_WIDTH)

        dataset = DynamicSystemDataset()
        dataset.inputs = torch.empty(tensor_shape, dtype=torch.float32)
        dataset.outputs = torch.empty(tensor_shape, dtype=torch.float32)

        for i in tqdm(range(DATA_COUNT)):
            u = generate_random_steps()
            _, y = system.run(u, 0)
            dataset.inputs[i, :] = torch.from_numpy(u)
            dataset.outputs[i, :] = torch.from_numpy(y)

        # Save dataset
        torch.save(dataset, DATASET_PATH)

        return dataset

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]