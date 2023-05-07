import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from var import *
from utils import generate_random_steps

class DynamicSystemDataset(Dataset):

    def __init__(self, system):

        tensor_shape = (DATA_COUNT, NUM_STEPS * STEP_WIDTH)

        self.inputs = torch.empty(tensor_shape, dtype=torch.float32)
        self.outputs = torch.empty(tensor_shape, dtype=torch.float32)

        for i in tqdm(range(DATA_COUNT)):
            u = generate_random_steps()
            ts, y = system.run(u, 0)
            self.inputs[i, :] = torch.from_numpy(u)
            self.outputs[i, :] = torch.from_numpy(y)

        self.ts = ts

        # Save dataset
        torch.save(self, DATASET_PATH)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]