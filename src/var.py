import torch


# Define execution device (CPU or GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define model and dataset paths
DATASET_PATH = "./data/system/van_der_pol.pt"
MODEL_PATH = lambda type: './models/sys_rnn.pth' if type == "rnn" else './models/sys_lstm.pth'


# Define signal generation parameters
DATA_COUNT = 1000

NUM_STEPS = 10
STEP_WIDTH = 100

MAX_STEP_HEIGHT = 3
MAX_SIGNAL_VAL = 3.0