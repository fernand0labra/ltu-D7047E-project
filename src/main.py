import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.var import *
from src.utils import train, test, plot_signal_data
from src.controller.controller import PIDController
from src.system.system import VanDerPol
from src.system.data import DynamicSystemDataset
from src.system.model import SystemRNN, SystemLSTM

print("\n###########################################################################")
print("#                    Identification of System with NN                     #")
print("###########################################################################\n")

# Instantiate the VanDerPol system
sys = VanDerPol(mu=1, dt=0.01)

# Generate dataset from system's output
dataset = DynamicSystemDataset.init_create_dataset(sys) if not os.path.isfile(DATASET_PATH) else torch.load(DATASET_PATH)

# Hyperparameter definition
epochs = 15
batch_size = DATA_COUNT // 100
learning_rate = 0.015

# Instantiate the SystemRNN model
input_dim = 1  # Input dimension
hidden_dim = 50  # Number of hidden units in the RNN
output_dim = 1  # Output dimension

network_type = "lstm"

sys_rnn = SystemLSTM(input_dim, hidden_dim, output_dim)
sys_rnn.to(DEVICE)

if not os.path.isfile(MODEL_PATH(network_type)):
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train(epochs, trainloader, optim.Adam(sys_rnn.parameters(), lr=learning_rate), nn.MSELoss(), sys_rnn)
    torch.save(sys_rnn.state_dict(), MODEL_PATH(network_type))
else:
    sys_rnn.load_state_dict(torch.load(MODEL_PATH(network_type)))

testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test(testloader, sys_rnn)
plot_signal_data(sys, sys_rnn)


print("\n###########################################################################")
print("#                  Identification of Controller with NN                   #")
print("###########################################################################\n")

kp, ki, kd = 20, 20, 10
pid = PIDController(kp, ki, kd)

t0 = 0
tf = 40.0  # final time
ts = np.zeros(int(tf/sys.dt))


for reference in np.arange(-1.5, 1.6, 0.5):
    ts, y = sys.run(np.ones(len(ts)) * reference, t0)
    # Plot the results
    plt.plot(ts, y, label=f'r = {reference}')

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('y(t)')
plt.title('Van der Pol System Output')
plt.show()

for reference in np.arange(-1.5, 1.6, 0.5):
    ts, y = sys.run_controlled(ts, t0, pid, reference)
    # Plot the results
    plt.plot(ts, y, label=f'r = {reference}')

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('y(t)')
plt.title('PID Controller - Van der Pol System Output')
plt.show()