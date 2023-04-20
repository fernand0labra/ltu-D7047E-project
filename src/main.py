import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils
import matplotlib.pyplot as plt
from controller.controller import PIDController
from system.system import VanDerPol
from system.data import DynamicSystemDataset
from system.rnn import SystemRNN
from torch.utils.tensorboard import SummaryWriter


# Define execution device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("\n###########################################################################")
print("#                    Identification of System with NN                     #")
print("###########################################################################\n")

# Instantiate the VanDerPol system
sys = VanDerPol(mu=1, dt=0.01)

PATH = "./data/system/van_der_pol.csv"
if not os.path.isfile(PATH):
    # Input signal properties (for identification)
    data_count = 1000
    num_steps = 10
    step_width = 50
    max_step_height = 3
    max_signal_val = 3.0

    dataset = DynamicSystemDataset.init_create_dataset(sys, data_count, num_steps, step_width, max_step_height, max_signal_val)
    dataset.save_dataset(PATH)
else:
    dataset = DynamicSystemDataset.init_load_dataset(PATH)

# Hyperparameter definition
epochs = 10
batch_size = 1
learning_rate = 0.01

# Instantiate the SystemRNN model
input_dim = 1  # Input dimension (e.g., number of features)
hidden_dim = 8  # Number of hidden units in the RNN
output_dim = 1  # Output dimension (e.g., prediction)
sys_rnn = SystemRNN(input_dim, hidden_dim, output_dim)
sys_rnn.to(device)

PATH = './models/sys_rnn.pth'
if not os.path.isfile(PATH):
    sys_rnn.init_hidden(batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    utils.train(epochs, trainloader, optim.Adam(sys_rnn.parameters(), lr=learning_rate), nn.MSELoss(), sys_rnn, device)
    torch.save(sys_rnn.state_dict(), PATH)
else:
    sys_rnn.load_state_dict(torch.load(PATH))

testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
utils.test(testloader, sys_rnn, device)
utils.plot_signal_data(sys, sys_rnn, device)


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