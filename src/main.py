import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from var import *
from utils import train_validate, test, plot_signal_data
from controller.controller import PIDController
from system.system import VanDerPol
from system.data import DynamicSystemDataset
from system.model import SystemRNN


###########################################################################
#                             VanDerPol System                            #
###########################################################################

# Instantiate the VanDerPol system
sys = VanDerPol(mu=1, dt=0.01)

t0 = 0
tf = 40.0  # final time
ts = np.zeros(int(tf / sys.dt))

for u in np.arange(-1.5, 1.6, 0.5):
    ts, y = sys.run(np.ones(len(ts)) * u, t0)
    # Plot the results
    plt.plot(ts, y, label=f'u = {u}')

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('y(t)')
plt.title('Van der Pol System Output')
plt.show()

###########################################################################
#                          Simple PID Controller                          #
###########################################################################

kp, ki, kd = 20, 10, 10
pid = PIDController(kp, ki, kd, -3, 3)

t0 = 0
tf = 10.0  # final time
ts = np.zeros(int(tf / sys.dt))

u_fig = plt.figure()
y_fig = plt.figure()

for reference in np.arange(1.5, -1.6, -0.5):
    ts, y, u = sys.run_controlled(np.ones(len(ts)) * reference, t0, pid)
    # Plot the results
    plt.figure(y_fig)
    plt.plot(ts, y, label=f'r = {reference}')

    plt.figure(u_fig)
    plt.plot(ts, u, label=f'r = {reference}')

plt.figure(u_fig)
plt.legend(loc = 'upper right')
plt.xlabel('Time (s)')
plt.ylabel('u(t)')
plt.title('PID Controller Output')
plt.grid(True)

plt.figure(y_fig)
plt.legend(loc = 'upper right')
plt.xlabel('Time (s)')
plt.ylabel('y(t)')
plt.title('Van der Pol System Output under PID Control')
plt.grid(True)

###########################################################################
#                    Identification of System with NN                     #
###########################################################################

# Generate dataset from system's output
dataset = DynamicSystemDataset.init_create_dataset(sys) if not os.path.isfile(DATASET_PATH) else torch.load(DATASET_PATH)
plot_signal_data(dataset, num_samples=3)

# Define the sizes of the training, validation, and test sets
train_size, val_size, test_size = 0.6, 0.2, 0.2

# Use random_split to split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Network types (Vanilla RNN and LSTM)
rnn_type_list = ['vanilla', 'lstm']

# Network parameter definitionn
hidden_size = 100
nonlinearity = 'tanh'
dropout = 0.0
num_layers = 6

# Hyperparameter definition
epochs = 100
batch_size = len(train_dataset) // 100
learning_rate = 0.00001
criterion = nn.MSELoss()

for rnn_type in rnn_type_list:
    sys_rnn = SystemRNN(hidden_size=hidden_size, num_layers=num_layers, nonlinearity=nonlinearity, dropout=dropout, rnn_type=rnn_type).to(DEVICE)

    if not os.path.isfile(MODEL_PATH(rnn_type)):
        os.makedirs(os.path.dirname(MODEL_PATH(rnn_type)), exist_ok=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        train_validate(epochs, train_loader, val_loader, optim.Adam(sys_rnn.parameters(), lr=learning_rate), criterion, sys_rnn)
        
        torch.save(sys_rnn.state_dict(), MODEL_PATH(rnn_type))
    else:
        sys_rnn.load_state_dict(torch.load(MODEL_PATH(rnn_type)))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loss = test(test_loader, criterion, sys_rnn)
    print(f'{rnn_type.capitalize()} Test loss = {test_loss:.3f}')

    plot_signal_data(test_dataset.dataset, net=sys_rnn, num_samples=3)