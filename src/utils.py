import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter

from src.var import *

def generate_random_steps():
    """
    Generates a signal of random steps.

    Args:
        num_steps (int): Number of steps in the signal.
        step_size (float): Size of each step.
        max_val (float): Maximum value of the signal.

    Returns:
        numpy.array: A 1D array representing the generated signal.
    """
    signal = np.zeros(NUM_STEPS * STEP_WIDTH)  # Initialize the signal with zeros
    current_val = 0  # Start from 0

    for i in range(NUM_STEPS):
        # Generate a random step between -step_size and step_size
        step = np.random.uniform(-MAX_STEP_HEIGHT, MAX_STEP_HEIGHT)
        # Update the current value with the step
        current_val += step
        # Clip the current value to be within the maximum value
        current_val = np.clip(current_val, -MAX_SIGNAL_VAL, MAX_SIGNAL_VAL)
        # Assign the current value to the signal
        signal[i * STEP_WIDTH:(i + 1) * STEP_WIDTH] = current_val

    return signal


def plot_signal_data(sys, net):
    # [sequence_length] = [NUM_STEPS * STEP_WIDTH]
    input_signal = generate_random_steps()
    ts, output_signal = sys.run(input_signal, 0)

    # [batch_size, sequence_length, input_size] = [1, NUM_STEPS * STEP_WIDTH, 1]
    u_tensor = torch.tensor(input_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(DEVICE)

    hidden_state = net.init_hidden(1)
    generated_signal, hidden_state = net.forward(u_tensor, hidden_state)

    # Plot the generated input signal
    plt.plot(ts, input_signal)
    plt.xlabel('Time (s)')
    plt.ylabel('u(t)')
    plt.title('A Sample Input Signal for Identification')
    plt.show()

    # Plot the generated output signal
    plt.plot(ts, output_signal)
    plt.xlabel('Time (s)')
    plt.ylabel('y(t)')
    plt.title('Van der Pol System Output')
    plt.show()

    # Plot the generated output signal
    plt.plot(ts, generated_signal.view(-1).cpu().detach().numpy())
    plt.xlabel('Time (s)')
    plt.ylabel('rnn(t)')
    plt.title('RNN System Output')
    plt.show()


def train(epochs, trainloader, optimizer, criterion, net):
    writer = SummaryWriter("../logs")

    # Train the network
    for epoch in range(epochs):  # Loop over the dataset multiple times

        running_loss = 0.0
        for inputs, ground_truth in trainloader:  # input/target [batch_size,  NUM_STEPS * STEP_WIDTH]

            hidden_state = net.init_hidden(len(inputs))

            # zero the parameter gradients
            optimizer.zero_grad()

            # input/target [batch_size, sequence_length, input_size] = [batch_size, NUM_STEPS * STEP_WIDTH, 1]

            inputs = inputs.unsqueeze(2).to(DEVICE)
            ground_truth = ground_truth.unsqueeze(2).to(DEVICE)

            # forward + backward + optimize
            outputs, hidden_state = net(inputs, hidden_state)
            outputs = outputs.to(DEVICE)  # GPU

            loss = criterion(outputs, ground_truth)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        writer.add_scalar("Loss/train", running_loss, epoch)
        print(f'[{epoch + 1}] loss: {running_loss / DATA_COUNT:.3f}')

    writer.flush()
    writer.close()


def test(testloader, net):
    mse = 0
    loss = nn.MSELoss()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for inputs, ground_truth in testloader:

            hidden_state = net.init_hidden(len(inputs))

            # input/target [batch_size, sequence_length, input_size] = [batch_size, NUM_STEPS * STEP_WIDTH, 1]
            inputs = inputs.unsqueeze(2).to(DEVICE)
            ground_truth = ground_truth.unsqueeze(2).to(DEVICE)
            
            outputs, hidden_state = net(inputs, hidden_state)
            outputs = outputs.to(DEVICE)
            
            mse += loss(outputs, ground_truth)

    print(f'Average MSE of the network on the {DATA_COUNT} signal points: {100 * mse / DATA_COUNT:.3f}')

