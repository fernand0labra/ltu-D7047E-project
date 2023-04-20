import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter


def generate_random_steps(num_steps, max_step_height, max_signal_val, step_width):
    """
    Generates a signal of random steps.

    Args:
        num_steps (int): Number of steps in the signal.
        step_size (float): Size of each step.
        max_val (float): Maximum value of the signal.

    Returns:
        numpy.array: A 1D array representing the generated signal.
    """
    signal = np.zeros(num_steps * step_width)  # Initialize the signal with zeros
    current_val = 0  # Start from 0

    for i in range(num_steps):
        # Generate a random step between -step_size and step_size
        step = np.random.uniform(-max_step_height, max_step_height)
        # Update the current value with the step
        current_val += step
        # Clip the current value to be within the maximum value
        current_val = np.clip(current_val, -max_signal_val, max_signal_val)
        # Assign the current value to the signal
        signal[i * step_width:(i + 1) * step_width] = current_val

    return signal


def save_signal_data(path, dataloader):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # write the header row
        writer.writerow(['input', 'output'])

        # iterate over the dataset and write each sample to the CSV file
        for batch_idx, (input_data, output_data) in enumerate(dataloader):

            # extract the data for each sample
            input_column = input_data.tolist()[0]
            output_column = output_data.tolist()[0]

            # write the data to the CSV file
            writer.writerow([input_column, output_column])


def plot_signal_data(sys, net, device):
    num_steps = 100
    step_width = 100
    max_step_height = 3
    max_signal_val = 3.0

    input_signal = generate_random_steps(num_steps, max_step_height, max_signal_val, step_width)
    ts, output_signal = sys.run(input_signal, 0)

    u_tensor = torch.tensor(input_signal, dtype=torch.float32).unsqueeze(0).to(device)
    generated_signal = net.forward(u_tensor)

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
    plt.plot(ts, generated_signal.detach().numpy())
    plt.xlabel('Time (s)')
    plt.ylabel('rnn(t)')
    plt.title('RNN System Output')
    plt.show()


def train(epochs, trainloader, optimizer, criterion, net, device):
    writer = SummaryWriter("../logs")

    # Train the network
    for epoch in range(epochs):  # Loop over the dataset multiple times

        running_loss = 0.0
        for i, (inputs, ground_truth) in enumerate(trainloader, 0):
            inputs = inputs.to(device)
            ground_truth = ground_truth[0].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).to(device)  # GPU
            loss = criterion(outputs, ground_truth)
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print(f'[{epoch + 1}] loss: {running_loss / 1000:.3f}')

    writer.flush()
    writer.close()


def test(testloader, net, device):
    mse = 0
    total = 0
    loss = nn.MSELoss()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for (inputs, ground_truth) in testloader:

            inputs = inputs.to(device)
            ground_truth = ground_truth[0].to(device)

            # calculate outputs by running images through the network
            outputs = net(inputs).to(device)
            # the class with the highest energy is what we choose as prediction
            total += inputs[0].size(0)
            mse += loss(outputs, ground_truth)

    print(f'Average MSE of the network on the 1000 signal points: {100 * mse / total:.3f}')

