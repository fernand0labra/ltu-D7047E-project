import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader


def generate_random_steps(num_steps, max_step_height, max_signal_val,
                          step_width):
    """
    Generates a signal of random steps.

    Args:
        num_steps (int): Number of steps in the signal.
        step_size (float): Size of each step.
        max_val (float): Maximum value of the signal.

    Returns:
        numpy.array: A 1D array representing the generated signal.
    """
    signal = np.zeros(num_steps *
                      step_width)  # Initialize the signal with zeros
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
        writer = csv.writer(file,
                            delimiter=',',
                            quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)

        # write the header row
        writer.writerow(['input', 'output'])

        # iterate over the dataset and write each sample to the CSV file
        for batch_idx, (input_data, output_data) in enumerate(dataloader):

            # extract the data for each sample
            input_column = input_data.tolist()[0]
            output_column = output_data.tolist()[0]

            # write the data to the CSV file
            writer.writerow([input_column, output_column])


def plot_signal_data(data_loader: DataLoader, sys, net: nn.Module, device):
    input_signals, output_signals = next(iter(data_loader))
    input_signal = torch.unsqueeze(torch.unsqueeze(input_signals[0], 0), -1).to(device)
    output_signal = output_signals[0]

    generated_signal, _ = net(input_signal)

    ts = data_loader.dataset.dataset.ts
    # Plot the generated input signal
    plt.plot(ts, torch.squeeze(input_signal.cpu()))
    plt.xlabel('Time (s)')
    plt.ylabel('u(t)')
    plt.title('A Sample Input Signal for Identification')
    plt.show()

    # Plot the generated output signal
    plt.plot(ts, output_signal, label="Sys Output")
    plt.plot(ts, torch.squeeze(generated_signal.detach().cpu()), label="Model Output")
    plt.xlabel('Time (s)')
    plt.ylabel('y(t)')
    plt.title('Van der Pol System Output')
    plt.legend()
    plt.show()


def train(epochs, train_loader, val_loader, optimizer, criterion,
          net: nn.Module, device):
    writer = SummaryWriter("./logs")

    # Train the network
    for epoch in range(epochs):  # Loop over the dataset multiple times
        net.train()
        train_loss = 0.0
        counter = 0
        for batch_index, (inputs, ground_truths) in enumerate(train_loader, 0):
            inputs = torch.unsqueeze(inputs, -1).to(device)
            ground_truths = torch.unsqueeze(ground_truths, -1).to(device)

            # zero the gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = net(inputs)
            loss = criterion(outputs, ground_truths)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            counter += 1

        train_loss /= counter
        val_loss = test(val_loader, criterion, net, device)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        print(
            f'[{int((epoch + 1.0) / epochs * 100)}%] train loss: {train_loss:.3f}, val loss: {val_loss:.3f}'
        )

    writer.flush()
    writer.close()


def test(data_loader, criterion, net: nn.Module, device):
    net.eval()
    loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        counter = 0
        for (inputs, ground_truths) in data_loader:
            inputs = torch.unsqueeze(inputs, -1).to(device)
            ground_truths = torch.unsqueeze(ground_truths, -1).to(device)
            outputs, _ = net(inputs)
            loss += criterion(outputs, ground_truths)
            counter += 1

    return loss / counter