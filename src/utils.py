import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader

from var import *

def generate_random_steps():
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


def plot_signal_data(dataset, net=None, num_samples=1):
    # Create a data loader for the dataset
    sample_loader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
    input_signals, output_signals = next(iter(sample_loader))  # input/target [num_samples,  NUM_STEPS * STEP_WIDTH]
    
    ts = sample_loader.dataset.ts

    with torch.no_grad():
        for input_signal, output_signal in zip(input_signals, output_signals):  # input/target [NUM_STEPS * STEP_WIDTH]
            
            plt.figure(figsize=(8, 4))

            # Plot the generated input signal
            plt.subplot(1, 2, 1)
            plt.plot(ts, torch.squeeze(input_signal.cpu()))
            plt.xlabel('Time (s)')
            plt.ylabel('u(t)')
            plt.title('A Sample Input Signal for Identification')
            plt.grid((True))

            # Plot the generated output signal
            plt.subplot(1, 2, 2)
            plt.plot(ts, output_signal, label="Sys Output")
            
            if net is not None:  # input/target [batch_size, sequence_length, input_size] = [NUM_STEPS * STEP_WIDTH, 1]
                input_signal = input_signal.unsqueeze(1).to(DEVICE)
                generated_signal, _ = net(input_signal)
                plt.plot(ts, torch.squeeze(generated_signal.cpu()), label="Model Output")
            
            plt.xlabel('Time (s)')
            plt.ylabel('y(t)')
            plt.title('Van der Pol System Output')
            plt.legend()
            plt.grid(True)
        
            plt.show()


def train_validate(epochs: int, train_loader: DataLoader, val_loader: DataLoader, optimizer, criterion, net: nn.Module):
    writer = SummaryWriter("../logs")

    # Train the network
    for epoch in range(epochs):  # Loop over the dataset multiple times
        net.train()
        train_loss = 0.0
        num_batches = 0
        for inputs, ground_truth in train_loader:  # input/target [batch_size,  NUM_STEPS * STEP_WIDTH]

            # zero the parameter gradients
            optimizer.zero_grad()

            # input/target [batch_size, sequence_length, input_size] = [batch_size, NUM_STEPS * STEP_WIDTH, 1]
            inputs = inputs.unsqueeze(2).to(DEVICE)
            ground_truth = ground_truth.unsqueeze(2).to(DEVICE)

            # forward + backward + optimize
            outputs, _ = net(inputs)

            loss = criterion(outputs, ground_truth)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches
        val_loss = test(val_loader, criterion, net)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        print(f'[{int((epoch + 1.0) / epochs * 100)}%] train loss: {train_loss:.3f}, val loss: {val_loss:.3f}')

    writer.flush()
    writer.close()


def test(test_loader: DataLoader, criterion, net: nn.Module):
    net.eval()
    loss = 0
    num_batches = 0
   
    with torch.no_grad():  # since we're not training, we don't need to calculate the gradients for our outputs
        for inputs, ground_truth in test_loader:

            # input/target [batch_size, sequence_length, input_size] = [batch_size, NUM_STEPS * STEP_WIDTH, 1]
            inputs = inputs.unsqueeze(2).to(DEVICE)
            ground_truth = ground_truth.unsqueeze(2).to(DEVICE)
            
            outputs, _ = net(inputs)
            
            loss += criterion(outputs, ground_truth)
            num_batches += 1
    
    return loss/num_batches