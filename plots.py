import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from automaton_learning import extract_events


def plot_states(valid_data, valid_states, valid_time, state_names, latent_data, model, data_name):
    """
    Plot states and events from a sequence of data.

    Parameters:
        valid_data (torch.Tensor): The list of the validation data sequences.
        valid_states (torch.Tensor): The list of states of the validation data sequences.
        valid_time (torch.Tensor): The time sequences for the validation data.
        valid_state_names (list): The state name of the validation data.
        latents (torch.Tensor): The latents of the neural network.
        index (int): It is possible to run this experiment with several sets of hyperparameters. This variable
        ascribes an integer to each experiment. If there aren't several experiments, index is simply 1.
        selected_option (string): Which model is chosen (SOM-VAE, RBM, CAT-VAE).
        file_path (String): The path to the folder where the data is located.

    Returns:
        fig (matplotlib figure): The figure.

    """
    # Convert variables
    valid_time = valid_time.detach().cpu()
    valid_data = valid_data[:, -3].detach().cpu()
    valid_states = valid_states.detach().cpu()
    latent_data = latent_data.detach().cpu()

    # Create a new figure and a set of subplots based on the length of sequence_plot.
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # Extract the timings and changes of the events
    timings, changes = extract_events(latent_data, valid_time, model)

    # Plot the original sequence.
    axes[0].plot(valid_time, valid_data, '.-')
    axes[0].set_xlim(valid_time[0], valid_time[-1] + 20)

    # Plot state labels on the current subplot.
    for idx, x in enumerate(timings):
        # Print vertical line
        x = x.cpu().numpy()
        axes[0].axvline(x=x, color='red', linestyle='--', label=f'Line at x={x}')
        axes[2].axvline(x=x, color='red', linestyle='--', label=f'Line at x={x}')

        # Print horizontal black lines representing when which clustered state is active
        if idx + 1 < len(timings):
            start = timings[idx].item()
            end = timings[idx + 1].item()
            y_values = changes[idx].item()

            axes[2].plot([start, end], [y_values, y_values], color='black', linestyle='-')
            axes[2].text((start + end) / 2, -4, f'{y_values}',
                         ha='center', va='top', fontsize=8, color='blue')
        else:
            start = timings[idx].item()
            end = valid_time[-1].item()
            y_values = changes[idx].item()
            axes[2].plot([start, end], [y_values, y_values], color='black', linestyle='-')
            axes[2].text((start + end) / 2, -4, f'{y_values}',
                         ha='center', va='top', fontsize=8, color='blue')

    # Find where the changes are in the actual classes
    diff_tensor = torch.diff(valid_states)
    change_indices = torch.nonzero(diff_tensor).squeeze()
    time_points = valid_time[change_indices]
    # Add first and last time point
    time_points = torch.cat((valid_time[0].unsqueeze(0), time_points), dim=0)

    # Plot, when the ground truth states are active
    for idx, x in enumerate(time_points):
        if idx + 1 < len(time_points):
            start = time_points[idx].item()
            end = time_points[idx+1].item()
            y_values = valid_states[change_indices[idx]].item()

            axes[1].plot([start, end], [y_values, y_values], color='black', linestyle='-')
            axes[1].text((start + end) / 2, -4, f'{y_values}',
                         ha='center', va='top', fontsize=8, color='blue')
        else:
            start = time_points[idx].item()
            end = valid_time[-1].item()
            y_values = valid_states[-1].item()

            axes[1].plot([start, end], [y_values, y_values], color='black', linestyle='-')
            axes[1].text((start + end) / 2, -4, f'{y_values}',
                         ha='center', va='top', fontsize=8, color='blue')

    # Adjust the vertical space between subplots and display the plot.
    plt.subplots_adjust(wspace=0.8, hspace=0.4)

    if model == 'denta':
        file_path = os.path.join(os.getcwd(), f'Plots/{data_name}/denta/plot.svg')
    elif model == 'somvae':
        file_path = os.path.expanduser(
            f'~/{os.path.relpath(os.getcwd(), start=os.path.expanduser("~"))}/Plots/{data_name}/somvae/plot.svg')
    elif model == 'catvae':
        file_path = os.path.expanduser(
            f'~/{os.path.relpath(os.getcwd(), start=os.path.expanduser("~"))}/Plots/{data_name}/catvae/plot.svg')
    else:
        raise ValueError('False model selection!')

    plt.savefig(file_path)

    return fig
