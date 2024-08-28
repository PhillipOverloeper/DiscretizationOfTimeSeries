import torch
import os
import json
import numpy as np
import denta
from pathlib import Path
from plots import plot_states
from CatVAE import exec_catvae, instantiate_catvae, compute_mse_error
from SOMVAE import train_somvae, anomaly_detection, instantiate_somvae
from tensorboardX import SummaryWriter
from automaton_learning import extract_events
from utils import (create_folders, preprocess_data, compute_purity,
                   preprocess_data_anom, composite_f1_score, split_data)

torch.manual_seed(2)
np.random.seed(2)


def main(split, data_name, logging):
    """
    Main function to execute different models based on user input.

    This script allows the user to choose between RBM, SOMVAE, and CatVAE models,
    preprocesses the data, trains the selected model, and evaluates its performance.

    Args:
        - split (float): Split ratio for train/validation.
        - data_name (str): Name of the folder where data is stored.
        - logging (bool): Whether the training shall be logged in Tensorboard.

    Returns:
        None
    """
    # Get the model which shall be evaluated from user prompt
    selected_option = None
    while selected_option is None:
        # Display the options for the user to choose from
        print('Select an option:')
        print('1. DENTA')
        print('2. SOMVAE')
        print('3. CatVAE')

        # Prompt the user to enter their choice (1, 2 or 3)
        choice = input(' Please enter the number of the desired option (1/2/3): ')
        if choice == '1':
            selected_option = 'denta'                   # User chose DENTA
            break
        elif choice == '2':
            selected_option = 'somvae'                  # User chose SOMVAE
            break
        elif choice == '3':
            selected_option = 'catvae'                  # User chose CatVAE
            break
        else:
            print('Invalid input. Please choose 1, 2, 3 or 4.')

    if selected_option == 'somvae':
        run_somvae(data_name, split, logging)
    elif selected_option == 'catvae':
        run_catvae(data_name, split, logging)
    elif selected_option == 'denta':
        run_denta(data_name, split, logging)
    else:
        raise RuntimeError('Wrong model!')


def run_denta(data_name, ratio, logging):
    """
    Train RBM model on the provided data.

    Args:
        - data_name (str): Name of the folder where data is stored.
        - split (float): Split ratio for train/validation.
        - logging (bool): Whether the training shall be logged in Tensorboard.
    Returns:
        None
    """
    # Create folders if they do not already exist
    create_folders('denta', data_name)

    # Preprocess the data
    time, data, states, train_names, state_names = preprocess_data(data_name, device='cpu')

    # Log the training
    if logging:
        log_dir = f'logs/denta/{data_name}'
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # Split into train and validation datasets
    train_time, valid_time, train_data, valid_data, train_states, valid_states = split_data(time, data, states, ratio)

    file_path = os.path.join(os.getcwd(), 'Hyperparameters/hyperparameters_denta.json')
    with open(file_path, 'r') as file:
        hyperparameters = json.load(file)

    # Train model
    model = denta.DENTA(train_data.shape[1], hyperparameters['dim3'], sigma=hyperparameters['sigma'],
                        sigma_learnable=False, device='cpu')
    model.learn(train_data, valid_data, max_epoch=hyperparameters['max_epoch'])
    latent_data = model.predict_discrete_mode(valid_data)

    purity = compute_purity(latent_data.numpy(), valid_states.numpy())
    print('Purity: ', purity)
    _, changes = extract_events(latent_data, valid_time, 'denta')
    print('Number of State Changes: ', len(changes))

    # Plot the data and the state labels and borders
    fig = plot_states(valid_data, valid_states, valid_time, state_names, latent_data, 'denta', data_name)

    if logging:
        writer.add_scalar("Purity", purity)
        writer.add_scalar('Number of State Changes: ', len(changes))
        writer.add_figure("Plots", fig, global_step=0)


def run_somvae(data_name, ratio, logging):
    """
    Run SOMVAE model on the provided data.

    Args:
        - data_name (str): Name of the folder where data is stored.
        - split (float): Split ratio for train/validation.
        - logging (bool): Whether the training shall be logged in Tensorboard.

    Returns:
        - None
    """
    # Create folders if they do not already exist
    create_folders('somvae', data_name)

    # Preprocess the data
    time, data, states, data_names, state_names = preprocess_data(data_name)

    # Set up Tensorboard
    if logging:
        log_dir = f'logs/somvae/{data_name}'
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # Split into train and validation datasets
    train_time, valid_time, train_data, valid_data, train_states, valid_states = split_data(time, data, states, ratio)

    # Train the SOM-VAE model
    model, latent_data = train_somvae(train_data, valid_data, data_name, writer)

    purity = compute_purity(latent_data.numpy(), valid_states.numpy())
    print('Purity: ', purity)
    _, changes = extract_events(latent_data, valid_time, 'somvae')
    print('Number of State Changes: ', len(changes))

    # Plot the data and the state labels and borders
    fig = plot_states(valid_data, valid_states, valid_time, state_names, latent_data, 'somvae', data_name)

    if logging:
        writer.add_scalar("Purity", purity)
        writer.add_scalar('Number of State Changes: ', len(changes))
        writer.add_figure("Plots", fig, global_step=0)


def run_catvae(data_name, ratio, logging):
    """
    Run CatVAE model on the provided data.

    Args:
        data_name (str): Name of the folder where data is stored.
        ratio (float): Split ratio for train/validation.
        logging (bool): Whether the training shall be logged in Tensorboard.

    Returns:
        None
    """
    # Create folders if they do not already exist
    create_folders('catvae', data_name)

    # Preprocess the data
    time, data, states, data_names, state_names = preprocess_data(data_name)

    if logging:
        log_dir = f'logs/catvae/{data_name}'
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # Split into train and validation datasets
    train_time, valid_time, train_data, valid_data, train_states, valid_states = split_data(time, data, states, ratio)

    # Train CAT-VAE model
    latent_data = exec_catvae(data_name, train_data, valid_data, writer)

    purity = compute_purity(latent_data.numpy(), valid_states.numpy())
    print('Purity: ', purity)
    _, changes = extract_events(latent_data, valid_time, 'catvae')
    print('Number of State Changes: ', len(changes))

    # Plot the data and the state labels and borders
    fig = plot_states(valid_data, valid_states, valid_time, state_names, latent_data, 'catvae', data_name)

    if logging:
        writer.add_scalar("Purity", purity)
        writer.add_scalar('Number of State Changes: ', len(changes))
        writer.add_figure("Plots", fig, global_step=0)


def main_anomdetect(data_name, split, logging, anomalies, threshold):
    selected_option = None
    while selected_option is None:
        # Display the options for the user to choose from
        print('Select an option:')
        print('1. DENTA')
        print('2. SOMVAE')
        print('3. CatVAE')

        # Prompt the user to enter their choice (1, 2 or 3)
        choice = input(' Please enter the number of the desired option (1/2/3): ')

        if choice == '1':
            selected_option = 'denta'               # User chose DENTA
            break
        elif choice == '2':
            selected_option = 'somvae'              # User chose SOMVAE
            break
        elif choice == '3':
            selected_option = 'catvae'              # User chose VAE
            break
        else:
            print('Invalid input. Please choose 1, 2, or 3.')

    if selected_option == 'denta':
        # Compute composite f1 score
        composite_f1 = run_ad_rbm(data_name, split, logging, anomalies, threshold)
        print(composite_f1)

    elif selected_option == 'somvae':
        # Compute composite f1 score
        composite_f1 = run_ad_somvae(data_name, split, logging, anomalies, threshold)
        print(composite_f1)

    elif selected_option == 'catvae':
        # Compute composite f1 score
        composite_f1 = run_ad_catvae(data_name, split, logging, anomalies, threshold)
        print(composite_f1)

    else:
        raise ValueError('Invalid option. This should not happen though :)')


def run_ad_rbm(data_name, split, logging, anomalies, threshold):
    fscore_arr = []
    # loop over anmomaly file names
    for anom_name in anomalies:
        # return of training data, idx of fault injection, indices of the complete anomalous time
        data, state_list, start_event_idx, true_anom_idx = preprocess_data_anom(anom_name)
        # check out whether your mehod is detecting an anomaly

        # anom_labels = np.where(np.array(some_mse_from_someone) < threshold, .5 , 0)
        # calculation of composite f1 score
        # comp_f1_score = composite_f1_score(anom_labels=anom_labels, start_event_idx=start_event_idx,
        # true_anom_idx=true_anom_idx)
        # append to array and return as mean over all detected anomalies
        # fscore_arr.append(comp_f1_score)
    return np.mean(fscore_arr)


def run_ad_somvae(data_name, split, logging, anomalies, threshold):
    fscore_arr = []
    model_path = os.path.expanduser(f'~/{str(Path.cwd().relative_to(Path.home()))}/'
                                           f'Models/simu_Tank/somvae_7/_model_2')
    model = instantiate_somvae()
    model.load_state_dict(torch.load(model_path).state_dict())
    time, train, states, train_names, state_names = preprocess_data('simu_tank')
    mse_nominal = anomaly_detection(train, model)

    threshold = np.mean(mse_nominal) + 0.1 * np.std(mse_nominal)
    # Loop over anmomaly file names
    for anom_name in anomalies:
        # return of training data, idx of fault injection, indices of the complete anomalous time
        data, state_list, start_event_idx, true_anom_idx = preprocess_data_anom(anom_name)
        mse_error = anomaly_detection(data, model)

        # Get labels
        anom_labels = np.where(np.array(mse_error) > threshold, .5 , 0)
        # Calculation of composite f1 score
        comp_f1_score = composite_f1_score(anom_labels=anom_labels, start_event_idx=start_event_idx, true_anom_idx=true_anom_idx)
        # Append to array and return as mean over all detected anomalies
        fscore_arr.append(comp_f1_score)

    return np.mean(fscore_arr)


def run_ad_catvae(data_name, split, logging, anomalies, threshold):
    fscore_arr = []
    model_path = os.path.expanduser(f'~/{str(Path.cwd().relative_to(Path.home()))}/'
                                    f'Models/simu_Tank/catvae/model')
    model = instantiate_catvae()
    model.load_state_dict(torch.load(model_path).state_dict())

    # Loop over anmomaly file names
    for anom_name in anomalies:
        # Return of training data, idx of fault injection, indices of the complete anomalous time
        data, state_list, start_event_idx, true_anom_idx = preprocess_data_anom(anom_name)
        # Check out whether your mehod is detecting an anomaly
        _, _, _, likelihood, mse_error = compute_mse_error(model, train=data)

        # Something above threshold ? Anomaly indicated with 0.5 in array and otherwise 0
        anom_labels = np.where(np.array(likelihood) < threshold, .5, 0)
        # Calculation of composite f1 score
        comp_f1_score = composite_f1_score(anom_labels=anom_labels, start_event_idx=start_event_idx, true_anom_idx=true_anom_idx)
        # Append to array and return as mean over all detected anomalies
        fscore_arr.append(comp_f1_score)
    return np.mean(fscore_arr)


if __name__ == '__main__':
    # List of datasets that should be involved within anomaly detection
    anomaly_scearios_tank = None

    anomaly_scenarios_tank = ['q1short1s', 'v12short1s', 'v23short1s', 'v3short1s',
                              'q1_50s', 'v12_50s', 'v23_50s', 'v3_50s',
                              'rest_100s', 'q1v3_100s', 'v12v23_100s',
                              'v12v3_100s', 'q1v23_100s']
    # anomaly_scearios_tank = None
    # Define some threshold to detect an anomaly
    anom_threshold = -200
    anom_detection = False

    if not anom_detection:
        main(split=0.75, data_name='Tank', logging=False)
        main(split=0.75, data_name='Siemens', logging=False)
        main(split=0.75, data_name='simu_tank', logging=False)  # the Steude et. al dataset

    else:
        # Perform Anomaly Detection and return composite f1 score over defined anomalies
        main_anomdetect(data_name='simu_tank', split=0.75, logging=True,
                        anomalies=anomaly_scenarios_tank, threshold=anom_threshold)
