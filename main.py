import torch
import os
import json
import numpy as np
from RBM import train_rbm
import denta
from pathlib import Path
from plots import plot_states
from CatVAE import exec_catvae
from SOMVAE import train_somvae, anomaly_detection, instantiate_somvae, get_latents
from tensorboardX import SummaryWriter
from utils import (create_folders, preprocess_data, create_association_dict,
                   compute_discretization_precision, compute_purity, convert_data,
                   preprocess_data_anom, composite_f1_score, preprocess_three_tank_states)

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
        print('1. RBM')
        print('2. SOMVAE')
        print('3. CatVAE')
        print('4. DENTA')

        # Prompt the user to enter their choice (1, 2 or 3)
        choice = input(' Please enter the number of the desired option (1/2/3/4): ')
        if choice == '1':
            selected_option = 'denta'                    # User chose DENTA
            break
        elif choice == '2':
            selected_option = 'somvae'                  # User chose SOMVAE
            break
        elif choice == '3':
            selected_option = 'catvae'                  # User chose VAE
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


def run_rbm(data_name, split, logging):
    """
    Run RBM model on the provided data.

    Args:
        data_name (str): Name of the folder where data is stored.
        split (float): Split ratio for train/validation.
        logging (bool): Whether the training shall be logged in Tensorboard.

    Returns:
        None
    """
    # Create folders if they do not already exist
    create_folders('rbm', data_name)

    # Preprocess the data
    time, data, states, train_names, state_names = preprocess_data(data_name)

    if 'simu_tank' in data_name:
        states = preprocess_three_tank_states(states)

    purities = []
    # Iterate several times to compute mean and variance
    for i in range(1):

        # Log the training
        if logging:
            log_dir = f'logs/rbm/{data_name}_{str(i)}'
            writer = SummaryWriter(log_dir)
        else:
            writer = None

        # Split into train and test datasets
        temp_train_data = data[:int(split * len(data))]
        temp_valid_data = data[int(split * len(data)):]
        temp_valid_time = time[int(split * len(time)):]
        valid_states = states[int(split * len(states)):]

        # Preprocess the data again (for Siemens and Three-Tank dataset)
        train_data, valid_data, valid_time = convert_data(temp_train_data, temp_valid_data, temp_valid_time, data_name)

        # Train model
        model, latent_data = train_rbm(train_data, valid_data, writer)
        amount_states = len(torch.unique(latent_data[0]))
        writer.add_scalar("Amount of states", amount_states, i)

        # Save the different model level of RBM
        for x, rbm in enumerate(model):
            torch.save(rbm, f'Models/rbm/{data_name}/rbm{x}_{str(i)}')

        # Compute the purity value
        latent_states = torch.cat(latent_data, dim=0)
        if 'simu_tank' not in data_name and 'siemens' not in data_name:
            temp_valid_states = torch.cat(valid_states, dim=0)
            purity = compute_purity(latent_states.numpy(), temp_valid_states.numpy())
        elif 'simu_tank' not in data_name:
            purity = compute_purity(latent_states.numpy(), valid_states.numpy())
        else:
            purity = compute_purity(latent_states.numpy(), valid_states)
        # Save the purity value in Tensorboard
        if writer:
            writer.add_scalar("Purity", purity, i)
        purities.append(purity)

        if 'labelled' in data_name or 'Artificial' in data_name:
            valid_states = valid_states[0]
            latent_data = latent_data[0]
            valid_time = valid_time[0]
            valid_data = valid_data[0]
        else:
            latent_data = torch.cat(latent_data)[-1000:]
            valid_time = torch.cat(valid_time)[-1000:]
            valid_data = torch.cat(valid_data)[-1000:]
            valid_states = torch.tensor(valid_states[-1000:])

        # Plot the data and the state labels and borders
        fig = plot_states(valid_data, valid_states, valid_time, state_names, latent_data, i,
                          'rbm', data_name)
        writer.add_figure("Plots", fig, global_step=0)
        writer.close()

    # Compute mean and variance of the purity values
    mean_value = np.mean(purities)
    variance_value = np.var(purities)

    print('Purity mean: ', mean_value, 'Purity variance: ', variance_value)
    print(f'Total of {i + 1} iterations')


def run_denta(data_name, split, logging, num_repetations=10):
    """
    Run RBM model on the provided data.

    Args:
        data_name (str): Name of the folder where data is stored.
        split (float): Split ratio for train/validation.
        logging (bool): Whether the training shall be logged in Tensorboard.
        :param num_repetations:
    Returns:
        None

    """
    create_folders('denta', data_name)  # Create folders if they do not already exist

    # Preprocess the data
    time, data, states, train_names, state_names = preprocess_data(data_name, device='cuda')
    purities = []
    # Iterate over different hyperparameter sets (optional)
    for rep_ind in range(num_repetations):
        # Log the training
        if logging:
            log_dir = f'logs/denta/{data_name}_{rep_ind}'
            writer = SummaryWriter(log_dir)
        else:
            writer = None

        # Split into train and test datasets
        train_data = data[:int(split * len(data))]
        valid_data = data[int(split * len(data)):]
        valid_time = time[int(split * len(time)):]
        valid_states = states[int(split * len(states)):]

        file_path = os.path.join(os.getcwd(), 'Hyperparameters/hyperparameters_denta.json')
        with open(file_path, 'r') as file:
            hyperparameters = json.load(file)

        if rep_ind is not None:
            dim3 = hyperparameters.get(f'dim3_{rep_ind + 1}', None)
        else:
            dim3 = None
        if dim3 is None:
            dim3 = hyperparameters['dim3']

        # Train model
        model = denta.DENTA(train_data[0].shape[1], dim3, sigma=hyperparameters['sigma'], sigma_learnable=False,
                            device='cuda')
        model.learn(torch.vstack(train_data), torch.vstack(valid_data), max_epoch=hyperparameters['max_epoch'])

        latent_data = model.predict_discrete_mode(valid_data)

        # Save the different model level of RBM
        # for x, rbm in enumerate(model):
        #     torch.save(rbm, f'Models/rbm/rbm{x}_{str(i)}')

        latent_ints = torch.cat(latent_data, dim=0)
        valid_states = torch.cat(valid_states, dim=0)
        # valid_ints = torch.tensor([int(''.join(str(int(bit)) for bit in row), 2) for row in valid_states], dtype=torch.int64)

        dict_states, df_fqitems = create_association_dict(latent_ints.numpy(), valid_states.numpy())
        # precision = compute_discretization_precision(dict_states, df_fqitems)
        purity = compute_purity(latent_ints.numpy(), valid_states.numpy())

        # Cause don't now at the moment where to store, let's print it
        # print(precision)
        purities.append(purity)
        print(f"Purity: {purity}")
        # Compute similarity score
        # compute_similarity('rbm', latent_data, valid_data, valid_states, valid_time, writer, data_name)
        latent_data = latent_ints.view(len(valid_time), -1)
        valid_ints = valid_states.view(len(valid_time), -1)

        # Plot the data and the state labels and borders
        fig = plot_states(1, valid_data, valid_ints, valid_time, train_names, state_names, latent_data,
                          rep_ind + 1, 'denta', data_name)
        writer.add_figure("Discretization", fig, global_step=0)

    mean_value = np.mean(purities)
    variance_value = np.var(purities)
    print('Purities: ', str(purities))
    print('Purity mean: ', mean_value, 'Purity variance: ', variance_value)
    print(f'Total of {rep_ind + 1} iterations')


def run_somvae(data_name, split, logging):
    """
    Run SOMVAE model on the provided data.

    Args:
        data_name (str): Name of the folder where data is stored.
        split (float): Split ratio for train/validation.
        logging (bool): Whether the training shall be logged in Tensorboard.

    Returns:
        None
    """
    # Create folders if they do not already exist
    create_folders('somvae', data_name)

    # Preprocess the data
    time, data, states, train_names, state_names = preprocess_data(data_name)

    if 'simu_tank' in data_name:
        states = preprocess_three_tank_states(states)

    # Anomalous data
    # _, anom_data, _, _, _ = preprocess_data("Tank/Anomalies_2")

    purities = []
    # Iterate over different hyperparameter sets (optional)
    for i in range(1):
        # Set up Tensorboard
        if logging:
            log_dir = f'logs/somvae/{data_name}_{str(i)}'
            writer = SummaryWriter(log_dir)
        else:
            writer = None

        # Split into train and test datasets
        temp_train_data = data[:int(split * len(data))]
        temp_valid_data = data[int(split * len(data)):]
        temp_valid_time = time[int(split * len(time)):]
        valid_states = states[int(split * len(states)):]

        # Preprocess the data again (for Siemens and Three-Tank dataset)
        train_data, valid_data, valid_time = convert_data(temp_train_data, temp_valid_data, temp_valid_time, data_name)

        # Train the SOM-VAE model
        model, latent_data = train_somvae(train_data, valid_data, writer)

        model_path = os.path.expanduser(f'~/{str(Path.cwd().relative_to(Path.home()))}/'
                                        f'Models/Tank/Berfipl_long/somvae_8/_model_2')
        #model = instantiate_somvae()
        #model.load_state_dict(torch.load(model_path).state_dict())
        #latent_data = get_latents(model, valid_data)

        # amount_states = len(torch.unique(latent_data[0]))
        #writer.add_scalar("Amount of states", amount_states, i)

        """
        pred_losses = anomaly_detection(train_data, model)
        threshold = np.mean(pred_losses) + 2 * np.std(pred_losses)
        anom_losses = anomaly_detection(anom_data, model, threshold)
        losses = anom_losses > threshold

        print(pred_losses)
        print(anom_losses)
        print(threshold)
        print(losses)
        """

        # Save the learned model
        # torch.save(model, f'Models/somvae/{data_name}/model_{str(i)}')

        # Compute the purity value
        latent_states = latent_data[0]

        # latent_states = torch.cat(latent_data, dim=0)
        if 'simu_tank' not in data_name and 'siemens' not in data_name and 'long' not in data_name:
            temp_valid_states = torch.cat(valid_states, dim=0)
            purity = compute_purity(latent_states.numpy(), temp_valid_states.numpy())
        elif 'simu_tank' not in data_name:
            purity = compute_purity(latent_states.numpy(), valid_states.numpy())
        else:
            purity = compute_purity(latent_states.numpy(), valid_states)
        # Save the purity value in Tensorboard
        #if writer:
        #    writer.add_scalar("Purity", purity, i)
        purities.append(purity)

        print(purity)

        if 'labelled' in data_name or 'Artificial' in data_name:
            valid_states = valid_states[0]
            latent_data = latent_data[0]
            valid_time = valid_time[0]
            valid_data = valid_data[0]
        else:
            latent_data = torch.cat(latent_data)[-1000:]
            valid_time = torch.cat(valid_time)[-1000:]
            valid_data = torch.cat(valid_data)[-1000:]
            valid_states = torch.tensor(valid_states[-1000:])

        #print(valid_states)

        unique_states = torch.unique(valid_states).tolist()
        unique_states.remove(9)
        unique_states.remove(11)
        new = []
        for state in valid_states:
            # state = state.item()
            if state == 9:
                state = 1
            elif state == 11:
                state = 10
            new.append(unique_states.index(state))

        new = torch.tensor(new)
        #print(new)

        # Plot the data and the state labels and borders
        #fig = plot_states(valid_data, valid_states, valid_time, state_names, latent_data, i,
        #                  'somvae', data_name)
        # fig = plot_states2(valid_data, new, valid_time, state_names, latent_data, i,
        #                  'somvae', data_name, 1)
        #writer.add_figure("Plots", fig, global_step=0)

    # Compute mean and variance of the purity values
    mean_value = np.mean(purities)
    variance_value = np.var(purities)

    print('Purity mean: ', mean_value, 'Purity variance: ', variance_value)
    print(f'Total of {i + 1} iterations')


def run_catvae(data_name, split, logging):
    """
    Run CatVAE model on the provided data.

    Args:
        data_name (str): Name of the folder where data is stored.
        split (float): Split ratio for train/validation.
        logging (bool): Whether the training shall be logged in Tensorboard.

    Returns:
        None
    """
    create_folders('catvae', data_name)  # Create folders if they do not already exist

    # Preprocess the data
    time, train, states, train_names, state_names = preprocess_data(data_name)

    # Iterate over different hyperparameter sets (optional)
    for i in range(1):
        if logging:
            log_dir = f'logs/catvae/{data_name}_{str(i)}'
            writer = SummaryWriter(log_dir)
        else:
            writer = None

        # Train CAT-VAE model
        latent_data, valid_data, unique_list, anom_label_likelihood, anom_label_mse = exec_catvae(data_name=data_name, train=train,
                                                           selected_option='catvae')

        valid_time = time[int(split * len(train)):]
        valid_states = states[int(split * len(train)):]

        # Compute similarity score
        if 'siemens' in data_name:
            valid_states = valid_states
            latent_states = latent_data
        else: 
            valid_states = torch.cat(valid_states, dim=0)
            latent_states = torch.cat(latent_data, dim=0).cpu()

        # powers_of_two_valid = 2 ** torch.arange(valid_states.size(-1) - 1, -1, -1, dtype=torch.float)
        # powers_of_two_latent = 2 ** torch.arange(latent_states.size(-1) - 1, -1, -1, dtype=torch.float)
        # valid_ints = torch.matmul(valid_states.float(), powers_of_two_valid)
        # latent_ints = torch.matmul(latent_states.float(), powers_of_two_latent)

        # valid_ints_ = torch.tensor([int(''.join(str(int(bit)) for bit in row), 2) for row in valid_states], dtype=torch.int64)
        # latent_ints_ = torch.tensor([int(''.join(str(int(bit)) for bit in row), 2) for row in latent_states], dtype=torch.int64)

        dict_states, df_fqitems = create_association_dict(latent_states, valid_states)
        precision = compute_discretization_precision(dict_states, df_fqitems)
        purity = compute_purity(latent_states.numpy(), valid_states.numpy())
        # Cause don't now at the moment where to store, let's print it
        print(purity)

        if 'Artificial' in data_name:
            valid_states = valid_states[0]
            latent_data = latent_data[0]
            valid_time = valid_time[0]
            valid_data = valid_data[0]
        elif 'labelled' in data_name:
            latent_data = torch.cat(latent_data)[-256:]
            valid_time = torch.cat(valid_time)[-256:]
            valid_data = valid_data.view(-1, valid_data.size(-1))[-256:]
            valid_states = torch.tensor(valid_states[-256:])
        else:
            latent_data = latent_data[-1000:]
            valid_time = valid_time[-1000:]
            valid_data = valid_data.view(-1, valid_data.size(-1))[-1000:]
            valid_states = torch.tensor(valid_states[-1000:])

        # Plot the data and the state labels and borders
        fig = plot_states(valid_data, valid_states, valid_time, state_names, latent_data, i,
                          'catvae', data_name)
        writer.add_figure("Discretization", fig, global_step=0)

def main_anomdetect(data_name, split, logging, anomalies, threshold):
    selected_option = None
    while selected_option is None:
        # Display the options for the user to choose from
        print('Select an option:')
        print('1. RBM')
        print('2. SOMVAE')
        print('3. CatVAE')

        # Prompt the user to enter their choice (1, 2 or 3)
        choice = input(' Please enter the number of the desired option (1/2/3): ')

        if choice == '1':
            selected_option = 'rbm'                 # User chose RBM
            break
        elif choice == '2':
            selected_option = 'somvae'              # User chose SOMVAE
            break
        elif choice == '3':
            selected_option = 'catvae'              # User chose VAE
            break
        else:
            print('Invalid input. Please choose 1, 2, or 3.')

    if selected_option == 'rbm':
        # please implement your stuff in the according function
        composite_f1 = run_ad_rbm(data_name, split, logging, anomalies, threshold)
        print(composite_f1)

    elif selected_option == 'somvae':
        # please implement your stuff in the according function
        composite_f1 = run_ad_somvae(data_name, split, logging, anomalies, threshold)
        print(composite_f1)

    elif selected_option == 'catvae':
        # please implement your stuff in the according function
        composite_f1 = run_ad_catvae(data_name, split, logging, anomalies, threshold)
        print(composite_f1)

    else:
        raise ValueError('Invalid option. This should not happen though :)')

# TODO: Implement your Anomaly Detection please
def run_ad_rbm(data_name, split, logging, anomalies, threshold):
    fscore_arr = []
    # loop over anmomaly file names
    for anom_name in anomalies:
        # return of training data, idx of fault injection, indices of the complete anomalous time
        data, state_list, start_event_idx, true_anom_idx = preprocess_data_anom(anom_name)
        # check out whether your mehod is detecting an anomaly
        # TODO: HERE COMES YOUR METHOD FOR ANOMALY DETECTION!!


        # something above threshold ? Anomaly indicated with 0.5 in array and otherwise 0
        # anom_labels = np.where(np.array(some_mse_from_someone) < threshold, .5 , 0)
        # calculation of composite f1 score
        # comp_f1_score = composite_f1_score(anom_labels=anom_labels, start_event_idx=start_event_idx, true_anom_idx=true_anom_idx)
        # append to array and return as mean over all detected anomalies
        # fscore_arr.append(comp_f1_score)
    return np.mean(fscore_arr)

# TODO: Implement your Anomaly Detection please
def run_ad_somvae(data_name, split, logging, anomalies, threshold):
    fscore_arr = []
    model_path = os.path.expanduser(f'~/{str(Path.cwd().relative_to(Path.home()))}/'
                                           f'Models/simu_Tank/somvae_7/_model_2')
    model = instantiate_somvae()
    model.load_state_dict(torch.load(model_path).state_dict())
    time, train, states, train_names, state_names = preprocess_data('simu_tank')
    mse_nominal = anomaly_detection(train, model)

    threshold = np.mean(mse_nominal) + 0.1 * np.std(mse_nominal)
    # loop over anmomaly file names
    for anom_name in anomalies:
        # return of training data, idx of fault injection, indices of the complete anomalous time
        data, state_list, start_event_idx, true_anom_idx = preprocess_data_anom(anom_name)
        #data = torch.tensor(data.values, dtype=torch.float64)

        mse_error = anomaly_detection(data, model)
        # check out whether your mehod is detecting an anomaly
        # TODO: HERE COMES YOUR METHOD FOR ANOMALY DETECTION!!


        # something above threshold ? Anomaly indicated with 0.5 in array and otherwise 0
        anom_labels = np.where(np.array(mse_error) > threshold, .5 , 0)
        # calculation of composite f1 score
        comp_f1_score = composite_f1_score(anom_labels=anom_labels, start_event_idx=start_event_idx, true_anom_idx=true_anom_idx)
        # append to array and return as mean over all detected anomalies
        fscore_arr.append(comp_f1_score)
        print(comp_f1_score)
    return np.mean(fscore_arr)


def run_ad_catvae(data_name, split, logging, anomalies, threshold):
    fscore_arr = []
    # Loop over anmomaly file names
    for anom_name in anomalies:
        # Return of training data, idx of fault injection, indices of the complete anomalous time
        data, state_list, start_event_idx, true_anom_idx = preprocess_data_anom(anom_name)
        # Check out whether your mehod is detecting an anomaly
        _, _, _, likelihood, mse_error = exec_catvae(data_name=data_name, train=data,
                                                     selected_option='catvae')

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
    a = False

    if not a:
        # main(split=0.75, data_name='Tank/Berfipl_long', logging=True)
        # main(split=0.75, data_name='Artificial/Dn1', logging=True)
        main(split=0.75, data_name='siemens', logging=True)
        main(split=0.75, data_name='simu_tank', logging=True)  # the Steude et. al dataset

    else:
        # Perform Anomaly Detection and return composite f1 score over defined anomalies
        main_anomdetect(data_name='simu_tank', split=0.75, logging=True,
                        anomalies=anomaly_scenarios_tank, threshold=anom_threshold)
