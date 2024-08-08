import os
import torch
import random
import numpy as np
import pandas as pd
import _pickle as pkl

from tqdm import tqdm
from pathlib import Path
from automaton_learning import extract_events
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns.fpgrowth import fpgrowth
from sklearn.metrics import precision_score

random.seed(42)


def preprocess_data(file_path, device='cpu'):
    """
    Preprocess the data loaded using the provided file path.

    Args:
        - file_path (str): The file path to the folder where the data is stored.

    Returns:
        - time (PyTorch tensor): A list of the time sequences of the data.
        - train_list (PyTorch tensor): A list of the sequences of the data.
        - state_list (PyTorch tensor): A list of the state sequences of the data.
        - train_names (list): A list of the names of the data variables (except time and states).
        - state_names (list): A list of the names of the states.
    """
    # Load data from file
    raw_data = load_data(file_path)

    if 'Tank' in file_path:
        # Extract time from the data
        time = torch.tensor(raw_data['time'].values, dtype=torch.float, device=device)

        # Rename column opening to avoid issues
        new_column_names = [col.replace('opening', 'alter') for col in raw_data.columns]
        raw_data.columns = new_column_names
        # Get all the data except the time and state information and convert it to PyTorch tensors
        data_columns = ~raw_data.columns.str.contains('condition|open|time|Unnamed')
        data_columns = raw_data.loc[:, data_columns]
        data_values = torch.tensor(data_columns.values, dtype=torch.float, device=device)

        # Get all the state information from the dataset and convert it to PyTorch tensor
        state_columns = raw_data.columns[raw_data.columns.str.contains('open|condition')]
        state_columns = raw_data.loc[:, state_columns]
        signal_values = torch.tensor(state_columns.values, dtype=torch.float)
        powers_of_two = 2**torch.arange(signal_values.size(1) - 1, -1, -1).float()
        categories = torch.matmul(signal_values, powers_of_two)
        _, state_values = torch.unique(categories, return_inverse=True)

        # Names of the data signals
        data_names = data_columns.columns.tolist()
        # Name of the state signals
        state_names = state_columns.columns.tolist()

        # Normalize the data
        mean = torch.mean(data_values, dim=0)
        std = torch.std(data_values, dim=0)
        std[std == 0] = 1e-8
        data_values = (data_values - mean) / std

        print('Data preprocessing done!')

        return time, data_values, state_values, data_names, state_names

    elif 'Siemens' in file_path:

        time = torch.tensor(raw_data.index, dtype=torch.float, device=device)
        data_values = torch.tensor(raw_data.drop(columns=['CuStepNo ValueY']).to_numpy(),
                                   dtype=torch.float, device=device)
        state_values = torch.tensor(raw_data['CuStepNo ValueY'].reset_index(drop=True).to_numpy(),
                                    dtype=torch.float, device=device)

        # Normalize the data
        mean = torch.mean(data_values, dim=0)
        std = torch.std(data_values, dim=0)
        std[std == 0] = 1e-8
        data_values = (data_values - mean) / std

        # Names of the data signals
        data_names = (raw_data.columns.to_list())[1:]
        # Name of the state signals
        state_names = [(raw_data.columns.to_list())[0]]

        print('Data preprocessing done!')

        return time, data_values, state_values, data_names, state_names
    
    elif 'simu_tank' in file_path:

        time = torch.tensor(raw_data.index, dtype=torch.float, device=device)
        data_values = torch.tensor(raw_data.drop(columns=['cycle_name']).to_numpy(), dtype=torch.float, device=device)
        state_values = torch.tensor(preprocess_three_tank_states(raw_data['cycle_name']),
                                    dtype=torch.float, device=device)

        # Normalize the data
        mean = torch.mean(data_values, dim=0)
        std = torch.std(data_values, dim=0)
        std[std == 0] = 1e-8
        data_values = (data_values - mean) / std

        # Names of the data signals
        train_names = (raw_data.columns.to_list())[:3]
        # Name of the state signals
        state_names = [(raw_data.columns.to_list())[3]]

        print('Data preprocessing done!')

        return time, data_values, state_values, train_names, state_names

    else:
        raise Exception('Missing data!')


def load_data(keyword):
    """
    Load data from CSV files in a specified directory based on the provided keyword.

    Args:
        - keyword (str): A keyword to identify the directory containing CSV files.

    Returns:
        - dataframes (list): A list of Pandas DataFrames loaded from CSV files.
    """
    # Determine the path based on the keyword
    home_path = Path.home()
    cwd_relative_path = Path.cwd().relative_to(home_path)
    data_dir = f'{cwd_relative_path}/Data/{keyword}'

    if 'Siemens' in keyword:
        path = os.path.join(home_path, data_dir, 'id1_norm.csv')
        dataframes = pd.read_csv(path).reset_index(drop=True)
    elif 'simu_tank' in keyword:
        path = os.path.join(home_path, data_dir, 'norm.csv')
        dataframes = pd.read_csv(path).iloc[1000:].reset_index(drop=True)
    else:
        path = os.path.join(home_path, data_dir, 'ds3n0.csv')
        dataframes = pd.read_csv(path)

    return dataframes


def create_folders(model_name, data_name):
    """
    Create folders for a given model name if they do not already exist.
    Folder are created for the PyTorch models and the plots.

    Args:
        - model_name (str): The name of the model for which folders will be created.
        - data_name (str): The name of the data to be used in folder structure.

    Returns:
        - None
    """
    # Define base paths
    base_path = Path.home() / Path.cwd().relative_to(Path.home())
    model_path = base_path / 'Models' / data_name / model_name
    plots_path = base_path / 'Plots' / data_name / model_name

    # List of paths to create
    paths_to_create = [model_path, plots_path]

    # Create folders if they do not already exist
    for path in paths_to_create:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f'Folders for {model_name} were successfully created at {path}.')
        else:
            print(f'Folders already exist at {path}.')


def compute_purity(cluster_assignments, class_assignments):
    """Computes the purity between cluster and class assignments.
    Compare to https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    
    Args:
        cluster_assignments (list): List of cluster assignments for every point.
        class_assignments (list): List of class assignments for every point.

    Returns:
        float: The purity value.
    """

    assert len(cluster_assignments) == len(class_assignments)

    num_samples = len(cluster_assignments)
    num_clusters = len(np.unique(cluster_assignments))
    num_classes = len(np.unique(class_assignments))
    
    cluster_class_counts = {cluster_: {class_: 0 for class_ in np.unique(class_assignments)}
                            for cluster_ in np.unique(cluster_assignments)}

    for cluster_, class_ in zip(cluster_assignments, class_assignments):
        cluster_class_counts[cluster_][class_] += 1

    total_intersection = sum([max(list(class_dict.values())) for cluster_, class_dict in cluster_class_counts.items()])
    
    purity = total_intersection/num_samples

    return purity


def create_association_dict(disc_cats, real_cats):
    """
    Compute a dictionary of associated items from "real" expert defined states/categories
    to discretized states from ML models

    Parameters:
    - disc_cats (Dataframe): In this case the expert knowledge is given by strings
    - real cats (Dataframe): Dataframe of discretized data into categories given from the ML models

    Returns:
    - dict_states_char: Returns dictionary from real categories to its associated discretized categories from ML models
    - df_merge_fqitems_dummies: Returns one hot encoded dataframe of categories
    """

    df_merge_fqitems = pd.DataFrame()
    df_merge_fqitems['real'] = real_cats
    df_merge_fqitems['disc'] = disc_cats
    # one hot encode categories as suitable input for fpgrowth method
    df_merge_fqitems_dummies = pd.get_dummies(df_merge_fqitems, columns=['disc', 'real'])
    # Association learning with FPGrowth
    frequent_itemsets = fpgrowth(df_merge_fqitems_dummies, min_support=.00005, use_colnames=True)
    # Generate association rules from frequent itemsets
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.08)
    # Just use rules indicating from real to discretized values and not the other way round
    real_columns = [col for col in df_merge_fqitems_dummies.columns if 'real' in col]
    true_rules_bool = pd.DataFrame([rules[rules.columns[0]] == frozenset({f'{comp}'})
                                    for comp in real_columns]).T.any(axis=1)
    true_rules = rules[true_rules_bool.values]
    # create dictionary with the relation of which real category goes to which discretized category/categories
    states_disc = list(true_rules[true_rules.columns[0]].apply(lambda x: ', '.join(list(x))).astype("unicode"))
    res_disc = list(true_rules[true_rules.columns[1]].apply(lambda x: ', '.join(list(x))).astype("unicode"))
    df_disc = pd.DataFrame({"states_disc": states_disc, "res_disc": res_disc})
    states_dict = df_disc.groupby('states_disc')['res_disc'].apply(list).to_dict()
    for _ in list(states_dict.keys()):
        dict_states_char = {key: value for key, value in states_dict.items()}
    return dict_states_char, df_merge_fqitems_dummies


def compute_discretization_precision(dict_states_char, df_merge_fqitems_dummies):
    """
    Compute discretization precision between expert and model classifications.

    Parameters:
    - dict_states_char (dict): Dictionary from real categories to its associated discretized categories from ML models
    - df_merge_fqitems_dummies (Dataframe): Returns one hot encoded dataframe of categories

    Returns:
    - precision: Returns precision metric as the percentage of correct predictions over the total number of rows
    """

    correct_predictions = 0
    # Iterate over each row in the dataset
    for index, row in df_merge_fqitems_dummies.iterrows():
        # Check if the row matches all rules in the dictionary
        matches_all_rules = any(any(row[col] for col in disc_cols) for real_col, disc_cols in dict_states_char.items())
        # If the row matches all rules, increment the counter
        if matches_all_rules:
            correct_predictions += 1
    # Calculate precision as the percentage of correct predictions over the total number of rows
    precision = (correct_predictions / len(df_merge_fqitems_dummies)) * 100
    return precision


def split_data(time, data, states, ratio=0.8):

    split_index = int(time.shape[0] * ratio)

    train_time = time[:split_index]
    valid_time = time[split_index:]

    train_data = data[:split_index, :]
    valid_data = data[split_index:, :]

    train_states = states[:split_index]
    valid_states = states[split_index:]

    return train_time, valid_time, train_data, valid_data, train_states, valid_states


def preprocess_three_tank_states(states):

    states = [state for state in states]
    unique_states = np.unique(states)
    # Create a mapping of states to numbers
    state_to_number = {state: i for i, state in enumerate(unique_states)}
    # Replace each state with its corresponding number
    states = [state_to_number[state] for state in states]

    return states


def preprocess_data_anom(anom_name):
    path = os.path.expanduser(f'~/{str(Path.cwd().relative_to(Path.home()))}/Data/simu_tank/{anom_name}_faulty.csv')
    data = pd.read_csv(f'{path}').iloc[:,:].reset_index(drop=True)

    anom_data = data.drop(columns=['cycle_name'])
    anom_data = torch.tensor(anom_data.values, dtype=torch.float64)
    mean = torch.mean(anom_data, dim=0)
    std = torch.std(anom_data, dim=0)

    anom_data = (anom_data -mean) / std

    state_list = data['cycle_name']
    true_anom_idx = state_list.str.contains('faulty').astype(int).reset_index(drop=True)
    true_anom_idx.iloc[true_anom_idx.idxmax():] = 1
    start_event_idx = true_anom_idx.idxmax()

    print('Data preprocessing done!')

    return anom_data, state_list, start_event_idx, true_anom_idx

def composite_f1_score(anom_labels, start_event_idx, true_anom_idx):
    true_anomalies = np.array(true_anom_idx) != 0
    pred_anomalies = np.array(anom_labels) != 0
    # True Positives (TP): True anomalies correctly predicted as anomalies
    tp = np.sum([pred_anomalies[start_event_idx:].any()])
    # False Negatives (FN): True anomalies missed by the prediction
    fn = 1 - tp
    # Recall for events (Rec_e): Proportion of true anomalies correctly identified
    rec_e = tp / (tp + fn) if (tp + fn) > 0 else 0
    # Precision for the entire time series (Prec_t)
    prec_t = precision_score(true_anomalies, pred_anomalies)
    # Composite F-score
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    else:
        fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    return fscore_c