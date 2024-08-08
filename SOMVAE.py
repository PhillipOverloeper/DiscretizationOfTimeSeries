import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


class SOMVAE(nn.Module):
    def __init__(self, latent_dim=8, som_dim=None, input_length=16,
                 hidden_dim=10, alpha=1., beta=1., gamma=1., tau=1.):
        """
        Initialize a Self-Organizing Map Variational Autoencoder (SOM-VAE) model.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            som_dim (list): Dimensions of the self-organizing map grid [rows, columns].
            input_length (int): Length of the input data.
            alpha (float): Weight parameter for the SOM loss.
            beta (float): Weight parameter for the reconstruction loss.
            gamma (float): Weight parameter for the commitment loss.
            tau (float): Weight parameter for the topographic loss.
        """

        super(SOMVAE, self).__init__()

        # Store hyperparameters
        if som_dim is None:
            self.som_dim = [4, 4]
        else:
            self.som_dim = som_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.som_dim = som_dim
        self.input_length = input_length
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

        # Initialize SOM embeddings
        self.embeddings = nn.Parameter(nn.init.trunc_normal_(torch.empty((*self.som_dim, self.latent_dim)),
                                                             std=0.05, a=-0.1, b=0.1))

        # Initialize SOM probabilities
        probs_raw = torch.zeros(*(som_dim + som_dim))
        probs_pos = torch.exp(probs_raw)
        probs_sum = torch.sum(probs_pos, dim=[-1, -2], keepdim=True)
        self.probs = nn.Parameter(probs_pos / probs_sum)

        # Define the encoder network
        self.encoder = nn.Sequential(
            nn.Linear(self.input_length, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.ReLU()
        )

        # Define the q-decoder network
        self.q_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_length),
            nn.Sigmoid()
        )

        # Define the e-decoder network
        self.e_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_length),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the SOM-VAE model.

        Args:
            x (Tensor): Input data.

        Returns:
            x_hat_q (Tensor): Reconstructed input using the q-decoder.
            x_hat_e (Tensor): Reconstructed input using the e-decoder.
            z_e (Tensor): Latent representation obtained from the encoder.
            z_q (Tensor): Latent representation obtained from the q-decoder.
            k (Tensor): SOM cluster assignments.
            z_dist_flat (Tensor): Flattened distribution of latent vectors.
            z_q_neighbors (Tensor): Latent representations of SOM neighbors.
        """
        # Compute z_e, z_q, z_q_neighbors, and other necessary components
        z_e = self.encoder(x)
        k = self.k(z_e)                                     # Compute SOM cluster assignments
        z_dist_flat = self.z_dist_flat(z_e)                 # Flatten distribution of latent vectors
        z_q = self.z_q_calc(k)                              # Compute latent representation from the q-decoder
        z_q_neighbors = self.z_q_neighbors(k, x, z_q)       # Compute latent representations of SOM neighbors

        # Compute reconstructions
        x_hat_q = self.q_decoder(z_q)                       # Reconstructed input using the q-decoder
        x_hat_e = self.e_decoder(z_e)                       # Reconstructed input using the e-decoder

        return x_hat_q, x_hat_e, z_e, z_q, k, z_dist_flat, z_q_neighbors

    def z_q_calc(self, k):
        """
        Find embeddings for each k.

        Parameters:
            k (torch.Tensor): Tensor of the positions of the embeddings.

        Returns:
            z_q (torch.Tensor): Embeddings for each k.
        """

        # Split k into row and column components
        k_1 = k // self.som_dim[1]
        k_2 = k % self.som_dim[1]

        # Stack row and column components
        k_stacked = torch.stack([k_1, k_2], dim=1)

        # Retrieve embeddings for each k
        z_q = self._gather_nd(self.embeddings, k_stacked)

        return z_q

    def z_q_neighbors(self, k, x, z_q):
        """
        Find the neighbors of each embedding.

        Parameters:
            k (torch.Tensor): Tensor of the position of the embeddings.
            x (torch.Tensor): The data.
            z_q (torch.Tensor): The embeddings.

        Returns:
            z_q_neighbors (torch.Tensor): The neighbors of each z_q.
        """
        # Split k into row and column components
        k_1 = k // self.som_dim[1]
        k_2 = k % self.som_dim[1]

        # Check neighbors in different directions
        k1_not_top = k_1 < self.som_dim[0] - 1
        k1_not_bottom = k_1 > 0
        k2_not_right = k_2 < self.som_dim[1] - 1
        k2_not_left = k_2 > 0

        # Determine neighbors based on position
        k1_up = torch.where(k1_not_top, k_1 + 1, k_1)
        k1_down = torch.where(k1_not_bottom, k_1 - 1, k_1)
        k2_right = torch.where(k2_not_right, k_2 + 1, k_2)
        k2_left = torch.where(k2_not_left, k_2 - 1, k_2)

        # Initialize neighbor embeddings
        z_q_up = torch.zeros(x.shape[0], self.latent_dim)
        z_q_down = torch.zeros(x.shape[0], self.latent_dim)
        z_q_right = torch.zeros(x.shape[0], self.latent_dim)
        z_q_left = torch.zeros(x.shape[0], self.latent_dim)

        # Gather neighbor embeddings
        z_q_up_ = self._gather_nd(self.embeddings, torch.stack([k1_up, k_2], dim=1))
        z_q_up[k1_not_top == 1] = z_q_up_[k1_not_top == 1]

        z_q_down_ = self._gather_nd(self.embeddings, torch.stack([k1_down, k_2], dim=1))
        z_q_down[k1_not_bottom == 1] = z_q_down_[k1_not_bottom == 1]

        z_q_right_ = self._gather_nd(self.embeddings, torch.stack([k_1, k2_right], dim=1))
        z_q_right[k2_not_right == 1] = z_q_right_[k2_not_right == 1]

        z_q_left_ = self._gather_nd(self.embeddings, torch.stack([k_1, k2_left], dim=1))
        z_q_left[k2_not_left == 1] = z_q_left_[k2_not_left == 1]

        # Stack neighbor embeddings
        z_q_neighbors = torch.stack([z_q, z_q_up, z_q_down, z_q_right, z_q_left], dim=1)

        return z_q_neighbors

    def z_dist_flat(self, z_e):
        """
        Computes the squared Euclidean distances between the encodings and the SOM embeddings.

        Args:
            z_e (Tensor): Latent representations from the encoder.

        Returns:
            z_dist_flat (Tensor): Flattened squared distances between z_e and SOM embeddings.
        """
        # Expand dimensions for broadcasting
        z_e_expanded = z_e.unsqueeze(1).unsqueeze(1)
        embeddings_expanded = self.embeddings.unsqueeze(0)

        # Compute squared Euclidean distances
        z_dist = torch.pow(z_e_expanded - embeddings_expanded, 2)

        # Reduce the distances by summing along the last dimension
        z_dist_red = torch.sum(z_dist, dim=-1)

        # Flatten the distance tensor
        z_dist_flat = z_dist_red.view(z_e.shape[0], -1)

        return z_dist_flat

    def k(self, z_e):
        """
        Picks the index of the closest embedding for every encoding.

        Args:
            z_e (Tensor): Latent representations from the encoder.

        Returns:
            k (Tensor): Indices of the closest SOM embeddings for each encoding.
        """
        # Calculate the squared distances between z_e and SOM embeddings
        z_dist_flat = self.z_dist_flat(z_e)

        # Find the index of the closest embedding for each encoding
        k = torch.argmin(z_dist_flat, dim=-1)

        return k

    def loss(self, x, x_hat_q, x_hat_e, z_q, z_e, z_q_neighbors, k, z_dist_flat):
        """
        Computes the overall loss for the SOM-VAE model.

        Args:
            x (Tensor): Input data.
            x_hat_q (Tensor): Reconstructed input using the q-decoder.
            x_hat_e (Tensor): Reconstructed input using the e-decoder.
            z_q (Tensor): Latent representation obtained from the q-decoder.
            z_e (Tensor): Latent representation obtained from the encoder.
            z_q_neighbors (Tensor): Latent representations of SOM neighbors.
            k (Tensor): SOM cluster assignments.
            z_dist_flat (Tensor): Flattened squared distances between z_e and SOM embeddings.

        Returns:
            total_loss (Tensor): Overall loss for the SOM-VAE model.
        """
        # Reconstruction loss for both q-decoder and e-decoder
        loss_rec_mse_zq = F.mse_loss(x_hat_q, x)
        loss_rec_mse_ze = F.mse_loss(x_hat_e, x)
        loss_rec_mse = loss_rec_mse_zq + loss_rec_mse_ze

        # Commitment loss
        loss_commit = torch.mean(torch.pow(z_e - z_q, 2))

        # Topographic loss
        loss_som = torch.mean(torch.pow(z_e.unsqueeze(1) - z_q_neighbors, 2))

        # Loss related to SOM probabilities
        prob_l = self.loss_prob(k)
        prob_z_l = self._loss_z_prob(k, z_dist_flat)

        # Calculate the total loss
        total_loss = (
                loss_rec_mse
                + self.alpha * loss_commit
                + self.beta * loss_som
                + prob_l
                + prob_z_l * self.tau
        )

        return total_loss

    def loss_prob(self, k):
        """
        Computes the probability loss.

        Parameters:
            k (torch.Tensor): Position of the embeddings.

        Returns:
            prob_l (float): Probability loss.
        """
        # Split k into row and column components
        k_1 = k // self.som_dim[1]
        k_2 = k % self.som_dim[1]

        # Shift k to obtain neighbor indices
        k_1_old = torch.cat([k_1[0:1], k_1[:-1]], dim=0)
        k_2_old = torch.cat([k_2[0:1], k_2[:-1]], dim=0)
        k_stacked = torch.stack([k_1_old, k_2_old, k_1, k_2], dim=1)

        # Compute probabilities
        probs_raw = self.probs
        probs_pos = torch.exp(probs_raw)
        probs_sum = torch.sum(probs_pos, dim=[-1, -2], keepdim=True)
        self.probs.data = probs_pos / probs_sum

        # Gather transition probabilities
        transitions_all = self._gather_nd(self.probs, k_stacked)

        # Compute probability loss
        prob_l = -self.gamma * torch.mean(torch.log(transitions_all))

        return prob_l

    def _loss_z_prob(self, k, z_dist_flat):
        """
        Computes the probability-weighted loss based on squared distances.

        Parameters:
            k (torch.Tensor): Position of the embeddings.
            z_dist_flat (torch.Tensor): Flattened squared distances between z_e and SOM embeddings.

        Returns:
            prob_z_l (float): Probability-weighted loss based on squared distances.
        """
        # Split k into row and column components
        k_1 = k // self.som_dim[1]
        k_2 = k % self.som_dim[1]

        # Shift k to obtain neighbor indices
        k_1_old = torch.cat([k_1[0:1], k_1[:-1]], dim=0)
        k_2_old = torch.cat([k_2[0:1], k_2[:-1]], dim=0)
        k_stacked = torch.stack([k_1_old, k_2_old], dim=1)

        # Compute probabilities
        probs_raw = self.probs
        probs_pos = torch.exp(probs_raw)
        probs_sum = torch.sum(probs_pos, dim=[-1, -2], keepdim=True)
        self.probs.data = probs_pos / probs_sum

        # Gather transition probabilities
        out_probabilities_old = self._gather_nd(self.probs, k_stacked)
        out_probabilities_flat = out_probabilities_old.view(k.shape[0], -1)

        # Compute weighted squared distances
        weighted_z_dist_prob = z_dist_flat * out_probabilities_flat

        # Calculate the probability-weighted loss
        prob_z_l = torch.mean(weighted_z_dist_prob)

        return prob_z_l

    @staticmethod
    def _gather_nd(params, idx):
        """
        Gather elements from the `params` tensor using the indices provided in `idx`.

        Parameters:
            params (torch.Tensor): The tensor from which elements are gathered.
            idx (torch.Tensor): The indices used for gathering.

        Returns:
            outputs (torch.Tensor): Gathered elements.
        """
        # Ensure that indices are of type long
        idx = idx.long()

        # Initialize an empty list for storing gathered elements
        outputs = []

        # Iterate through the rows of `idx`
        for i in range(len(idx)):
            # Gather elements using the indices in each row of `idx`
            row_output = params[[idx[i][j] for j in range(idx.shape[1])]]

            # Append the row's output to the list
            outputs.append(row_output)

        # Stack the gathered elements to form the final output tensor
        outputs = torch.stack(outputs)

        return outputs


def anomaly_detection(anomalous_data, model,):

    model.eval()
    losses = []
    y = []

    for data in anomalous_data:
        data = data.view(-1, data.shape[0]).float()

        _, x_hat_e, _, _, _, _, _ = model(data)
        losses.append(F.mse_loss(x_hat_e, data).item())
        y.append(x_hat_e.detach().numpy()[0])
    #print(anomalous_data)
    #anomalous_data = torch.stack(anomalous_data)
    #anomalous_data = anomalous_data.detach().numpy()
    #print(losses)

    x = np.arange(1, len(anomalous_data)+1)
    plt.plot(x, y)
    plt.plot(x, anomalous_data)


    return losses


def train_somvae(train_data, valid_data, data_name, writer=None):
    """
    Train a SOM-VAE model using the provided training data.

    Parameters:
    - train_data (list): A list of the training data sequences.
    - valid_data (list): A list of the validation data sequences.
    - writer (TensorBoard.SummaryWriter, optional): TensorBoard summary writer for logging.
    - index (int, optional): Index for selecting specific hyperparameters. Defaults to None.

    Returns:
    - Tuple: A tuple containing the trained model, latent data and a list of unique latent
    representations.
    """
    # Open JSON file with the SOM-VAE hyperparameters
    file_path = os.path.expanduser(
        f'~/{os.path.relpath(os.getcwd(), start=os.path.expanduser("~"))}/Hyperparameters/'
        f'hyperparameters_somvae_{data_name}.json')
    with open(file_path, 'r') as file:
        hyperparameters = json.load(file)

    # Setting hyperparameters
    learning_rate = hyperparameters['learning_rate']
    num_epochs = hyperparameters['num_epochs']

    input_dim = hyperparameters['input_dim']
    hidden_dim = hyperparameters['hidden_dim']
    latent_dim = hyperparameters['latent_dim']

    som_dim = hyperparameters['som_dim6']

    alpha = hyperparameters['alpha']
    beta = hyperparameters['beta']
    gamma = hyperparameters['gamma']
    tau = hyperparameters['tau']

    if writer is not None:
        writer.add_scalar('Learning Rate', learning_rate)
        writer.add_scalar('Number of epochs', num_epochs)
        writer.add_scalar('Alpha', alpha)
        writer.add_scalar('Beta', beta)
        writer.add_scalar('SOM Dim 1', som_dim[0])
        writer.add_scalar('SOM Dim 2', som_dim[1])
        writer.add_scalar('Input Dim', input_dim)
        writer.add_scalar('Hidden Dim', hidden_dim)
        writer.add_scalar('Latent Dim', latent_dim)

    # Initialize model and optimizers
    model = SOMVAE(alpha=alpha, beta=beta, gamma=gamma, tau=tau, som_dim=som_dim, input_length=input_dim,
                   hidden_dim=hidden_dim, latent_dim=latent_dim)
    opt_model = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('Start training')

    # Train the model
    for epoch in range(num_epochs):

        opt_model.zero_grad()

        # Train model
        x_hat_q, x_hat_e, z_e, z_q, k, z_dist_flat, z_q_neighbors = model(train_data)

        # Compute loss
        loss = model.loss(train_data, x_hat_e, x_hat_q, z_e, z_q, z_q_neighbors, k, z_dist_flat)

        # Take optimization step
        loss.backward()
        opt_model.step()

        if writer is not None:
            writer.add_scalar('Train Loss', loss.item(), epoch)
        print(f'Loss of {epoch + 1}. epoch: ', loss.item())

    # Initialize lists
    embeddings = []
    model.eval()

    # Get the embeddings (SOM nodes)
    for i in range(model.som_dim[0]):
        for j in range(model.som_dim[1]):
            embeddings.append(model.embeddings[i][j])

    # Create dictionary that assigns an integer to each embedding
    embeddings_dict = {str(k): v for v, k in enumerate(embeddings)}

    # Get latents of model
    x_hat_q, x_hat_e, z_e, z_q, k, z_dist_flat, z_q_neighbors = model(valid_data)

    # Compute loss
    loss = model.loss(valid_data, x_hat_e, x_hat_q, z_e, z_q, z_q_neighbors, k, z_dist_flat)

    # Map embeddings to corresponding integers
    latent_data = torch.tensor([embeddings_dict[str(z_q[i])] for i, _ in enumerate(z_q)])

    if writer is not None:
        writer.add_scalar('Valid Loss', loss.item())
        writer.add_scalar('Number of States', len(torch.unique(latent_data)))
        torch.save(model, f'Models/somvae/{data_name}')
    print('Validation Loss: ', loss.item())
    print('Number of States: ', len(torch.unique(latent_data)))

    return model, latent_data


def instantiate_somvae():
    # Open JSON file with the SOM-VAE hyperparameters
    file_path = os.path.expanduser(
        f'~/{os.path.relpath(os.getcwd(), start=os.path.expanduser("~"))}/Hyperparameters/hyperparameters_somvae.json')
    with open(file_path, 'r') as file:
        hyperparameters = json.load(file)

    som_dim = hyperparameters['som_dim6']
    alpha = hyperparameters['alpha']
    beta = hyperparameters['beta']
    gamma = hyperparameters['gamma']
    tau = hyperparameters['tau']

    # Initialize model and optimizers
    model = SOMVAE(alpha=alpha, beta=beta, gamma=gamma, tau=tau,
                   som_dim=som_dim, input_length=hyperparameters['input_dim'],
                   hidden_dim=hyperparameters['hidden_dim'], latent_dim=hyperparameters['latent_dim'])

    return model
