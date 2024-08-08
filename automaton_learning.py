import torch
import numpy as np


def extract_events(h, time, selected_option):
    """
    Extracts the timing of event changes from the latent data.

    Args:
        h (PyTorch Tensor): Latent representation of time series data.
        time (PyTorch Tensor): Time values of the time series data.
        selected_option (string): Which model is trained.

    Returns:
        tuple: A tuple containing:
            - timings_list (list): List of timings corresponding to the event changes.
            - latent (list): List of unique latent states in the time series data.
    """

    # Trained model is the SOM-VAE
    if selected_option == 'somvae':
        # Reference is first latent representation
        href = h[0]
        # List of unique latent representations in this time series
        latent = [href.detach().cpu().numpy()]
        # List of time values where changes occured
        timings = [time[0]]

        # Iterate through the data and extract timing of events
        for k in range(len(h)):
            h_val = h[k]                                                    # Kth entry in latent data matrix

            if not torch.all(torch.eq(href, h_val)):                        # If change occured
                timings.append(time[k])                                     # Add time at which change occured
                href = h_val                                                # Set new reference
                latent.append(href.detach().cpu().numpy())                  # Add new reference value

        return timings, latent

    # Trained model is RBM
    elif selected_option == 'rbm' or selected_option == 'denta':
        # Reference is the first latent representation
        href = np.round(h[0])
        # List of unique latent representations in this time series
        latent = [href]
        # List of time values where changes occured
        timings = [time[0]]

        # Iterate through the data and extract events
        for k in range(h.shape[0]):
            h_val = h[k]                                        # Kth entry in latent data matrix

            if not href == h_val:                               # If difference greater than threshold

                timings.append(time[k])                         # Add time at which change occured
                href = np.round(h_val)                          # Set new reference
                latent.append(href)                             # Add new reference value
        return timings, latent

    # Trained model is CAT-VAE
    elif selected_option == 'catvae':
        # Reference is the first latent representation
        href = h[0]
        # List of unique latent representations in this time series
        latent = [href.detach().cpu().numpy()]
        # List of time values where changes occured
        timings = [time[0]]

        # Iterate through the data and extract events
        for k in range(len(h)):
            h_val = h[k]                                        # Kth entry in latent data matrix

            if not torch.equal(href, h_val):                    # If difference greater than threshold
                timings.append(time[k])                         # Add time at which change occured
                href = h_val                                    # Set new reference
                latent.append(href.detach().cpu().numpy())      # Add new reference value

        return timings, latent

    else:
        raise ValueError('Invalid choice of model!')


def bin_vec_to_dec_cell(binary_array):
    """
        Transforms a binary vector to a decimal number.

        Args:
            binary_array (list, np.ndarray): The array of binary vectors.

        Returns:
            dec_numbers (float): The array of decimal numbers.
        """

    dec_numbers = []
    # Transform every binary vector in a decimal number.
    for row in binary_array:
        binary_row = [int(x) for x in row]
        binary_string = ''.join(map(str, binary_row))
        decimal_number = int(binary_string, 2)
        dec_numbers.append(decimal_number)
    dec_numbers = torch.tensor(dec_numbers)
    return dec_numbers
