import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from CatVAE_datamodule import DataModule
from CatVAE_experimentstool import *
import os
import yaml
import pandas as pd


class CategoricalVAE(pl.LightningModule):
    def __init__(self, hparams:dict) -> None:
        super(CategoricalVAE, self).__init__()
        # parameters from hparams dictionary
        self.in_dim = hparams["IN_DIM"]
        self.enc_out_dim = hparams["ENC_OUT_DIM"]
        self.dec_out_dim = hparams["DEC_OUT_DIM"]
        self.categorical_dim = hparams["CATEGORICAL_DIM"]
        self.temp = hparams["TEMPERATURE"]
        self.beta = hparams["BETA"]

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.enc_out_dim))
        self.fc_z_cat = nn.Linear(self.enc_out_dim, self.categorical_dim)

        # Build Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.categorical_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.dec_out_dim))
        self.fc_mu_x = nn.Linear(self.dec_out_dim, self.in_dim)
        self.fc_logvar_x = nn.Linear(self.dec_out_dim, self.in_dim)

        # Categorical prior
        self.pz = torch.distributions.OneHotCategorical(
            1. / self.categorical_dim * torch.ones(1, self.categorical_dim, device='cpu'))

        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.)[0], requires_grad=True )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input tensor to encoder
        :return z_out: (Tensor) Latent code
        """
        result = self.encoder(x)
        z = self.fc_z_cat(torch.flatten(result, start_dim=1))
        z_out = z.view(-1, self.categorical_dim)
        return z_out

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes parameters for pxz from sampels of pzx
        :param z: (Tensor) 
        :return: mu (Tensor) 
        :return: sigma (Tensor)
        """
        result = self.decoder(z)
        mu = self.fc_mu_x(result)
        logvar = self.fc_logvar_x(result)
        sigma = torch.cat(
                        [torch.diag(torch.exp(logvar[i, :])) for i in range(z.shape[0])]
                       ).view(-1, self.in_dim, self.in_dim)
        return mu, sigma

    def sample_gumble(self, logits: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param logits: (Tensor) Latent Codes 
        :return: (Tensor)
        """
        # Sample from Gumbel
        u = torch.rand_like(logits)
        g = - torch.log(- torch.log(u + eps) + eps)
        s = F.softmax((logits + g) / self.temp, dim=-1)
        return s

    def shared_eval(self, x: torch.Tensor):
        """
        shared computation of all steps/methods in CatVAE
        """
        # first compute parameters of categorical dist. pzx
        pzx_logits = self.encode(x)
        # create one hot categorical dist. object for use in loss func
        pzx = torch.distributions.OneHotCategorical(logits=pzx_logits)
        # sample from pzx
        z = self.sample_gumble(logits=pzx_logits)
        # decode into mu and sigma
        mu, sigma = self.decode(z)
        # construct multivariate distribution object for pxz
        pxz = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=sigma)
        return pzx_logits, pzx, mu, sigma, pxz, z
    
    def shared_testing(self, x: torch.Tensor):
        """
        shared computation of all steps/methods in CatVAE
        """
        # first compute parameters of categorical dist. pzx
        pzx_logits = self.encode(x)
        # create one hot categorical dist. object for use in loss func
        pzx = torch.distributions.OneHotCategorical(logits=pzx_logits)
        # sample from pzx
        z = self.sample_gumble(logits=pzx_logits)
        # decode into mu and sigma
        mu, sigma = self.decode(z)
        # construct multivariate distribution object for pxz
        pxz = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=sigma)
        return pzx_logits, pzx, mu, sigma, pxz, z
    
    def anom_detect(self, x, **kwargs):
        pzx_logits, pzx, mu, sigma, pxz, z = self.shared_testing(x)
        likelihood = pxz.log_prob(x[0]).detach().cpu().numpy()
        # anom_labels_likelihood = np.where(likelihood<anom_threshold_likelihood, 1, 0)

        mse_error = ((x - mu)**2).mean(dim=1).detach().cpu().numpy()
        # anom_labels_mse = np.where(anom_label_mse<anom_threshold_mse, 1, 0)
        return likelihood, mse_error

    def forward(self, x: torch.Tensor):
        pzx_logits, pzx, mu, sigma, pxz, z = self.shared_eval(x)
        loss_dct = self.loss_function(x=x, pzx=pzx, pxz=pxz)

        return loss_dct['Loss']

    def loss_function(self, x: torch.Tensor,
                    pzx: torch.distributions.OneHotCategorical, 
                    pxz: torch.distributions.MultivariateNormal) -> dict:
    
        likelihood = pxz.log_prob(x[0])
        recon_loss = torch.mean(likelihood)
        # compute kl divergence for categorical dist
        kl_categorical = torch.distributions.kl.kl_divergence(pzx, self.pz)
        kl_categorical_batch = torch.mean(kl_categorical)
        loss = -recon_loss + self.beta*kl_categorical_batch
        return {'Loss': loss, 'recon_loss': recon_loss, 'KLD_cat': kl_categorical_batch}

    def discret_comp(self, train_data):

        unique_list = []

        pzx_logits, _, _, _, _, z = self.shared_testing(train_data)
        latent_data = torch.zeros(z.shape).to(device='cpu').scatter(1, torch.argmax(pzx_logits, dim=1).unsqueeze(1), 1).cpu().detach().numpy()
        cats = torch.tensor(pd.DataFrame(latent_data).idxmax(axis=1))

        flattened_tensors = [tuple(tensor.tolist()) for tensor in latent_data]
        unique_tensors = list(set(flattened_tensors))
        for item in unique_tensors:
            unique_list.append(item)
        unique_list = list(set(unique_list))

        return cats, unique_list


def exec_catvae(data_name, train_data, valid_data, writer=None):

    with open(f'./Hyperparameters/hparams_{data_name}.yaml') as f:
        hparam = (yaml.safe_load(f))['hparams']

    learning_rate = hparam['LEARNING_RATE']
    num_epochs = hparam['NUM_EPOCHS']

    model = CategoricalVAE(hparams=hparam)
    opt_model = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    print('Start training')

    for epoch in range(num_epochs):

        opt_model.zero_grad()
        # for idx, batch in enumerate(batches):
        loss = model(train_data)

        loss.backward()

        # Take optimization step
        opt_model.step()

        # avg_batch_loss = batch_loss / len(train_data)
        if writer is not None:
            writer.add_scalar('Train Loss', loss.item(), epoch)
        print(f'Loss of {epoch + 1}. epoch: ', loss.item())

    latent_data, _ = model.discret_comp(valid_data)

    return latent_data


def instantiate_catvae():

    with open(f'.Hyperparameters/hparams-simu_tank.yaml') as f:
        hparam = yaml.safe_load(f)
    model = CategoricalVAE(hparams=hparam["hparams"])
    return model


def compute_mse_error(model, train):

    likelihood, mse_error = model.anom_detect(x=torch.tensor(train.values, dtype=torch.float).to(device='cuda'))
    