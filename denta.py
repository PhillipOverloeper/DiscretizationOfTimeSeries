import pprint
import time
import pandas as pd
from plotly import graph_objects as go
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from datetime import timedelta
from multiprocessing import cpu_count
import automaton_learning


class DENTA(nn.Module):
    def __init__(self, num_signals, num_hidden, sigma=1., sigma_learnable=False,
                 device='cpu'):
        super(DENTA, self).__init__()
        self.variant = 'gbrbm'

        if sigma_learnable:
            self.is_sigma_learnable = True
            self.log_sigma_x = nn.Parameter(np.log(sigma) * torch.ones(1, num_signals, requires_grad=True)).to(device)
        else:
            self.is_sigma_learnable = False
            self.log_sigma_x = np.log(sigma) * torch.ones(1, num_signals, requires_grad=False).to(device)

        self._encoder = nn.Sequential().to(device)
        self._decoder = nn.Sequential().to(device)

        # bias of the input we take as separate parameter for implementation reasons
        self.bx = nn.Parameter(torch.zeros(1, num_signals, requires_grad=True)).to(device)

        # add encoder layers
        enc_lin = nn.Linear(num_signals, num_hidden, device=device)
        self._encoder.add_module('linear_v2h', enc_lin)
        self._encoder.add_module('sigmoid_v2h', nn.Sigmoid())

        dec_lin = nn.Linear(num_hidden, num_signals, device=device)
        dec_lin.weight = nn.Parameter(enc_lin.weight.transpose(0, 1))

        self._decoder.add_module('linear_h2v', dec_lin)

        self.free_energy_components = nn.Sequential()
        self.free_energy_components.add_module('linear_energy', enc_lin)
        self.free_energy_components.add_module('softplus_energy', nn.Softplus())

        self.threshold = 0
        self.learning_curve = []
        self.valid_curve = []
        self.num_epoch = 0

        # print initialized pytorch model
        print(self)

    def encode(self, x):
        x = torch.div(x, torch.exp(self.log_sigma_x))
        return self._encoder(x)

    def predict_discrete_mode(self, data):
        h = [torch.round(self.encode(d)) for d in data]
        h = [automaton_learning.bin_vec_to_dec_cell(d) for d in h]
        return torch.cat(h)

    def decode(self, h):
        y = self._decoder(h)
        y = torch.mul(y, torch.exp(self.log_sigma_x))
        return y

    def forward(self, x):
        return self.decode(self.encode(x))

    def energy(self, x, h):
        vis = torch.sum(torch.div(torch.square(x - self.bx), (2 * torch.square(torch.exp(self.log_sigma_x)))), dim=1)
        hid = torch.matmul(h, self._encoder[-2].bias)
        xWh = torch.sum(torch.matmul(x, self._encoder[-2].weight.T) * h, dim=1)
        return vis - hid - xWh

    def free_energy(self, x):
        vis = torch.sum(torch.div(torch.square(x - self.bx), (2 * torch.square(torch.exp(self.log_sigma_x)))), dim=1)
        x = torch.div(x, torch.exp(self.log_sigma_x))
        return vis - torch.sum(self.free_energy_components(x), dim=1)

    def score(self, x, sigma=None):
        x = x.requires_grad_()
        logp = -self.free_energy_components(x).sum()
        grad = torch.autograd.grad(logp, x, create_graph=True)[0] # Create graph True to allow later backprop
        return grad

    def dsm_loss(self, x, v, sigma=0.1):
        """DSM loss from
        A Connection Between Score Matching
            and Denoising Autoencoders
        The loss is computed as
        x_ = x + v   # noisy samples
        s = -dE(x_)/dx_
        loss = 1/2*||s + (x-x_)/sigma^2||^2
        Args:
            x (torch.Tensor): input samples
            v (torch.Tensor): sampled noises
            sigma (float, optional): noise scale. Defaults to 0.1.

        Returns:
            DSM loss
        """
        x = x.requires_grad_()
        v = v * sigma
        x_ = x + v
        s = self.score(x_)
        loss = torch.norm(s + v / (sigma ** 2), dim=-1) ** 2
        loss = loss.mean() / 2.
        return loss

    def num_x(self):
        l = self._encoder[0]
        return l.in_features

    def num_h(self):
        l = self._decoder[0]
        return l.in_features

    # def sample_h(self, x):
    #     p_h_given_v = self.encode(x)
    #     return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_h(self, h):
        return torch.bernoulli(h)

    # def sample_x(self, h):
    #     p_v_given_h = self.decode(h)
    #     return p_v_given_h, torch.normal(p_v_given_h, std=torch.exp(self.log_sigma_x))
    def sample_x(self, x):
        return torch.normal(x, std=torch.exp(self.log_sigma_x))

    def generate(self, num_examples, num_steps=10):
        x = torch.randn([num_examples, self.num_x()])
        for k in range(num_steps):
            ph, h = self.sample_h(x)
            px, x = self.sample_x(h)
        return x

    def contrastive_divergence(self, v0, h0, vk, hk):
        # return self.free_energy(v0) - self.free_energy(vk)
        return self.energy(v0, h0) - self.energy(vk, hk)

    def recon(self, v):
        # if self.variant == 'dsebm':
        #     v = v.requires_grad_()
        #     logp = -self.free_energy_components(v).sum()
        #     return torch.autograd.grad(logp, v, create_graph=True)[0]
        h = self.encode(v)
        r = self.decode(h)
        return r

    def learn(self, train_data, valid_data, learning_rule='re', valid=0., max_epoch=10, min_epoch=0, weight_decay=0.,
              batch_size=128, shuffle=True, num_k=1, verbose=True, early_stopping=False, early_stopping_patience=3,
              use_probability_last_x_update=False):
        print('Training {} using {}'.format(self.variant, learning_rule))
        is_ebm = self.variant in ['dsebm', 'gbrbm']

        valid_data = None
        if valid > 0:
            valid = round(valid * len(train_data))
            train_data, valid_data = random_split(train_data, [len(train_data) - valid, valid])
            valid_data = next(iter(DataLoader(valid_data, batch_size=len(valid_data))))
            progress = dict()
            progress['MSE'] = torch.mean(self.recon_error(valid_data)).item()
            if is_ebm:
                valid_energy = torch.mean(self.free_energy(valid_data)).item()
                progress['Energy'] = valid_energy
            self.valid_curve.append(progress)

        train_data_loaded = next(iter(DataLoader(train_data, batch_size=len(train_data))))
        progress = dict(MSE=torch.mean(self.recon_error(train_data_loaded)).item())
        if is_ebm:
            progress['Energy'] = torch.mean(self.free_energy(train_data_loaded)).item()
        self.learning_curve.append(progress)

        data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle)
        opt = torch.optim.RMSprop(self.parameters(), weight_decay=weight_decay)
        t_start = time.time()
        for epoch in range(1, max_epoch + 1):
            for i, d in enumerate(data_loader):
                xk = d
                if self.variant in ['dsebm', 'dae']:
                    xk = torch.normal(mean=xk, std=torch.exp(self.log_sigma_x))
                x0 = d
                if learning_rule == 'cd':
                    with torch.no_grad():
                        eh0 = self.encode(x0)
                        h0 = self.sample_h(eh0)
                        hk = h0
                        for k in range(num_k):
                            exk = self.decode(hk)
                            xk = self.sample_x(exk)
                            ehk = self.encode(xk)
                            hk = self.sample_h(ehk)

                    if use_probability_last_x_update:
                        cd = torch.mean(self.contrastive_divergence(x0, h0, exk, ehk))
                    else:
                        cd = torch.mean(self.contrastive_divergence(x0, h0, xk, ehk))
                    opt.zero_grad()
                    cd.backward()
                    opt.step()
                elif learning_rule == 'sm':
                    r = self.recon(xk)
                    loss = (r - x0).pow(2).sum()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                elif learning_rule == 're':
                    r = self.recon(xk)
                    loss = (r - x0).pow(2).sum()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                elif learning_rule == 'dsm':
                    x0noise = torch.normal(torch.zeros_like(x0))
                    loss = self.dsm_loss(x0, x0noise)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            with torch.no_grad():
                progress = dict(MSE=torch.mean(self.recon_error(train_data_loaded)).item())
                if is_ebm:
                    progress['Energy'] = torch.mean(self.free_energy(train_data_loaded)).item()
            if verbose:
                print(f'\n############### Epoch {epoch} ###############')
                print('Train: ')
                pprint.pp(progress)
            self.learning_curve.append(progress)
            if valid_data:
                with torch.no_grad():
                    progress = dict(MSE=torch.mean(self.recon_error(valid_data)).item())
                    if is_ebm:
                        progress['Energy'] = torch.mean(self.free_energy(valid_data)).item()

                self.valid_curve.append(progress)
                if verbose:
                    print('Valid: ')
                    pprint.pp(progress)

                if early_stopping and epoch > min_epoch and epoch > early_stopping_patience:
                    if not is_ebm:
                        valid_metrics = np.array([v['MSE'] for v in self.valid_curve[-early_stopping_patience-1:]])
                        if np.all(valid_metrics[1:] > valid_metrics[0]):
                            print('Early stop after valid metrics: ', valid_metrics)
                            break

            if self.is_sigma_learnable:
                print(torch.exp(self.log_sigma_x))
        self.eval()
        self.num_epoch = epoch
        print('Training finished after ', timedelta(seconds=time.time() - t_start))

    def recon_error(self, data, input=None):
        if input is None:
            input = data
        recon = self.recon(input)
        squared_error = torch.sum(torch.square(data - recon), dim=1)
        return squared_error

    def anomaly_score(self, s, threshold=None):
        # d = d[:]
        # t, s, = d['time'], d
        if self.variant in ['gbrbm']:
            score_in_time = self.free_energy(s)
        else:
            score_in_time = self.recon_error(s)
        score = score_in_time.cpu().detach().numpy()
        if threshold is None:
            return score
        else:
            return score > threshold

    def plot_learning_curve(self):
        return vis.plot_data([pd.DataFrame(self.learning_curve), pd.DataFrame(self.valid_curve)],
                             title='Learning curve', names=['Train', 'Valid'], x_axis_title='Epoch')

    # Transforms the p(v) into mixture of Gaussians and returns the weight, mean and sigma for each Gaussian component as
    # well the corresponding hidden states.This function is for use with very small models.Otherwize it will last forever
    def gmm_model(self):
        def gbrbm_h2v(type, h, W, bv, sigma):
            x = np.matmul(np.atleast_2d(h), W)
            if type == 'gbrbm':
                x *= sigma
            return x + bv
        sigma = np.exp(self.log_sigma_x.detach().numpy())
        bv = self.bx.detach().numpy()
        bh = self._encoder[-2].bias.detach().numpy()
        W = self._encoder[-2].weight.detach().numpy()

        num_components = 2 ** self.num_h()

        if sigma.size == 1:
            sigma = np.repeat(sigma, self.num_x(), axis=1)
            sigma = sigma[None]

        gmm_sigmas = np.repeat(sigma, num_components, axis=0)

        # Initialize
        weights = np.zeros((num_components, 1))
        means = np.zeros((num_components, self.num_x()))
        hid_states = np.zeros((num_components, self.num_h()))

        i = 1
        phi0 = np.prod(np.sqrt(2 * np.pi) * sigma)

        weights[0] = phi0
        means[0, :] = bv
        for i in range(1, num_components):
            hs = list(bin(i)[2:])
            hid_states[i, -len(hs):] = hs
            hs = hid_states[i, :]
            # Calc means
            mean = gbrbm_h2v(self.variant, hs, W, bv, sigma)
            means[i, :] = mean

            # Calc phi
            phi = (np.sum(mean ** 2 / (2 * sigma ** 2)) - np.sum(bv ** 2 / (2 * sigma ** 2)))
            phi = np.sum(phi) + np.sum(bh * hs)
            phi = phi0 * np.exp(phi)

            weights[i] = phi

        # Normalize weights
        Z = sum(weights)
        weights = weights / Z
        return weights, means, gmm_sigmas, hid_states, Z

    def plot_input_space(self, data=None, samples=None, show_gaussian_components=False, data_limit=10000,
                         xmin=None, xmax=None, ymin=None, ymax=None, figure_width=600, figure_height=600,
                         show_axis_titles=True, show_energy_contours=False, showlegend=True,
                         show_recon_error_contours=False, ncontours=None,
                         plot_code_positions=True, show_recon_error_heatmap=False, plot_bias_vector=False,
                         show_reconstructions=False, **kwargs):
        fig = go.Figure()
        if show_recon_error_heatmap:
            if xmin is None and xmax is None and ymin is None and ymax is None:
                if data is None:
                    xmin, xmax = -5, 5
                    ymin, ymax = -5, 5
                else:
                    xmin = ymin = data.min().min()
                    xmax = ymax = data.max().max()

            x = np.linspace(xmin, xmax, 100)
            y = np.linspace(ymin, ymax, 100)
            xv, yv = np.meshgrid(x, y)
            d = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1)])
            with torch.no_grad():
                fe = self.recon_error(torch.Tensor(d)).numpy()

            trace = go.Heatmap(x=x, y=y, z=np.reshape(fe, xv.shape),
                               name="Reconstruction Error", showlegend=True, showscale=False)
            fig.add_trace(trace)

        if show_recon_error_contours and data.shape[0] == 2:
            if xmin is None and xmax is None and ymin is None and ymax is None:
                if data is None:
                    xmin, xmax = -5, 5
                    ymin, ymax = -5, 5
                else:
                    xmin = ymin = data.min().min()
                    xmax = ymax = data.max().max()

            x = np.linspace(xmin, xmax, 100)
            y = np.linspace(ymin, ymax, 100)
            xv, yv = np.meshgrid(x, y)
            d = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1)])
            with torch.no_grad():
                fe = self.recon_error(torch.Tensor(d)).numpy()
                fe = np.reshape(fe, xv.shape)

                trace = go.Contour(x=x, y=y, z=fe, contours=dict(coloring='lines'), name="Reconstruction Error",
                                   showlegend=True, showscale=False, ncontours=ncontours)
                fig.add_trace(trace)

        if show_energy_contours:
            if xmin is None and xmax is None and ymin is None and ymax is None:
                if data is None:
                    xmin, xmax = -5, 5
                    ymin, ymax = -5, 5
                else:
                    xmin = ymin = data.min().min()
                    xmax = ymax = data.max().max()

            x = np.linspace(xmin, xmax, 100)
            y = np.linspace(ymin, ymax, 100)
            xv, yv = np.meshgrid(x, y)
            d = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1)])
            fe = self.free_energy(torch.Tensor(d)).detach().numpy()

            trace = go.Contour(x=x, y=y, z=np.reshape(fe, xv.shape),
                               contours=dict(coloring='lines'), name="Free energy", ncontours=ncontours, showlegend=True, showscale=False)
            fig.add_trace(trace)

        if data is not None:
            if data_limit is not None and data.shape[0] > data_limit:
                data = data.sample(data_limit)
            fig.add_trace(vis.plot2d(data[data.columns[0]], data[data.columns[1]], name='Data',
                                     marker=dict(size=3, opacity=0.2, color='MediumPurple')))
            if show_reconstructions:
                recon = self.recon(torch.Tensor(data.values)).detach().numpy()
                fig.add_trace(vis.plot2d(recon[:,0], recon[:, 1], name='Reconstruction',
                                         marker=dict(size=3, opacity=0.2, color='limegreen')))
        if samples is not None:
            fig.add_trace(vis.plot2d(samples[:, 0], samples[:, 1], name='Samples',
                                     marker=dict(size=3, opacity=0.2, color='darkgreen')))

        if show_axis_titles:
            fig.update_layout(
                xaxis_title="$x_1$",
                yaxis_title="$x_2$",
            )
        if plot_code_positions:
            num_h = self.num_h()
            num_v = self.num_x()
            num_components = 2 ** num_h
            # Initialize
            means = np.zeros((num_components, num_v))
            hid_states = np.zeros((num_components, num_h))
            for i in range(0, num_components):
                hs = list(bin(i)[2:])
                hid_states[i, -len(hs):] = hs
                hs = hid_states[[i], :]
                # Calc means
                mean = self.decode(torch.Tensor(hs))
                means[i, :] = mean.detach().numpy()

            hm_mapping = dict()
            for h, m in zip(list(hid_states), list(means)):
                hm_mapping[str(h)] = m
            for i in range(means.shape[0]):
                mean = means[i, :]
                hid = hid_states[i, :]
                for i, hi in enumerate(hid):
                    if hi == 1:
                        hid_prev = hid.copy()
                        hid_prev[i] = 0
                        mean_start = hm_mapping[str(hid_prev)]
                        fig.add_annotation(xref="x", yref="y", axref="x", ayref="y",
                                           ax=mean_start[0], ay=mean_start[1], x=mean[0], y=mean[1],
                                           showarrow=True, arrowhead=2, arrowsize=1.5)

            fig.add_trace(go.Scatter(x=means[:, 0], y=means[:, 1], text=hid_states, mode='text+markers',
                                     name='Codes', textfont_size=12,
                                     textposition="top left", marker_color='orange', marker_size=4))
        if plot_bias_vector:
            bx = self.bx.detach().numpy()
            fig.add_annotation(xref="x", yref="y", axref="x", ayref="y",
                               x=bx[0][0], y=bx[0][1], ax=0, ay=0,
                               showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1,
                               arrowcolor="#636363")
        if show_gaussian_components:
            weights, means, gmm_sigmas, hid_states, Z = self.gmm_model()
            hm_mapping = dict()
            for h, m in zip(list(hid_states), list(means)):
                hm_mapping[str(h)] = m
            for i in range(weights.shape[0]):
                weight = weights[i, 0]
                mean = means[i, :]
                sigma = gmm_sigmas[i, :]
                hid = hid_states[i, :]
                fig.add_shape(type="circle",
                              xref="x", yref="y",
                              x0=mean[0] - 2 * sigma[0], y0=mean[1] - 2 * sigma[1],
                              x1=mean[0] + 2 * sigma[0], y1=mean[1] + 2 * sigma[1],
                              # opacity=weight/max(max(weights)),
                              fillcolor='rgba(23, 156, 125, {:.2f})'.format(0.7 * weight / max(max(weights))),
                              line_color='rgba(23, 156, 125)',
                              line_width=1,
                              layer='below')
                for i, hi in enumerate(hid):
                    if hi == 1:
                        hid_prev = hid.copy()
                        hid_prev[i] = 0
                        mean_start = hm_mapping[str(hid_prev)]
                        fig.add_annotation(xref="x", yref="y", axref="x", ayref="y",
                                           ax=mean_start[0], ay=mean_start[1], x=mean[0], y=mean[1],
                                           showarrow=True, arrowhead=2, arrowsize=1.5)

            weights = list(weights[i, :] for i in range(weights.shape[0]))
            hid_states = [' '.join(list(hid_states[i, :].astype(int).astype(str))) for i in range(hid_states.shape[0])]
            # fig.add_trace(go.Scatter(x=means[:, 0], y=means[:, 1], text=hid_states, mode='text+markers',
            #                          hovertext=weights,
            #                          name='GMM',
            #                          textposition="top left", marker_color='orange'))
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            title_standoff=0,
            range=[ymin, ymax]
        )
        fig.update_xaxes(
            title_standoff=0,
            range=[xmin, xmax]
        )
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                          width=figure_width,
                          height=figure_height,
                          showlegend=showlegend,
                          legend=dict(yanchor="bottom", y=1, xanchor="left", x=0.01, orientation="h",
                                      font=dict(size=8)))
        fig.update_layout(**kwargs)
        return fig

    def find_optimal_threshold_for_f1(self, data, search_every=1, plot=False):
        scores_unsorted = self.anomaly_score(data[:])
        sort_ind = np.argsort(scores_unsorted)
        sort_ind = sort_ind[0::search_every]
        thresholds = scores_unsorted[sort_ind]
        labels_unsorted = data[:]['label'].cpu().detach().numpy()

        f1_scores = [f1_score(labels_unsorted != 0, scores_unsorted > th) for th in thresholds]
        opt_ind = np.argmax(f1_scores)
        opt_th = thresholds[opt_ind]
        max_f1 = f1_scores[opt_ind]
        if plot:
            fig = vis.plot2d(np.arange(0, scores_unsorted.shape[0]), scores_unsorted, return_figure=True)
            fig.add_trace(vis.plot2d(np.arange(0, labels_unsorted.shape[0]), labels_unsorted))
            fig.show()
            vis.plot2d(thresholds, f1_scores, return_figure=True).show()
        return opt_th, max_f1

    def get_auroc(self, data):
        scores = self.anomaly_score(data[:])
        labels = data[:]['label'].cpu().detach().numpy()
        labels = labels != 0
        return roc_auc_score(labels, scores)
