import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os,sys, shutil
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from loss import InfoNCE
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch
import wandb
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertModel, BertConfig
from Models.utils import DefineMetricCallback_cls, DefineMetricCallback_reg
from Encoders.Linear import linear_sequential
from NormalizingFlows.NormalizingFlowDensity import NormalizingFlowDensity
from NormalizingFlows.BatchedNormalizingFlowDensity import BatchedNormalizingFlowDensity
from NormalizingFlows.MixtureDensity import MixtureDensity
from torch.distributions.dirichlet import Dirichlet
from torchmetrics.classification import BinaryAUROC, AUROC
from torchmetrics.classification import BinaryF1Score, BinaryAveragePrecision
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns




__budget_functions__ = {'one': lambda N: torch.ones_like(N),
                        'log': lambda N: torch.log(N + 1.),
                        'id': lambda N: N,
                        'id_normalized': lambda N: N / N.sum(),
                        'exp': lambda N: torch.exp(N),
                        'parametrized': lambda N: torch.nn.Parameter(torch.ones_like(N).to(N.device))}


class BrainPostNet(pl.LightningModule):
    def __init__(self, args):
        super(BrainPostNet, self).__init__()

        self.args = args
        self.in_dim = self.args.in_feature_dim if self.args.BOLD_signal else 85 # 132, 177
        self.hidden_dim = 1024
        self.latent_dim = self.args.latent_dim
        self.n_density = self.args.n_density
        self.out_dim = 2 if self.args.target in self.args.targets else 1
        self.regr = 1e-5
        self.loss_name = self.args.loss_name
        self.no_density = False if self.loss_name == 'UCE' else True
        self.encoder_type = self.args.encoder_type
        self.flow_type = self.args.flow_type

        self.best_val_accuracy = 0.0


        self.model = PosteriorNetwork(N=[100, 100], # Count of data from each class in training set. list of ints
                                    args=self.args, 
                                    input_dims=self.in_dim,  # Input dimension. list of ints
                                    output_dim=self.out_dim,  # Output dimension. int
                                    hidden_dims=[self.hidden_dim, self.hidden_dim],  # Hidden dimensions. list of ints
                                    kernel_dim=None,  # Kernel dimension if conv encoder_type. int
                                    latent_dim=self.latent_dim,  # Latent dimension. int
                                    encoder_type=self.encoder_type,  # Encoder encoder_type name. int
                                    k_lipschitz=None,  # Lipschitz constant. float or None (if no lipschitz)
                                    no_density=self.no_density,  # Use density estimation or not. boolean
                                    density_type=self.flow_type,  # radial_flow, planar_flow, iaf_flow, normal_mixture. string
                                    n_density=self.n_density,  # Number of density components. int
                                    budget_function='id',  # Budget function name applied on class count. name
                                    regr=self.regr,  # Regularization factor in Bayesian loss. float
                                    seed=123,
                                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Random seed for init. int
        

        # self.loss = nn.CrossEntropyLoss() if self.args.target in self.args.targets else nn.L1Loss()
        self.metric_name = 'accuracy' if self.args.target in self.args.targets else 'L1loss'
        self.metric = torchmetrics.classification.BinaryAccuracy().to(self.device) \
        if self.args.target in self.args.targets else torchmetrics.regression.MeanAbsoluteError()
        self.auc = BinaryAUROC().to(self.device)
        self.aupr = BinaryAveragePrecision().to(self.device)
        self.f1 = BinaryF1Score().to(self.device)
        self.ce_loss = nn.CrossEntropyLoss()

        # self.save_hyperparameters()
    



    def forward(self, x):
        
        alpha, y_pred = self.model(x, return_output='soft')

        return y_pred
    

    def training_step(self, batch, batch_idx):
        y_true, seq, _ = batch
        alpha, y_pred = self.model(seq, return_output='soft')
        y_true_one_hot = F.one_hot(y_true, num_classes=self.out_dim)

        if self.args.target == 'age':
            y_pred = y_pred.squeeze()

        if self.loss_name == 'UCE':
            loss = self.UCE_loss(alpha, y_true_one_hot)

        elif self.loss_name == 'CE':
            # loss = self.CE_loss(y_pred, y_true_one_hot)
            loss = self.ce_loss(y_pred, y_true)

        if self.args.target in self.args.targets:
            y_pred = self.predict(y_pred)




        self.log("train_loss", loss, prog_bar=True,sync_dist=True)
        self.log(f"train_{self.metric_name}", self.metric(y_pred, y_true), prog_bar=True,sync_dist=True)

        return loss

    # def on_validation_start(self):
    #     self.validation_outputs = []
    def on_validation_epoch_start(self):
        self.validation_outputs = []



    def validation_step(self, batch, batch_idx):
        y_true, seq, _ = batch
        alpha, y_pred = self.model(seq, return_output='soft')
        y_true_one_hot = F.one_hot(y_true, num_classes=self.out_dim)

        if self.args.target == 'age':
            y_pred = y_pred.squeeze(0)

        if self.loss_name == 'UCE':
            loss = self.UCE_loss(alpha, y_true_one_hot)

        elif self.loss_name == 'CE':
            # loss = self.CE_loss(y_pred, y_true_one_hot)
            loss = self.ce_loss(y_pred, y_true)

        if self.args.target in self.args.targets:
            # soft_pred,_ = torch.max(y_pred, 1)
            soft_pred = y_pred
            y_pred = self.predict(y_pred)


        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        # self.log('val_accuracy', self.metric(y_pred, y_true), prog_bar=True, on_epoch=True)
        # self.log('val_auc', self.auc(y_pred, y_true), prog_bar=True, on_epoch=True)
        # self.log('val_f1', self.f1(y_pred, y_true), prog_bar=True, on_epoch=True)

        self.validation_outputs.append({"y_pred": y_pred,
                                        "soft_pred":soft_pred, 
                                        "y_true": y_true,
                                        "alpha": alpha})

    def plot_and_metrics(self, testloader):
        self.test_results = []
        self.model = self.model.cuda()
        self.auc = self.auc.cuda()
        self.aupr = self.aupr.cuda()
        self.f1 = self.f1.cuda()
        self.metric = self.metric.cuda()


        self.model.eval()
        for batch in testloader:
            y_true, seq, _ = batch
            y_true, seq = y_true.cuda(), seq.cuda()
            alpha, y_pred = self.model(seq, return_output='soft')
            y_true_one_hot = F.one_hot(y_true, num_classes=self.out_dim)

            if self.args.target in self.args.targets:
                # soft_pred,_ = torch.max(y_pred, 1)
                soft_pred = y_pred
                y_pred = self.predict(y_pred)

            self.test_results.append({"y_pred": y_pred,
                                            "soft_pred":soft_pred, 
                                            "y_true": y_true,
                                            "alpha": alpha})
            
        retrieve_y_pred = [x['y_pred'] for x in self.test_results]
        retrieve_y_soft_pred = [x['soft_pred'] for x in self.test_results]
        retrieve_y_true = [x['y_true'] for x in self.test_results]
        retrieve_alpha = [x['alpha'] for x in self.test_results]


        # unc, current_val_accuracy, current_val_f1 = self._compute_metrics(torch.cat(retrieve_y_pred), 
        #                                                                 torch.cat(retrieve_y_soft_pred), 
        #                                                                 torch.cat(retrieve_y_true),
        #                                                                 torch.cat(retrieve_alpha)
        #                                                                 )
        # self.test_results ={"unc": unc, "accuracy": current_val_accuracy, "f1": current_val_f1}
        # print(self.test_results)
        self.plot_confidence(torch.cat(retrieve_y_pred), torch.cat(retrieve_y_soft_pred), torch.cat(retrieve_y_true), torch.cat(retrieve_alpha))


        if self.args.OOD:
            ood_X = self.args.ood_X.cuda()
            val_size = torch.cat(retrieve_y_pred).size(0)

            _indices = torch.randperm(ood_X.size(0))[:val_size]
            ood_X = ood_X[_indices]

            ood_alpha, _ = self.model(ood_X, return_output='soft')
            self.plot_entropy(torch.cat(retrieve_alpha), ood_alpha)

    def plot_entropy(self, alpha, ood_alpha):

        eps = 1e-6
        alpha = alpha + eps
        ood_alpha = ood_alpha + eps
        alpha0 = alpha.sum(-1)
        ood_alpha0 = ood_alpha.sum(-1)

        id_log_term = torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(alpha0)
        id_digamma_term = torch.sum((alpha - 1.0) * (
                    torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))), dim=-1)
        id_differential_entropy = id_log_term - id_digamma_term

        ood_log_term = torch.sum(torch.lgamma(ood_alpha), dim=-1) - torch.lgamma(ood_alpha0)
        ood_digamma_term = torch.sum((ood_alpha - 1.0) * (torch.digamma(ood_alpha) - torch.digamma(
            (ood_alpha0.reshape((ood_alpha0.size()[0], 1))).expand_as(ood_alpha))), dim=-1)
        ood_differential_entropy = ood_log_term - ood_digamma_term

        scores =  id_differential_entropy.cpu().detach().numpy()
        ood_scores =  ood_differential_entropy.cpu().detach().numpy()

        # np.save('in_entropy.npy', scores)
        # np.save('ood_entropy.npy', ood_scores)

        plt.figure(figsize=(10, 6))
        sns.kdeplot(scores, label='ADNI/ADNID', fill=True, color='blue', alpha=0.3)
        sns.kdeplot(ood_scores, label='OASIS', fill=True, color='red', alpha=0.3)
        plt.xlabel('Differential Entropy')
        plt.ylabel('Density')
        plt.title('BPEN')
        plt.legend()
        plt.savefig('entropy_plot.png', bbox_inches='tight')
        plt.show()


    def plot_confidence(self,y_pred, y_pred_soft, y_true, alpha):
        corrects = (y_true.squeeze() == alpha.max(-1)[1]).cpu().detach().numpy()
        p = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        aleatoric_confidence_calibraton_scores = p.max(-1)[0]
        scores = alpha.max(-1)[0]

        scores_norm = F.normalize(scores, p=1, dim=-1).cpu().detach().numpy()


        # Sort predictions by descending confidence scores
        sorted_indices = np.argsort(scores_norm)[::-1]
        sorted_scores = scores_norm[sorted_indices]
        sorted_corrects = corrects[sorted_indices]

        # Define the percentiles (e.g., top 1%, top 5%, ..., up to top 100%)
        percentiles = np.arange(1, 11)

        # Initialize a list to store accuracies for each percentile
        accuracies = []

        for percentile in percentiles:
            # Calculate the index up to which to consider predictions (for the top percentile)
            index = int(len(sorted_scores) * (percentile / 10))
            # Calculate accuracy for the top percentile
            if index > 0:
                accuracy = np.mean(sorted_corrects[:index])  # This works directly with boolean arrays
            else:
                accuracy = None  # In case there are no predictions in the top percentile
            accuracies.append(accuracy)

        # Convert percentiles to confidence levels for plotting (percentiles divided by 100)
        confidence_levels = percentiles / 10

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(confidence_levels, accuracies, marker='o', linestyle='-', color='b')

        # np.save('x_axis.npy',confidence_levels)
        # np.save('BPEN_accs_vs_uncertainty_y.npy',np.array(accuracies))

        plt.title('Accuracy vs Uncertainty Threshold')
        plt.xlabel('Uncertainty Threshold')
        plt.ylabel('Accuracy')

        plt.grid(True)
        plt.show()
        plt.savefig('conf_plot.png', bbox_inches='tight')



    def brier_score(self, y_pred, y_true):

        return torch.mean(torch.norm(y_pred - y_true, p=2, dim=1))
    
    def _compute_metrics(self, y_pred, y_pred_soft, y_true, alpha):
        y_true_one_hot = torch.nn.functional.one_hot(y_true)
        Brier_score = self.brier_score(y_pred_soft, y_true_one_hot)
        # auc = self.auc(y_pred_soft, y_true)
        accuracy = self.metric(y_pred, y_true)
        f1 = self.f1(y_pred, y_true)

        corrects = (y_true.squeeze() == alpha.max(-1)[1])
        p = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        aleatoric_confidence_calibraton_scores = p.max(-1)[0]
        epistemic_confidence_calibraton_scores = alpha.max(-1)[0]

        # aupr = metrics.average_precision_score(corrects.cpu().detach().numpy(), epistemic_confidence_calibraton_scores.cpu().detach().numpy())
        ale_aupr = self.aupr(aleatoric_confidence_calibraton_scores, corrects)
        ale_auc = self.auc(aleatoric_confidence_calibraton_scores, corrects)

        # ale_aupr = metrics.average_precision_score(corrects.cpu().detach().numpy(), aleatoric_confidence_calibraton_scores.cpu().detach().numpy())
        # ale_auc = metrics.roc_auc_score(corrects.cpu().detach().numpy(), aleatoric_confidence_calibraton_scores.cpu().detach().numpy())

        epi_aupr = self.aupr(epistemic_confidence_calibraton_scores, corrects)
        epi_auc = self.auc(epistemic_confidence_calibraton_scores, corrects)

        # epi_aupr = metrics.average_precision_score(corrects.cpu().detach().numpy(), epistemic_confidence_calibraton_scores.cpu().detach().numpy())
        # epi_auc = metrics.roc_auc_score(corrects.cpu().detach().numpy(), epistemic_confidence_calibraton_scores.cpu().detach().numpy())

        unc = {'ale_aupr': ale_aupr, 'ale_auc': ale_auc, 'epi_aupr': epi_aupr, 'epi_auc': epi_auc, 'Brier_score': Brier_score}

        return unc, accuracy, f1

    def ood_detection(self, alpha, ood_alpha):
        p = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        aleatoric_id_scores = p.max(-1)[0]
        epistemic_id_scores = alpha.sum(-1)

        ood_p = ood_alpha / torch.sum(ood_alpha, dim=-1, keepdim=True)
        aleatoric_ood_scores = ood_p.max(-1)[0]
        epistemic_ood_scores = ood_alpha.sum(-1)
        # epistemic_ood_scores = torch.ones(ood_alpha.size(0)).to(self.device)

        corrects = torch.cat([torch.ones(alpha.size(0)), torch.zeros(ood_alpha.size(0))], axis=0).long().to(self.device)
        ale_scores = torch.cat([aleatoric_id_scores, aleatoric_ood_scores], axis=0)
        epi_scores = torch.cat([epistemic_id_scores, epistemic_ood_scores], axis=0)

        ale_scores = F.normalize(ale_scores, p=1, dim=-1)
        epi_scores = F.normalize(epi_scores, p=1, dim=-1)

        ale_ood_aupr = self.aupr(ale_scores, corrects)
        ale_ood_auc = self.auc(ale_scores, corrects)
        epi_ood_aupr = self.aupr(epi_scores, corrects)
        epi_ood_auc = self.auc(epi_scores, corrects)

        # ale_ood_aupr = metrics.average_precision_score(corrects.cpu().detach().numpy(), ale_scores.cpu().detach().numpy())
        # ale_ood_auc = metrics.roc_auc_score(corrects.cpu().detach().numpy(), ale_scores.cpu().detach().numpy())
        # epi_ood_aupr = metrics.average_precision_score(corrects.cpu().detach().numpy(), epi_scores.cpu().detach().numpy())
        # epi_ood_auc = metrics.roc_auc_score(corrects.cpu().detach().numpy(), epi_scores.cpu().detach().numpy())


        unc_ood = {'ale_ood_aupr': ale_ood_aupr, 'ale_ood_auc': ale_ood_auc, 'epi_ood_aupr': epi_ood_aupr, 'epi_ood_auc': epi_ood_auc}

        return unc_ood



    def on_validation_epoch_end(self):
        # current_val_accuracy = sum(x['val_accuracy'] for x in self.validation_outputs) / len(self.validation_outputs)

        retrieve_y_pred = [x['y_pred'] for x in self.validation_outputs]
        retrieve_y_soft_pred = [x['soft_pred'] for x in self.validation_outputs]
        retrieve_y_true = [x['y_true'] for x in self.validation_outputs]
        retrieve_alpha = [x['alpha'] for x in self.validation_outputs]


        unc, current_val_accuracy, current_val_f1 = self._compute_metrics(torch.cat(retrieve_y_pred), 
                                                                        torch.cat(retrieve_y_soft_pred), 
                                                                        torch.cat(retrieve_y_true),
                                                                        torch.cat(retrieve_alpha)
                                                                        )
        

        if self.args.OOD:
            ood_X = self.args.ood_X.to(self.device)
            val_size = torch.cat(retrieve_y_pred).size(0)

            # ood_X = torch.randn_like(ood_X)

            if self.args.OOD_dataset == 'HCP':

                step = ood_X.size(-1) // 200
                indices = torch.arange(0, ood_X.size(-1), step)[:200]
                ood_X = ood_X[:, :, indices]

                _indices = torch.randperm(ood_X.size(0))[:val_size]
                ood_X = ood_X[_indices]
            elif self.args.OOD_dataset == 'F1000':
                _indices = torch.randperm(ood_X.size(0))[:val_size]
                ood_X = ood_X[_indices]

            elif self.args.OOD_dataset == 'OASIS':
                # step = ood_X.size(-1) // 164
                # indices = torch.arange(0, ood_X.size(-1), step)[:164]
                # ood_X = ood_X[:, :, indices]

                _indices = torch.randperm(ood_X.size(0))[:val_size]
                ood_X = ood_X[_indices]

            ood_alpha, _ = self.model(ood_X, return_output='soft')
            ood_unc = self.ood_detection(torch.cat(retrieve_alpha), ood_alpha)

        if self.args.perturbation:
            perturbation_X = self.args.perturbation_X.to(self.device)
            val_size = torch.cat(retrieve_y_pred).size(0)

            perturbation_alpha, _ = self.model(perturbation_X, return_output='soft')
            perturbation_unc = self.ood_detection(torch.cat(retrieve_alpha), perturbation_alpha)
        




        self.log('val_accuracy', current_val_accuracy, prog_bar=True, on_epoch=True,sync_dist=True)
        self.log('ale_aupr', unc["ale_aupr"], prog_bar=True, on_epoch=True,sync_dist=True)
        # self.log('ale_auc', unc["ale_auc"], prog_bar=True, on_epoch=True,sync_dist=True)
        self.log('epi_aupr', unc["epi_aupr"], prog_bar=True, on_epoch=True,sync_dist=True)
        # self.log('epi_auc', unc["epi_auc"], prog_bar=True, on_epoch=True,sync_dist=True)
        self.log('val_f1', current_val_f1, prog_bar=True, on_epoch=True,sync_dist=True)
        self.log('Brier_score', unc["Brier_score"], prog_bar=True, on_epoch=True,sync_dist=True)

        if self.args.OOD:
            self.log('ale_ood_aupr', ood_unc["ale_ood_aupr"], prog_bar=True, on_epoch=True,sync_dist=True)
            # self.log('ale_ood_auc', ood_unc["ale_ood_auc"], prog_bar=True, on_epoch=True,sync_dist=True)
            self.log('epi_ood_aupr', ood_unc["epi_ood_aupr"], prog_bar=True, on_epoch=True,sync_dist=True)
            # self.log('epi_ood_auc', ood_unc["epi_ood_auc"], prog_bar=True, on_epoch=True,sync_dist=True)

        if self.args.perturbation:
            self.log('ale_pert_aupr', perturbation_unc["ale_ood_aupr"], prog_bar=True, on_epoch=True,sync_dist=True)
            # self.log('ale_pert_auc', perturbation_unc["ale_ood_auc"], prog_bar=True, on_epoch=True,sync_dist=True)
            self.log('epi_pert_aupr', perturbation_unc["epi_ood_aupr"], prog_bar=True, on_epoch=True,sync_dist=True)
            # self.log('epi_pert_auc', perturbation_unc["epi_ood_auc"], prog_bar=True, on_epoch=True,sync_dist=True)


    def set_checkpoint_callback(self):
        if self.args.target in self.args.targets:
            return ModelCheckpoint(monitor='val_accuracy', mode='max', dirpath='saved_models', 
                                   save_top_k=1, filename='{epoch}, {val_accuracy:.04f}',every_n_epochs=1), \
            DefineMetricCallback_cls()


    def configure_optimizers(self):
        # Define optimizer and optionally learning rate schedulers
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.args.factor, patience=self.args.patience, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_accuracy',  # Replace with your validation metric
                'interval': 'epoch',
                'frequency': 1,
            }
        }
    

    def CE_loss(self, soft_output_pred, soft_output):
        CE_loss = - torch.sum(soft_output.squeeze() * torch.log(soft_output_pred))

        return CE_loss


    def UCE_loss(self, alpha, soft_output):
        alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.out_dim)
        entropy_reg = Dirichlet(alpha).entropy()
        UCE_loss = torch.sum(soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))) - self.regr * torch.sum(entropy_reg)

        return UCE_loss
    

    def predict(self, p):
        output_pred = torch.max(p, dim=-1)[1]
        return output_pred



class PosteriorNetwork(nn.Module):
    def __init__(self, N,  # Count of data from each class in training set. list of ints
                 args,
                 input_dims,  # Input dimension. list of ints
                 output_dim,  # Output dimension. int
                 hidden_dims,  # Hidden dimensions. list of ints
                 device,
                 kernel_dim=None,  # Kernel dimension if conv encoder_type. int
                 latent_dim=10,  # Latent dimension. int
                 encoder_type='linear',  # Encoder encoder_type name. int
                 k_lipschitz=None,  # Lipschitz constant. float or None (if no lipschitz)
                 no_density=False,  # Use density estimation or not. boolean
                 density_type='radial_flow',  # Density type. string
                 n_density=8,  # Number of density components. int
                 budget_function='id',  # Budget function name applied on class count. name
                 batch_size=64,  # Batch size. int
                 lr=1e-3,  # Learning rate. float
                 regr=1e-5,  # Regularization factor in Bayesian loss. float
                 seed=123
                ):  # Random seed for init. int
        super().__init__()

        torch.cuda.manual_seed(seed)
        self.args = args
        # torch.set_default_tensor_type(torch.DoubleTensor)
        self.encoder_type = encoder_type
        self.num_nodes = 85
        self.device = device
        self.ROI_level = self.args.ROI_level
        # encoder_type parameters
        self.input_dims, self.output_dim, self.hidden_dims, self.kernel_dim, self.latent_dim = input_dims, output_dim, hidden_dims, kernel_dim, latent_dim
        self.k_lipschitz = k_lipschitz
        self.no_density, self.density_type, self.n_density = no_density, density_type, n_density
        if budget_function in __budget_functions__:
            self.N, self.budget_function = __budget_functions__[budget_function](N), budget_function
        else:
            raise NotImplementedError
        # Training parameters
        self.batch_size, self.lr = batch_size, lr
        self.regr =  regr

        # Encoder -- Feature selection
        if encoder_type == 'linear':
            self.sequential = linear_sequential(input_dims=self.input_dims,
                                                hidden_dims=self.hidden_dims,
                                                output_dim=self.latent_dim,
                                                k_lipschitz=self.k_lipschitz)
            
        elif encoder_type == 'GCN':
            self.sequential = GCN(input_dim=self.input_dims, hidden_dims=self.hidden_dims, output_dim=self.latent_dim)

        elif encoder_type == 'Transformer':
            self.sequential = Transformer(input_dim=self.input_dims, hidden_dims=self.hidden_dims, output_dim=self.latent_dim)

        else:
            raise NotImplementedError
        if self.ROI_level:
            self.batch_norm = nn.BatchNorm1d(num_features=self.num_nodes)
        else:
            self.batch_norm = nn.BatchNorm1d(num_features=self.latent_dim)
        self.linear_classifier = linear_sequential(input_dims=[self.latent_dim],  # Linear classifier for sequential training
                                                   hidden_dims=[self.hidden_dims[-1]],
                                                   output_dim=self.output_dim,
                                                   k_lipschitz=self.k_lipschitz)

        # Normalizing Flow -- Normalized density on latent space
        if self.density_type == 'planar_flow':
            self.density_estimation = nn.ModuleList([NormalizingFlowDensity(dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type, ROI_level=self.ROI_level, Attention=self.args.Attention) for c in range(self.output_dim)])
        elif self.density_type == 'radial_flow':
            self.density_estimation = nn.ModuleList([NormalizingFlowDensity(dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type, ROI_level=self.ROI_level, Attention=self.args.Attention) for c in range(self.output_dim)])
        elif self.density_type == 'batched_radial_flow':
            self.density_estimation = BatchedNormalizingFlowDensity(c=self.output_dim, dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type.replace('batched_', ''))
        elif self.density_type == 'iaf_flow':
            self.density_estimation = nn.ModuleList([NormalizingFlowDensity(dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type,ROI_level=self.ROI_level, Attention=self.args.Attention) for c in range(self.output_dim)])
        elif self.density_type == 'normal_mixture':
            self.density_estimation = nn.ModuleList([MixtureDensity(dim=self.latent_dim, n_components=n_density, mixture_type=self.density_type) for c in range(self.output_dim)])
        else:
            raise NotImplementedError
        self.softmax = nn.Softmax(dim=-1)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, input, return_output='hard'):
        batch_size = input.size(0)

        # if self.N.device != input.device:
        #     self.N = self.N.to(input.device)

        if self.budget_function == 'parametrized':
            N = self.N / self.N.sum()
        else:
            N = self.N

        # Forward
        if self.encoder_type == 'linear':
            if self.ROI_level:
                zk = self.sequential(input)
            else:
                zk = torch.mean(self.sequential(input), dim=1)

        elif self.encoder_type == 'GCN':
            data = self.process_batch(input)
            zk = self.sequential.forward(data)

        elif self.encoder_type == 'Transformer':
            if self.ROI_level:
                zk = self.sequential(input)
            else:
                zk = torch.mean(self.sequential(input), dim=1)

        if self.no_density:  # Ablated model without density estimation
            logits = self.linear_classifier(zk)
            alpha = torch.exp(logits)
            soft_output_pred = self.softmax(logits)
        else:  # Full model with density estimation
            if self.ROI_level:
                pass
            else:
                zk = self.batch_norm(zk)
            log_q_zk = torch.zeros((batch_size, self.output_dim)).to(zk.device.type)
            alpha = torch.zeros((batch_size, self.output_dim)).to(zk.device.type)

            if isinstance(self.density_estimation, nn.ModuleList):
                for c in range(self.output_dim):
                    log_p = self.density_estimation[c].log_prob(zk)
                    log_q_zk[:, c] = log_p
                    alpha[:, c] = 1. + (N[c] * torch.exp(log_q_zk[:, c]))
            else:
                log_q_zk = self.density_estimation.log_prob(zk)
                alpha = 1. + (N[:, None] * torch.exp(log_q_zk)).permute(1, 0)

            pass

            soft_output_pred = torch.nn.functional.normalize(alpha, p=1)
        output_pred = self.predict(soft_output_pred)



        if return_output == 'hard':
            return alpha, output_pred
        elif return_output == 'soft':
            return alpha, soft_output_pred
        elif return_output == 'alpha':
            return alpha
        elif return_output == 'latent':
            return zk
        else:
            raise AssertionError

    def CE_loss(self, soft_output_pred, soft_output):
        CE_loss = - torch.sum(soft_output.squeeze() * torch.log(soft_output_pred))

        return CE_loss

    def UCE_loss(self, alpha, soft_output):
        alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.output_dim)
        entropy_reg = Dirichlet(alpha).entropy()
        UCE_loss = torch.sum(soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))) - self.regr * torch.sum(entropy_reg)

        return UCE_loss


    def predict(self, p):
        output_pred = torch.max(p, dim=-1)[1]
        return output_pred
    
    def process_batch(self, batch_adj_matrices):
        # Number of graphs in the batch
        num_graphs = batch_adj_matrices.size(0)
        num_nodes = batch_adj_matrices.size(1)

        # Node features for all graphs (stacked identity matrices)
        x = torch.eye(num_nodes).repeat(num_graphs, 1).to(self.device)

        # Adjust edge indices for each graph, concatenate them, and collect edge attributes
        edge_indices = []
        edge_attrs = []  # List to store edge attributes
        for idx in range(num_graphs):
            adj_matrix = batch_adj_matrices[idx]
            # Extract indices and values
            nz_indices = torch.nonzero(adj_matrix, as_tuple=False)
            nz_values = adj_matrix[nz_indices[:, 0], nz_indices[:, 1]]

            edge_index = nz_indices.t().contiguous()
            edge_index += idx * num_nodes  # Adjust for the node offset
            edge_indices.append(edge_index)
            edge_attrs.append(torch.abs(nz_values))

        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs)  # Concatenate edge attributes

        # Create batch vector
        batch = torch.arange(num_graphs).repeat_interleave(num_nodes).to(self.device)

        # Create a single Batch object including edge attributes
        data = Batch(batch=batch, x=x, edge_index=edge_index, edge_weight=edge_attr)

        return data
