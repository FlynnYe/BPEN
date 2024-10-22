import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import torch
import numpy as np
import pandas as pd
import os,sys
from os.path import join
from scipy.io import loadmat
import mat73
from .utils import _HCP_dataset, _F1000_dataset, _OASIS_dataset
from sklearn.model_selection import StratifiedKFold



class ADNI_BS(pl.LightningDataModule):
    def __init__(self, args, seed, np_seed, fold=0, n_folds=5):
        super().__init__()
        self.args = args
        self.seed = seed
        self.np_seed = np_seed
        self.fold = fold
        self.n_folds = n_folds

    def prepare_data(self):
        if self.args.target == 'NC_ND_vs_MCI_ND':
            self.label_path = 'Data/ADNI/OASIS_ADNI3_Combined_09052022.xlsx' 
            self.gdata_path = 'Data/ADNI/ADNI_updated_020823.mat'
        elif self.args.target == 'NC_D_vs_MCI_D':
            self.label_path = 'Data/ADNI/OASIS_ADNI3_Combined_09052022.xlsx' 
            self.gdata_path = 'Data/ADNI/ADNI_D_updated_020623.mat'
        elif self.args.target in ['NC_ND_vs_NC_D', 'NC_ND_vs_MCI_D', 'NC_D_vs_MCI_ND', 'MCI_ND_vs_MCI_D']:
            self.label_path = 'Data/ADNI/OASIS_ADNI3_Combined_09052022.xlsx' 
            self.gdata_path = ['Data/ADNI/ADNI_updated_020823.mat', 'Data/ADNI/ADNI_D_updated_020623.mat']

        if self.args.OOD:
            if self.args.OOD_dataset == 'HCP':
                self.hcp_graph_data_path = 'Data/adj/structure_hcp82'
                self.hcp_base_path = 'Data/HCP82_fMRI_seq'
                self.hcp_sex_label_path = 'Data/HCP82_fMRI_seq/annontations.csv'
            elif self.args.OOD_dataset == 'F1000':
                self.f1000_label_path = 'Data/F1000/F1000_subject_sex.csv' 
                self.f1000_gdata_path = 'Data/F1000/F1000_NW.mat'
            elif self.args.OOD_dataset == 'OASIS':
                self.oasis_graph_data_path = 'Data/OASIS_Data/original_data/graph_normsc_oas'
                self.oasis_base_path = 'Data/OASIS_Data/original_data/BOLD_ts_oas'
                self.oasis_sex_label_path = 'Data/OASIS_Data/labels/gender'


    def setup(self, stage=None):

        data_set = _ADNI_BS_dataset(args=self.args, gdata_path=self.gdata_path, label_path=self.label_path,
                               is_train=True)
        self.data_set = data_set

        # Generate splits using StratifiedKFold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.np_seed)

        # Obtain train and validation indices for the current fold
        splits = list(skf.split(self.get_X(), self.get_y()))
        train_idx, val_idx = splits[self.fold]

        self.train_dataset = Subset(data_set, train_idx)
        self.val_dataset = Subset(data_set, val_idx)

        # Mark train and validation subsets
        self.train_dataset.dataset.train_status = True
        self.val_dataset.dataset.train_status = False



    def get_X(self):
        X_loader = DataLoader(self.data_set, batch_size=self.args.batch_size,shuffle=False)
        seqs = []
        for batch in X_loader:
            target, seq, adj = batch
            seqs.append(seq)
        X = np.concatenate(seqs)
        return X
    
    def get_y(self):
        y_loader = DataLoader(self.data_set, batch_size=self.args.batch_size,shuffle=False)
        labels = []
        for batch in y_loader:
            target, seq, adj = batch
            labels.append(target)
        y = np.concatenate(labels)
        return y

    def setup_ood(self, stage=None):
        if self.args.OOD_dataset == 'HCP':
            # Load and split the dataset into train, val, and test sets
            data_set = _HCP_dataset(args=self.args, data_basedir=self.hcp_base_path, label_path=self.hcp_sex_label_path, gdatadir=self.hcp_graph_data_path,
                                is_train=True)
            train, val = _dataset_train_test_split(
                data_set, train_size=0.8, generator=self.seed
            )
            train.train_status = True
            val.train_status = False
            # self.train_dataset = train
            # self.val_dataset = val
            self.ood_set = val
        elif self.args.OOD_dataset == 'F1000':
            # Load and split the dataset into train, val, and test sets
            data_set = _F1000_dataset(args=self.args, gdata_path=self.f1000_gdata_path, label_path=self.f1000_label_path,
                                is_train=True)
            train, val = _dataset_train_test_split(
                data_set, train_size=0.8, generator=self.seed
            )
            train.train_status = True
            val.train_status = False
            # self.train_dataset = train
            # self.val_dataset = val
            self.ood_set = val  
        elif self.args.OOD_dataset == 'OASIS':
            # Load and split the dataset into train, val, and test sets
            data_set = _OASIS_dataset(args=self.args, data_basedir=self.oasis_base_path, label_path=self.oasis_sex_label_path, gdatadir=self.oasis_graph_data_path,
                                is_train=True)
            train, val = _dataset_train_test_split(
                data_set, train_size=0.8, generator=self.seed
            )
            train.train_status = True
            val.train_status = False
            # self.train_dataset = train
            # self.val_dataset = val
            self.ood_set = val


    def get_ood(self):
        ood_loader = DataLoader(self.ood_set, batch_size=self.args.batch_size,shuffle=False)
        seqs = []
        for batch in ood_loader:
            target, seq, adj = batch
            seqs.append(seq)
        
        # return torch.stack(seqs).squeeze(0)
        return torch.cat(seqs, dim=0)
    
    def setup_perturbation(self):
        self.perturbation_set = self.val_dataset

    def get_perturbation(self,mean=0, sigma=0.1):
        perturbation_loader = DataLoader(self.perturbation_set, batch_size=self.args.batch_size,shuffle=False)
        seqs = []
        for batch in perturbation_loader:
            target, seq, adj = batch
            seqs.append(seq)
        X = np.concatenate(seqs)
        

        noise_level = sigma 
        noise = np.random.normal(mean, noise_level, X.shape)
        noisy_X = X + noise

        return noisy_X


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size,shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_set, batch_size=1,shuffle=False)
    
    def all_dataloader(self):
        return DataLoader(self.data_set, batch_size=1,shuffle=False)



class _ADNI_BS_dataset(Dataset):
    def __init__(self, args, gdata_path=None, label_path=None,
                               is_train=True):
        super(_ADNI_BS_dataset, self).__init__()
        self.args = args
        self.label_path = label_path
        self.gdata_path = gdata_path

        # self.pdlabels = pd.read_csv(self.label_path, header=None)
        # data = mat73.loadmat(self.gdata_path)

        if self.args.target == 'NC_ND_vs_MCI_ND':
            data = mat73.loadmat(self.gdata_path)
            data_np = np.transpose(data['ts_adni'][::5])
        elif self.args.target == 'NC_D_vs_MCI_D':
            data = mat73.loadmat(self.gdata_path)
            data_np = np.transpose(data['ts_adniD'])
            data_np_filtered = np.delete(data_np, 86, axis=0)
            data_np = data_np_filtered

        elif self.args.target in ['NC_ND_vs_MCI_D', 'NC_D_vs_MCI_ND', 'MCI_ND_vs_MCI_D', 'NC_ND_vs_NC_D']:
            adni_data_path, adni_D_data_path =self.gdata_path
            data_adni = mat73.loadmat(adni_data_path)
            data_adniD = mat73.loadmat(adni_D_data_path)
            data_np_adni = np.transpose(data_adni['ts_adni'][::5])
            data_np_adniD = np.transpose(data_adniD['ts_adniD'])

            # filter out the 86th row of data_np_adniD
            data_np_adniD_filtered = np.delete(data_np_adniD, 86, axis=0)
            data_np_adniD = data_np_adniD_filtered
            



        if self.args.target == 'NC_ND_vs_MCI_ND':
            df = pd.read_excel(self.label_path, sheet_name='ADNI Demographics')
            mask = df['Group'].isin(['CN', 'LMCI', 'EMCI', 'MCI'])
            df = df[mask]
            data_np = data_np[mask]
            self.AD_labels = df['Group'].apply(lambda x: 0 if x == 'CN' else 1).values

        elif self.args.target == 'NC_D_vs_MCI_D':
            df = pd.read_excel(self.label_path, sheet_name='ADNI-D Clinical Measures')
            df = df[:-2]
            mask = df['DX_v0'].isin(['NL', 'MCI'])
            df = df[mask]
            data_np = data_np[mask]
            self.AD_labels = df['DX_v0'].apply(lambda x: 0 if x == 'NL' else 1).values

        elif self.args.target == 'NC_ND_vs_NC_D':
            df_adni = pd.read_excel(self.label_path, sheet_name='ADNI Demographics')
            mask_adni = df_adni['Group'].isin(['CN'])
            df_adni = df_adni[mask_adni]
            data_np_adni = data_np_adni[mask_adni]

            df_adniD = pd.read_excel(self.label_path, sheet_name='ADNI-D Clinical Measures')
            df_adniD = df_adniD[:-2]
            mask_adniD = df_adniD['DX_v0'].isin(['NL'])
            df_adniD = df_adniD[mask_adniD]
            data_np_adniD = data_np_adniD[mask_adniD]

            self.AD_labels = np.concatenate((np.zeros(len(data_np_adni)), np.ones(len(data_np_adniD))), axis=0)
            data_np = np.concatenate((data_np_adni, data_np_adniD), axis=0)

        elif self.args.target == 'NC_ND_vs_MCI_D':
            df_adni = pd.read_excel(self.label_path, sheet_name='ADNI Demographics')
            mask_adni = df_adni['Group'].isin(['CN'])
            df_adni = df_adni[mask_adni]
            data_np_adni = data_np_adni[mask_adni]

            df_adniD = pd.read_excel(self.label_path, sheet_name='ADNI-D Clinical Measures')
            df_adniD = df_adniD[:-2]
            mask_adniD = df_adniD['DX_v0'].isin(['MCI'])
            df_adniD = df_adniD[mask_adniD]
            data_np_adniD = data_np_adniD[mask_adniD]

            self.AD_labels = np.concatenate((np.zeros(len(data_np_adni)), np.ones(len(data_np_adniD))), axis=0)
            data_np = np.concatenate((data_np_adni, data_np_adniD), axis=0)

        elif self.args.target == 'NC_D_vs_MCI_ND':
            df_adni = pd.read_excel(self.label_path, sheet_name='ADNI Demographics')
            mask_adni = df_adni['Group'].isin(['LMCI', 'EMCI', 'MCI'])
            df_adni = df_adni[mask_adni]
            data_np_adni = data_np_adni[mask_adni]

            df_adniD = pd.read_excel(self.label_path, sheet_name='ADNI-D Clinical Measures')
            df_adniD = df_adniD[:-2]
            mask_adniD = df_adniD['DX_v0'].isin(['NL'])
            df_adniD = df_adniD[mask_adniD]
            data_np_adniD = data_np_adniD[mask_adniD]

            self.AD_labels = np.concatenate((np.ones(len(data_np_adni)), np.zeros(len(data_np_adniD))), axis=0)
            data_np = np.concatenate((data_np_adni, data_np_adniD), axis=0)

        elif self.args.target == 'MCI_ND_vs_MCI_D':
            df_adni = pd.read_excel(self.label_path, sheet_name='ADNI Demographics')
            mask_adni = df_adni['Group'].isin(['LMCI', 'EMCI', 'MCI'])
            df_adni = df_adni[mask_adni]
            data_np_adni = data_np_adni[mask_adni]

            df_adniD = pd.read_excel(self.label_path, sheet_name='ADNI-D Clinical Measures')
            df_adniD = df_adniD[:-2]
            mask_adniD = df_adniD['DX_v0'].isin(['MCI'])
            df_adniD = df_adniD[mask_adniD]
            data_np_adniD = data_np_adniD[mask_adniD]

            self.AD_labels = np.concatenate((np.zeros(len(data_np_adni)), np.ones(len(data_np_adniD))), axis=0)
            data_np = np.concatenate((data_np_adni, data_np_adniD), axis=0)


        target = torch.from_numpy(self.AD_labels).long()
        if self.args.OOD_dataset == 'OASIS':
            total_elements = 200
            desired_elements = 164
            step_size = total_elements / desired_elements
            # Generate the indices to sample
            indices = np.arange(0, total_elements, step_size).astype(int)

            # Sample the array
            data_np = data_np[:, :, indices]


        data_tensor = torch.from_numpy(data_np).float()

        self.data = data_tensor
        self.target = target
        self.is_train = is_train


    def train_status(self, status):
        self.is_train = status


    def __getitem__(self, index):

        return self.target[index], self.data[index], torch.empty(1)
    
    def __len__(self):
        return len(self.data)



def _dataset_train_test_split(
    dataset, train_size: float, generator: torch.Generator,
):

    num_items = len(dataset)  
    num_train = round(num_items * train_size)
    permutation = torch.randperm(num_items, generator=generator)
    return (
        Subset(dataset, permutation[:num_train].tolist()),
        Subset(dataset, permutation[num_train:].tolist()),
    )
