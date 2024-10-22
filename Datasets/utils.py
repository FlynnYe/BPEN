import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import torch
import numpy as np
import pandas as pd
import os,sys
from os.path import join
from scipy.io import loadmat
import mat73





class _F1000_dataset(Dataset):
    def __init__(self, args, gdata_path=None, label_path=None,
                               is_train=True):
        super(_F1000_dataset, self).__init__()
        self.args = args
        self.label_path = label_path
        self.gdata_path = gdata_path

        self.pdlabels = pd.read_csv(self.label_path, header=None)


        self.task = 'classification'

        ptarget = self.pdlabels[0].apply(lambda x: 0 if x == 'Female' else 1)
        target = torch.from_numpy(ptarget.values.reshape(-1)).long()
                



        data = loadmat(self.gdata_path)
        data_np = np.transpose(np.array(data['NW']))
        data_tensor = torch.from_numpy(data_np).float()

        self.data = data_tensor
        self.target = target
        self.is_train = is_train


    def train_status(self, status):
        if status:
            self.is_train = True
        else:
            self.is_train = False


    def __getitem__(self, index):

        return self.target[index], self.data[index], torch.empty(1)
    
    def __len__(self):
        return len(self.data)
    
def make_apoe_label(row):
    if (row[-1],row[-2]) == (0,0):
        apoe_label = 0
    elif (row[-1],row[-2]) == (0,1):
        apoe_label = 1
    elif (row[-1], row[-2]) == (1, 0):
        apoe_label = 2
    elif (row[-1], row[-2]) == (1, 1):
        apoe_label = 3
    row['APOE_labels'] = apoe_label

    return row

class _OASIS_dataset(Dataset):
    def __init__(self, args, gdatadir=None, label_path=None, data_basedir=None , is_train=True):
        super(_OASIS_dataset, self).__init__()
        self.args = args
        self.data_basedir = data_basedir
        self.label_path = label_path
        self.graph_dir = gdatadir
        original_labels_dir = 'Data/OASIS_Data/oasis_demos_1326_subs.xlsx'
        df = pd.read_excel(original_labels_dir, usecols=['MR ID', 'Primary Diagnosis', 'APOE'])  # Cognitively normal
        df['disease_labels'] = df['Primary Diagnosis'].apply(lambda x: 0 if x == 'Cognitively normal' else 1)
        df['APOE_transform'] = df['APOE'].apply(lambda x: 1 if '4' in list(str(x)) else 0)
        df = df.apply(make_apoe_label, axis=1)
        self.disease_labels_dict = {str(k): v for k, v in zip(list(df['MR ID']), list(df['disease_labels']))}
        self.apoe_labels_dict = {str(k): v for k, v in zip(list(df['MR ID']), list(df['APOE_labels']))}


        seqfiles = os.listdir(self.data_basedir)
        self.seqs_dict = {}
        for i in range(len(seqfiles)):
            seqdir = os.path.join(self.data_basedir, seqfiles[i])
            if seqfiles[i][-4:] == '.npy':
                subject = seqfiles[i][:-4]
                seq = np.load(seqdir).T[:, :self.args.sample_size]
                self.seqs_dict[subject] = seq

        labelfiles = os.listdir(self.label_path)
        self.labels_dict = {}
        for i in range(len(labelfiles)):
            labeldir = os.path.join(self.label_path, labelfiles[i])
            if labelfiles[i][-4:] == '.npy':
                subject = labelfiles[i][:-4]
                label = np.load(labeldir)
                self.labels_dict[subject] = label

        graphfiles = os.listdir(self.graph_dir)
        self.graphs_dict = {}
        for i in range(len(graphfiles)):
            dir = os.path.join(self.graph_dir, graphfiles[i])
            if graphfiles[i][-4:] == '.npy':
                subject = graphfiles[i][:-4]
                gra = np.load(dir)
                self.graphs_dict[subject] = gra


        names = []
        adjfiles = os.listdir(self.graph_dir)
        for file in adjfiles:
            if file[-4:] == '.npy':
                names.append(file)

        self.names = names


        self.args = args
        self.is_train = is_train

        if self.args.OASIS_type == 'NC':
            self.names = [name for name in self.names if self.disease_labels_dict[name[:-4]] == 0]
        elif self.args.OASIS_type == 'not_NC':
            self.names = [name for name in self.names if self.disease_labels_dict[name[:-4]] == 1]



    def __getitem__(self, index):
        subject = self.names[index][:-4]
        # classification_target = self.labels_dict[subject]
        # classification_target = self.apoe_labels_dict[subject]
        classification_target = self.disease_labels_dict[subject]
        seq = self.seqs_dict[subject]


        adj = self.graphs_dict[subject]

        return torch.tensor(classification_target).long(), torch.from_numpy(seq).float(), torch.from_numpy(adj).float()

    def __len__(self):
        return len(self.names)

class _HCP_dataset(Dataset):
    def __init__(self, args, gdatadir=None, label_path=None, data_basedir=None , is_train=True):
        super(_HCP_dataset, self).__init__()
        self.args = args
        self.data_basedir = data_basedir
        self.label_path = label_path
        self.gdatadir = gdatadir

        self.pdlabels = pd.read_csv(self.label_path)
        pdsubjedt = self.pdlabels['Subject']

        pgender = self.pdlabels['Gender']
        self.task = 'classification'

        ptarget = pgender
        ptarget = ptarget.apply(lambda x: 0 if x == 'F' else 1)
            




        _target_dict = {str(k): v for k, v in zip(list(pdsubjedt), list(ptarget))}
        target_dict = {}
        for k, v in _target_dict.items():
            if not np.isnan(v):
                target_dict[k] = v

        sequence_dir = self.data_basedir + '/QCed_sequence'
        seqfiles = os.listdir(sequence_dir)
        self.seqs_dict = {}
        for i in range(len(seqfiles)):
            seqdir = os.path.join(sequence_dir, seqfiles[i])
            if seqfiles[i][-4:] == '.npy':
                subject = seqfiles[i][:-4]
                seq = np.load(seqdir).T[:, :self.args.sample_size]
                self.seqs_dict[subject] = seq
        self.filtered_target_dict = {}
        for k, v in self.seqs_dict.items():
            if k not in list(target_dict.keys()):
                continue

            elif v.shape[1] == self.args.sample_size and v.shape[0] == 82:
                self.filtered_target_dict[k] = target_dict[k]
        # self.subject = list(self.filtered_target_dict.keys())

        # Load Graph Data
        graph_dir = join(self.gdatadir)
        names = []
        adjfiles = os.listdir(graph_dir)
        for file in adjfiles:
            if file[-4:] == '.npy':
                names.append(file)

        self.graph_dir = graph_dir
        self.is_train = is_train

        # names = self.names
        self.names = []
        for name in names:
            if name[:-4] not in list(self.filtered_target_dict.keys()):
                continue
            else:
                self.names.append(name)
        # self.normalize_target()

    def train_status(self, status):
        if status:
            self.is_train = True
        else:
            self.is_train = False

    def normalize_target(self):
        targets = np.array(self.labels)  # Convert list to NumPy array
        min_target = np.min(targets)
        max_target = np.max(targets)

        # Assuming you have determined the scale and shift values
        scale = 1 / (max_target - min_target)
        shift = -min_target / (max_target - min_target)

        # Directly manipulate the targets
        normalized_targets = scale * targets + shift

        for name in self.names:
            self.filtered_target_dict[name[:-4]] = normalized_targets[self.names.index(name)]


        # denormalized_targets = normalized_targets * (max_target - min_target) + min_target

    @property
    def labels(self):
        return [self.filtered_target_dict[name[:-4]] for name in self.names]

    def __getitem__(self, index):
        subject = self.names[index][:-4]
        target = self.filtered_target_dict[subject]

        seq = self.seqs_dict[subject]
        seq = torch.from_numpy(seq).float()

        adj = np.load(join(self.graph_dir, self.names[index]))

        adj = torch.from_numpy(adj)
        if self.task == 'classification':
            target = torch.tensor(target).long()
        else:
            target = torch.tensor(target).float()

        return target, seq, adj
    
    def __len__(self):
        return len(self.names)
