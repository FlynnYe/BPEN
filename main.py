from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from Models.ADNI.BrainPostNet import BrainPostNet as Model
from Datasets.ADNI import ADNI
from Datasets.ADNI_BS_5_fold import ADNI_BS
import torch
import pytorch_lightning as pl
import argparse
import os
import time
from Models.utils import PrintValAccuracyCallback
import numpy as np


'''
NC without depression vs. NC with depression    -   NC_ND_vs_NC_D
NC without depression vs. MCI without depression    -   NC_ND_vs_MCI_ND
NC without depression vs. MCI with depression   -   NC_ND_vs_MCI_D
NC with depression vs. MCI with depression   -   NC_D_vs_MCI_D
MCI without depression vs. MCI with depression  -   MCI_ND_vs_MCI_D
'''
targets = ['NC_ND_vs_NC_D', 'NC_ND_vs_MCI_ND', 'NC_D_vs_MCI_ND', 'NC_D_vs_MCI_D', 'MCI_ND_vs_MCI_D'] 

def main():

    parser = argparse.ArgumentParser(description="ADNI/ADNID experiment")
    parser.add_argument('--batch_size', type=int, default=256)  # change batch size
    parser.add_argument('--max_epoch', type=int, default=400)  # 200
    parser.add_argument('--sample_size', type=int, default=1200)
    parser.add_argument('--in_feature_dim', type=int, default=164)  # 200, 164
    parser.add_argument('--hidden_dim', type=int, default=1024) # 1024
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--n_density', type=int, default=8) # 8

    parser.add_argument('--lr', type=float, default=2e-2)  # 1e-3 5e-2 2e-2
    parser.add_argument('--factor', type=float, default=0.5)  
    parser.add_argument('--patience', type=int, default=30)  # 20

    parser.add_argument('--encoder_type', type=str, default='linear', choices=['linear', 'GCN'])
    parser.add_argument('--flow_type', type=str, default='radial_flow', choices=['radial_flow', 'iaf_flow']) # radial_flow

    parser.add_argument('--targets', type=list, default=targets)
    parser.add_argument('--save', type=bool, default=False) 
    parser.add_argument('--task', type=int, default=1, choices=[1,2,3,4,5]) 

    parser.add_argument('--BOLD_signal', type=bool, default=True) 
    parser.add_argument('--Attention', type=bool, default=True) 
    parser.add_argument('--loss_name', type=str, default='UCE', choices=['UCE','CE']) 
    parser.add_argument('--ROI_level', type=bool, default=True) 
    parser.add_argument('--OOD', type=bool, default=True) 
    parser.add_argument('--OOD_dataset', type=str, default='OASIS', choices=['HCP', 'OASIS','F1000']) 
    parser.add_argument('--perturbation', type=bool, default=True) 
    parser.add_argument('--Stratified', type=bool, default=True)

    parser.add_argument('--OASIS_type', type=str, default='mix', choices=['mix', 'NC','not_NC']) 

    parser.add_argument('--fold', type=int, default=1, choices=[0,1,2,3,4], help='Fold index for cross-validation (0-4)')

    args = parser.parse_args()
    print(torch.cuda.is_available())
    print(torch.cuda.device_count(), torch.cuda.get_device_name())

    args.target = targets[args.task - 1]

    pl.seed_everything(42) 
    seed = torch.Generator().manual_seed(42) 
    np_seed = np.random.seed(42)

    if args.BOLD_signal:
        dm = ADNI_BS(args=args, seed=seed, np_seed = np_seed, fold=args.fold, n_folds=5)
    else:
        dm = ADNI(args=args, seed=seed,np_seed = np_seed, fold=args.fold, n_folds=5)
    dm.prepare_data()
    dm.setup()

    if args.OOD:
        dm.setup_ood()
        args.ood_X = dm.get_ood()

    if args.perturbation:
        dm.setup_perturbation()
        args.perturbation_X = torch.from_numpy(dm.get_perturbation()).float()

    model = Model(args=args)
    model_name = model.__class__.__name__
    log_name = (
        f"ADNI_{model_name}_{args.target}_fold_{args.fold}_seed_42_batch_{args.batch_size}_"
        f"epoch_{args.max_epoch}_lr_{args.lr}_loss_{args.loss_name}_BOLD_{args.BOLD_signal}_"
        f"latent_{args.latent_dim}_hidden_{args.hidden_dim}_encoder_{args.encoder_type}"
    )
    wandb_logger = WandbLogger(log_model=False, save_dir = 'saved_models', name=log_name) # log_model="all" / False / True
    checkpoint_callback, defineMetricCallback = model.set_checkpoint_callback()

    trainer = Trainer(logger=wandb_logger, max_epochs=args.max_epoch, callbacks=[checkpoint_callback, defineMetricCallback], 
                      strategy='ddp_find_unused_parameters_true', num_sanity_val_steps=0, log_every_n_steps=10)    # strategy='ddp_find_unused_parameters_true'

    trainer.fit(model, datamodule=dm)

    best_module = Model.load_from_checkpoint(
        checkpoint_callback.best_model_path, args=args
    )

    trainer.validate(model=best_module, dataloaders=dm.val_dataloader())

    best_module.plot_and_metrics(dm.val_dataloader())



if __name__ == '__main__':
    main()
