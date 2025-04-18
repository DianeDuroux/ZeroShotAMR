import sys
sys.path.insert(0, "..")
sys.path.insert(0, "./")
sys.path.insert(0, "../..")
sys.path.insert(0, "data_split")
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
from os.path import join, exists
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from argparse import ArgumentParser
import json
from experiments.pl_experiment import Classifier_Experiment
import itertools

from data_split.data_utils import DataSplitter
from models.data_loaders import DrugResistanceDataset_Fingerprints, SampleEmbDataset, DrugResistanceDataset_Embeddings
from models.classifier import Residual_AMR_Classifier
import sys

import shap

TRAINING_SETUPS = list(itertools.product(['A', 'B', 'C', 'D'], ["random", "drug_species_zero_shot", "drugs_zero_shot"], np.arange(5)))

def main(args):
    config = vars(args)
    seed = args.seed
    # Setup output folders to save results
    output_folder = join(args.output_folder, args.experiment_group, args.experiment_name, str(args.seed))
    if not exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    results_folder = join(args.output_folder, args.experiment_group, args.experiment_name + "_results")
    if not exists(results_folder):
        os.makedirs(results_folder, exist_ok=True)

    experiment_folder = join(args.output_folder, args.experiment_group, args.experiment_name)
    if exists(join(results_folder, f"test_metrics_{args.seed}.json")):
        sys.exit(0)
    if not exists(experiment_folder):
        os.makedirs(experiment_folder, exist_ok=True)

    # Read data
    driams_long_table = pd.read_csv(args.driams_long_table)
    spectra_matrix = np.load(args.spectra_matrix).astype(float)
    drugs_df = pd.read_csv(args.drugs_df, index_col=0)
    driams_long_table = driams_long_table[driams_long_table["drug"].isin(drugs_df.index)]
    split_idsList=args.split_ids
    dataset_investigated=args.driams_dataset
    
    # Instantate data split
    dsplit = DataSplitter(driams_long_table, dataset=args.driams_dataset)
    samples_list = sorted(dsplit.long_table["sample_id"].unique())

    # Split selection for the different experiments.
    if args.split_type == "random":
        train_df, val_df, test_df = dsplit.random_train_val_test_split(val_size=0.1, test_size=0.2, random_state=args.seed)
    ### Added Diane
    elif args.split_type == "specific":
        train_df, val_df, test_df = dsplit.specific_train_val_test_split(split_idsList, dataset_investigated, driams_long_table)
    ###
    elif args.split_type =="drug_species_zero_shot":
        trainval_df, test_df = dsplit.combination_train_test_split(dsplit.long_table, test_size=0.2, random_state=args.seed)
        train_df, val_df = dsplit.baseline_train_test_split(trainval_df, test_size=0.2, random_state=args.seed)
    elif args.split_type =="drugs_zero_shot":
        drugs_list = sorted(dsplit.long_table["drug"].unique())
        if args.seed>=len(drugs_list):
            print("Drug index out of bound, exiting..\n\n")
            sys.exit(0)
        target_drug = drugs_list[args.seed]
        # target_drug = args.drug_name
        test_df, trainval_df = dsplit.drug_zero_shot_split(drug=target_drug)
        train_df, val_df = dsplit.baseline_train_test_split(trainval_df, test_size=0.2, random_state=args.seed)

    test_df.to_csv(join(output_folder, "test_set.csv"), index=False)

    if args.drug_emb_type=="fingerprint":
        train_dset = DrugResistanceDataset_Fingerprints(train_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
        val_dset = DrugResistanceDataset_Fingerprints(val_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
        test_dset = DrugResistanceDataset_Fingerprints(test_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
    elif args.drug_emb_type=="vae_embedding" or args.drug_emb_type=="gnn_embedding":
        train_dset = DrugResistanceDataset_Embeddings(train_df, spectra_matrix, drugs_df, samples_list)
        val_dset = DrugResistanceDataset_Embeddings(val_df, spectra_matrix, drugs_df, samples_list)
        test_dset = DrugResistanceDataset_Embeddings(test_df, spectra_matrix, drugs_df, samples_list)


    sorted_species = sorted(dsplit.long_table["species"].unique())
    idx2species = {i: s for i, s in enumerate(sorted_species)}
    species2idx = {s: i for i, s in idx2species.items()}

    config["n_unique_species"] = len(idx2species)
    del config["seed"]
    # Save configuration
    if not exists(join(experiment_folder, "config.json")):
        with open(join(experiment_folder, "config.json"), "w") as f:
            json.dump(config, f)
    if not exists(join(results_folder, "config.json")):
        with open(join(results_folder, "config.json"), "w") as f:
            json.dump(config, f)



    train_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(
        val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Instantiate model and pytorch lightning experiment
    model = Residual_AMR_Classifier(config)
    experiment = Classifier_Experiment(config, model)

    # Save summary of the model architecture
    if not exists(join(experiment_folder, "architecture.txt")):
        with open(join(experiment_folder, "architecture.txt"), "w") as f:
            f.write(model.__repr__())
    if not exists(join(results_folder, "architecture.txt")):
        with open(join(results_folder, "architecture.txt"), "w") as f:
            f.write(model.__repr__())


    # Setup training callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(output_folder, "checkpoints"),
                                          monitor="val_loss", filename="gst-{epoch:02d}-{val_loss:.4f}")
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", mode="min", patience=args.patience
    )
    callbacks = [checkpoint_callback, early_stopping_callback]

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=join(output_folder, "logs/"))

    # Train model
    print("Training..")
    trainer = pl.Trainer(devices="auto", accelerator="auto", 
        default_root_dir=output_folder, max_epochs=args.n_epochs,#, callbacks=callbacks,
                        #  logger=tb_logger, log_every_n_steps=3
                        # limit_train_batches=6, limit_val_batches=4, limit_test_batches=4
                         )
    trainer.fit(experiment, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    print("Training complete!")

    # Test model
    print("Testing..")
    test_results = trainer.test(ckpt_path="best", dataloaders=test_loader)
    with open(join(results_folder, "test_metrics_{}.json".format(seed)), "w") as f:
        json.dump(test_results[0], f, indent=2)
        
    background_loader = DataLoader(test_dset, batch_size=100, shuffle=True)
    background = next(iter(background_loader))
    X_background = torch.cat(background[:3], dim=1)

    
    test_fi_loader = DataLoader(test_dset, batch_size=len(test_dset), shuffle=False)
    fi_batch = next(iter(test_fi_loader))
    X_fi_batch = torch.cat(fi_batch[:3], dim=1)
    
    experiment.model.eval()
    test_df["Predictions"] = experiment.test_predictions
    test_df.to_csv(join(results_folder, f"test_set_seed{seed}.csv"), index=False)
    
    if args.eval_importance:
        print("Evaluating feature importance")
        class ShapWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, X):
                species_idx = X[:, 0:1]
                x_spectrum = X[:, 1:6001]
                dr_tensor = X[:, 6001:]
                response = []
                dataset = []
                batch = [species_idx, x_spectrum, dr_tensor, response, dataset]
                return experiment.model(batch)
            
        shap_wrapper = ShapWrapper(experiment.model)
        explainer = shap.DeepExplainer(shap_wrapper, X_background)
        shap_values = explainer.shap_values(X_fi_batch)
        column_names = ["Species"] + [f"Spectrum_{i}" for i in range(6000)] + [f"Fprint_{i}" for i in range(1024)]
        np.save(join(results_folder, f"shap_values_seed{seed}.npy"), shap_values)
        if not exists(join(output_folder, "shap_values_columns.json")):
            with open(join(output_folder, "shap_values_columns.json"), "w") as f:
                json.dump(column_names, f, indent=2)
        
        shap_values_df = pd.DataFrame(shap_values, columns=column_names)
        shap_values_df.to_csv(join(output_folder, "shap_values.csv"), index=False)
  
    
    print("Testing complete")





if __name__=="__main__":

    parser = ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="GNN")
    parser.add_argument("--experiment_group", type=str, default="ResAMR")
    parser.add_argument("--output_folder", type=str, default="outputs")
    ### Modified Diane
    parser.add_argument("--split_type", type=str, default="random", choices=["random", "specific", "drug_species_zero_shot", "drugs_zero_shot"])
    parser.add_argument("--split_ids", type=str)
    ###

    parser.add_argument("--training_setup", type=int)
    parser.add_argument("--eval_importance", action="store_true")
    
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--driams_dataset", type=str, choices=['A', 'B', 'C', 'D', 'any'], default='any')
    parser.add_argument("--driams_long_table", type=str)
    parser.add_argument("--spectra_matrix", type=str)
    parser.add_argument("--drugs_df", type=str)

    parser.add_argument("--conv_out_size", type=int, default=512)
    parser.add_argument("--sample_embedding_dim", type=int, default=512)
    #parser.add_argument("--sample_embedding_dim", type=int, default=6000)
    parser.add_argument("--drug_embedding_dim", type=int, default=512)
    

    parser.add_argument("--drug_emb_type", type=str, default="fingerprint", choices=["fingerprint", "vae_embedding", "gnn_embedding"])
    parser.add_argument("--fingerprint_class", type=str, default="morgan_1024", choices=["all", "MACCS", "morgan_512", "morgan_1024", "pubchem", "none", "molformer_huggingFace", "molformer_github", "selfies_label", "selfies_flattened_one_hot"])
    parser.add_argument("--fingerprint_size", type=int, default=128)


    parser.add_argument("--n_hidden_layers", type=int, default=5)

    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()
    args.num_workers = os.cpu_count()


    if args.training_setup is not None:
        dataset, split_type, seed = TRAINING_SETUPS[args.training_setup]
        
        args.seed = seed
        args.driams_dataset = dataset
        args.split_type = split_type
    args.species_embedding_dim = 0
    
    args.experiment_name = args.experiment_name + f"_DRIAMS-{args.driams_dataset}_{args.split_type}"


    main(args)
