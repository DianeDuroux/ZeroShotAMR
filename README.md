
# 📄 Generalizable machine learning models for rapid antimicrobial resistance prediction in unseen healthcare settings

This repository contains the code used for the experiments in the paper:

**_Generalizable machine learning models for rapid antimicrobial resistance prediction in unseen healthcare settings_**  
by *Diane Duroux, Paul P. Meyer, Giovanni Visonà, and Niko Beerenwinkel*.

---

## 🦠 Overview

The goal of this work is to develop a generalizable Antimicrobial Resistance (AMR) prediction model based on robust feature representations derived from MALDI-TOF mass spectra and chemical structure encodings.

We investigate the model's ability to generalize under two settings: Hospital generalizability, and Temporal generalizability.

We introduce Masked Autoencoder (MAE)-based representations for MALDI-TOF MS data, and explore MolFormer and SELFIES encodings for antimicrobial structures. We demonstrate improvements in zero-shot performance (predicting on unseen data distributions) and stability when transitioning from non-zero-shot to zero-shot settings.

---
## ⚙️ Install the dependencies
You can set up the project with either pip or uv.

### Option A - pip:
Install the necessary dependencies listed in the requirements.txt file

```bash
pip install -r requirements.txt
```

### Option B - uv:
We provide pyproject.toml and uv.lock for macOS, Windows, and Linux.

Note: On a Linux or non-apple silicon  please use the pyproject.toml file for Mac and rewrite the uv.lock after installation. 

```bash
# 0) Install uv (one-time)
# mac/linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# windows (PowerShell):
iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex
 
# 1) Ensure the pinned Python is available (adjust if your pyproject pins a version)
uv python install 3.11
 
# 2) Create the exact environment from the lockfile
uv sync --frozen
 
# 3) Run your code within the env
uv run python -V
uv run python your_script.py
```

---

## 🗂️ DRIAMS dataset

The MALDI-TOF spectra used to reproduce the experiments can be obtained by downloading the DRIAMS dataset from:  
🔗 [https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q)

Instructions:

1. Download the most recent versions of DRIAMS A, B, C, and D (`tar.gz`).

2. For each dataset (A–D), the binned mass spectra used in this analysis are located in the `binned_6000/` folder.

3. Inside each yearly subfolder, you will find the mass spectra of individual samples, with filenames corresponding to their unique sample IDs.

### Directory Structure

```
/DRIAMS/
├── DRIAMS-A/
│   └── binned_6000/
│       ├── 2015/
│       ├── 2016/
│       └── ...
├── DRIAMS-B/
├── DRIAMS-C/
└── DRIAMS-D/
```

## ⚙️ DRIAMS Preprocessing Pipeline

This script preprocesses the DRIAMS dataset by performing data splitting and encoding MALDI-TOF spectra using a Masked Autoencoder. It prepares the data for downstream AMR prediction tasks.

### 📦 Output Files

For each specified dataset-year (e.g., `A2015`, `B2018`), the script generates:

- `combined_long_table.csv`  
  ➤ Sample metadata: `species`, `sample_id`, `drug`, `response`, and `dataset`.

- `data_splits.csv`  
  ➤ Dataset split into `train`, `validation`, and `test`.

- `rawSpectra_data.npy`  
  ➤ Raw MALDI-TOF spectra in NumPy format.

- `maskedAE_[...]_data.npy`  
  ➤ MAE-encoded spectra.

### 🛠 Required Arguments

| Argument                | Description                                                                                 |
|-------------------------|---------------------------------------------------------------------------------------------|
| `--base_data_dir`       | Path to the base directory containing DRIAMS folders.                                   |
| `--working_dir`         | Path to your working directory (outputs will be saved in `ProcessedData/`).                 |
| `--long_table_A_path`   | Path to the metadata CSV file for DRIAMS-A (includes workstation info).                     |
| `--long_table_all_path` | Path to the metadata CSV file for all DRIAMS.                                               |
| `--dataset_years`       | List of dataset-year identifiers (e.g., `A2015 B2018 C2018`).                               |
| `--encoding_dim`        | Dimension of the MAE latent space (default: `512`).                                         |
| `--epochs`              | Number of epochs for MAE training (default: `10`).                                          |
| `--batch_size`          | Batch size for MAE training (default: `50`).                                                |
| `--num_copies`          | Number of masked versions per spectrum (default: `10`).                                     |
| `--mask_range`          | Range for feature masking (e.g., `0.2 0.5`).                                                |

### 🚀 Example Usage

```bash
python3 code/preprocessing.py \
  --base_data_dir /path/to/DRIAMS \
  --working_dir /path/to/ZeroShotAMR \
  --long_table_A_path OriginalData/DRIAMS-A_ws_long_table.csv \
  --long_table_all_path OriginalData/DRIAMS_combined_long_table.csv \
  --dataset_years A2015 A2016 A2017 A2018 B2018 C2018 D2018 \
  --encoding_dim 512 \
  --epochs 2 \
  --batch_size 50 \
  --num_copies 10 \
  --mask_range 0.2 0.5
```

---

## 💻 AMR Classifier Training with ResMLP and inference

The following command trains a ResMLP model for AMR classification using the preprocessed DRIAMS data.

### 📦 Output

In `output/<experiment_group>/<experiment_name>_results/`, the script generates:

- `test_set_seed0.csv`  
  ➤ Contains predictions: `species`, `sample_id`, `drug`, `response`, and `Prediction`.

### 🛠 Required Arguments

| Argument                | Description                                                                                     |
|-------------------------|-------------------------------------------------------------------------------------------------|
| `--driams_long_table`   | Path to the metadata file for the current dataset.                                              |
| `--spectra_matrix`      | Path to the input mass spectra (either raw or MAE-encoded).                                     |
| `--sample_embedding_dim`| Dimension of the spectra input (6000 for raw, or same as <encoding_dim> for MAE).               |
| `--drugs_df`            | Path to the antimicrobial compound encoding file.                                               |
| `--fingerprint_class`   | Type of encoding: `'morgan_1024'`, `'molformer_github'`, or `'selfies_flattened_one_hot'`.      |
| `--fingerprint_size`    | Size of the encoding: 1024 (Morgan), 768 (Molformer), or 24160 (SELFIES).                       |
| `--split_type`          | Set to `specific` if splits are pre-defined, else random.                                       |
| `--split_ids`           | Path to the `data_splits.csv` file.                                                             |
| `--experiment_group`    | Name of the output folder.                                                                      |
| `--experiment_name`     | Name of the output subfolder.                                                                   |
| `--seed`                | Random seed for reproducibility.                                                                |
| `--n_epochs`            | Number of epochs for classifier training.                                                       |
| `--learning_rate`       | Learning rate for the optimizer.                                                                |
| `--patience`            | Number of epochs to wait before early stopping.                                                 |
| `--batch_size`          | Batch size for classifier training.                                                             |

### 🚀 Example: ResMLP Training on DRIAMS B2018 with Raw Spectra + Morgan Fingerprints

```bash
ulimit -Sn 10000  # Optional: increase file descriptor limit if needed

python3 code/ResAMR_classifier.py \
    --driams_long_table ProcessedData/B2018/combined_long_table.csv \
    --spectra_matrix ProcessedData/B2018/rawSpectra_data.npy \
    --sample_embedding_dim 6000 \
    --drugs_df OriginalData/drug_fingerprints_Mol_selfies.csv \
    --fingerprint_class morgan_1024 \
    --fingerprint_size 1024 \
    --split_type specific \
    --split_ids ProcessedData/B2018/data_splits.csv \
    --experiment_group rawMS_MorganFing \
    --experiment_name ResMLP \
    --seed 0 \
    --n_epochs 2 \
    --learning_rate 0.0003 \
    --patience 10 \
    --batch_size 128
```

---

## 💰 Funding

This research was primarily supported by the ETH AI Center.
