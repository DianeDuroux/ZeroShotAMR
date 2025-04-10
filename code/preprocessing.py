import os
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import argparse



def run_full_pipeline(
    base_data_dir,
    working_dir,
    long_table_A_path,
    long_table_all_path,
    dataset_years,
    encoding_dim=512,
    epochs=10,
    batch_size=50,
    num_copies=10,
    mask_range=(0.25, 0.25)
):
    # Create ProcessedData folder
    processed_dir = os.path.join(working_dir, 'ProcessedData')
    os.makedirs(processed_dir, exist_ok=True)

    # Define source dirs for raw data and create sample listing
    source_dirs = [
        ('DRIAMS-A', '2018'), ('DRIAMS-A', '2017'), ('DRIAMS-A', '2016'), ('DRIAMS-A', '2015'),
        ('DRIAMS-B', '2018'), ('DRIAMS-C', '2018'), ('DRIAMS-D', '2018')
    ]
    all_dirs = [(os.path.join(base_data_dir, ds, 'binned_6000', yr), int(yr)) for ds, yr in source_dirs]
    dfs = []
    for path, year in all_dirs:
        files = [f for f in os.listdir(path) if f.endswith('.txt')]
        sample_ids = [f.replace('.txt', '') for f in files]
        dfs.append(pd.DataFrame({'all_files': files, 'date': year, 'sample_id': sample_ids}))
    samples = pd.concat(dfs, ignore_index=True)

    # Load and combine metadata tables
    A = pd.read_csv(long_table_A_path)
    A = A[A['workstation'] != 'HospitalHygiene']
    rest = pd.read_csv(long_table_all_path)
    rest = rest[rest['dataset'] != 'A']
    rest['workstation'] = np.nan
    combined = pd.concat([A, rest], ignore_index=True)

    # Merge metadata with samples
    data = pd.merge(samples, combined, on='sample_id')
    data = data[data['workstation'] != 'HospitalHygiene']
    data['dataset_year'] = data['dataset'] + data['date'].astype(str)

    # Assign folds
    def assign_folds(group):
        sample_ids = group['sample_id'].unique()
        folds = np.random.permutation(np.repeat(np.arange(1, 11), len(sample_ids) // 10 + 1)[:len(sample_ids)])
        return pd.DataFrame({'sample_id': sample_ids, 'fold': folds})
    fold_map = data.groupby('dataset_year').apply(assign_folds).reset_index(drop=True)
    data = data.merge(fold_map, on='sample_id')
    data['Set'] = data['fold'].apply(lambda x: 'train' if x in range(1, 9) else 'test' if x == 9 else 'validation')
    data.to_csv(os.path.join(processed_dir, 'DRIAMS_folds_trainTestVal.csv'), index=False)

    # Save dataset-specific splits
    for dy in data['dataset_year'].unique():
        dy_dir = os.path.join(processed_dir, dy)
        os.makedirs(dy_dir, exist_ok=True)
        
        subset = pd.concat([
    data[(data['dataset_year'] == dy) & (data['Set'] != 'test')],
    data[data['Set'] == 'test']
])
        subset[['sample_id', 'Set']].drop_duplicates().sort_values(by='sample_id').to_csv(
            os.path.join(dy_dir, 'data_splits.csv'), index=False)

    # Define raw data directories
    raw_dirs = [os.path.join(base_data_dir, ds, 'binned_6000', yr) for ds, yr in source_dirs]

    # Process each dataset_year
    for dy in dataset_years:
        print(f"Processing dataset year: {dy}")
        split_file = os.path.join(processed_dir, dy, 'data_splits.csv')
        df = pd.read_csv(split_file)

        def load_samples(set_type):
            subset = df[df['Set'] == set_type]
            data_list = []
            for sid in subset['sample_id']:
                for directory in raw_dirs:
                    path = os.path.join(directory, f"{sid}.txt")
                    if os.path.exists(path):
                        data = np.loadtxt(path, delimiter=' ', skiprows=1)[:, 1]
                        data_list.append((sid, data))
                        break
            return pd.DataFrame({sid: arr for sid, arr in data_list}).T

        train_df = load_samples('train')
        val_df = load_samples('validation')
        test_df = load_samples('test')
        all_df = pd.concat([train_df, val_df, test_df])
        np.save(os.path.join(processed_dir, dy, 'rawSpectra_data.npy'), all_df.values)

        # Save combined metadata table
        lt = pd.read_csv(os.path.join(processed_dir, 'DRIAMS_folds_trainTestVal.csv'), low_memory=False)
        lt = lt[lt['dataset_year'] == dy]
        combined_table = lt[['species', 'sample_id', 'drug', 'response']].copy()
        combined_table['dataset'] = 'any'
        combined_table = combined_table.sort_values(by='sample_id')
        combined_table.to_csv(os.path.join(processed_dir, dy, 'combined_long_table.csv'), index=False)

        # Prepare scaled spectra for autoencoder
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train_df.values)
        scaled_val = scaler.transform(val_df.values)
        scaled_test = scaler.transform(test_df.values)

        # Masked input creation
        def create_masked_data(data):
            masked, original = [], []
            for row in data:
                for _ in range(num_copies):
                    masked_row = row.copy()
                    mask_size = int(mask_range[0] * len(row)) + random.randint(0, int((mask_range[1] - mask_range[0]) * len(row)))
                    masked_indices = np.random.choice(len(row), mask_size, replace=False)
                    masked_row[masked_indices] = 0
                    masked.append(masked_row)
                    original.append(row)
            return np.array(masked), np.array(original)

        masked_input, original_input = create_masked_data(scaled_train)

        # Autoencoder
        input_dim = scaled_train.shape[1]
        inp = Input(shape=(input_dim,))
        enc = Dense(encoding_dim, activation='relu')(inp)
        dec = Dense(input_dim, activation='sigmoid')(enc)
        autoencoder = Model(inputs=inp, outputs=dec)
        encoder = Model(inputs=inp, outputs=enc)
        autoencoder.compile(optimizer=Adam(0.001), loss='mse')
        autoencoder.fit(masked_input, original_input, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2)

        def encode(df, prefix):
            encoded = encoder.predict(df.values)
            encoded_df = pd.DataFrame(encoded, index=df.index)
            encoded_df.columns = [f'{prefix}{i+1}' for i in range(encoded.shape[1])]
            return encoded_df

        encoded = pd.concat([
            encode(train_df, 'X'),
            encode(val_df, 'X'),
            encode(test_df, 'X')
        ])

        sample_order = lt[lt['dataset_year'] == dy]['sample_id'].unique()
        final_array = encoded.loc[sample_order].values
        output_name = f'maskedAE_copy{num_copies}_batch{batch_size}_embSize{encoding_dim}_epoch{epochs}_MR{int(mask_range[0]*100)}_data.npy'
        np.save(os.path.join(processed_dir, dy, output_name), final_array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_data_dir')
    parser.add_argument('--working_dir')
    parser.add_argument('--long_table_A_path')
    parser.add_argument('--long_table_all_path')
    parser.add_argument('--dataset_years', nargs='+')
    parser.add_argument('--encoding_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_copies', type=int, default=10)
    parser.add_argument('--mask_range', type=float, nargs=2, default=(0.25, 0.25))

    args = parser.parse_args()

    run_full_pipeline(
        base_data_dir=args.base_data_dir,
        working_dir=args.working_dir,
        long_table_A_path=args.long_table_A_path,
        long_table_all_path=args.long_table_all_path,
        dataset_years=args.dataset_years,
        encoding_dim=args.encoding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_copies=args.num_copies,
        mask_range=tuple(args.mask_range)
    )

if __name__ == '__main__':
    main()