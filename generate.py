import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.dataset import SMILESDataset
from src.transformer import Transformer
from train import create_vocab, idx_to_char

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the model.")
    parser.add_argument("--data", type=str, help="Path to the data file.")
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "--n_molecules", type=int, help="Number of molecules to generate."
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data)

    # Obtain mean, std logP for normalization
    log_p = np.array(df.Kow.tolist())
    mean_log_p = log_p.mean()
    std_log_p = log_p.std()

    print("Fitting GMM...")
    gmm = GaussianMixture(n_components=3)
    gmm.fit(log_p.reshape(-1, 1))
    print("Finished fitting GMM.")

    # df = df.sample(100)
    train, test_val = train_test_split(df, test_size=0.1)
    # test, val = train_test_split(test_val, test_size=0.5)

    vocab = create_vocab(df)
    print("Vocab size: ", len(vocab))

    train_dataset = SMILESDataset(train, vocab, mean_log_p, std_log_p)
    print("With special tokens", train_dataset.vocab_size)
    print(train_dataset.char_to_idx)

    state_dict = torch.load(args.model)
    model = Transformer(
        vocab_size=36,
        dmodel=512,  # 512
        nhead=8,
        decoder_layers=6,  # 6
        dim_feedforward=1024,  # 1024
        dropout=0.1,
        num_positions=1024,
        n_conditional_channels=1,
    )

    n_iters = args.n_molecules // args.batch_size

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    generated_molecules = []
    corresponding_log_p = []
    start = time.time()
    for it in tqdm(range(n_iters), total=n_iters, ncols=80):
        sample_log_p, _ = gmm.sample(args.batch_size)
        sample_log_p = torch.from_numpy(sample_log_p).squeeze(1).float().to(device)
        output = model.generate(
            batch_size=args.batch_size,
            max_target_sequence_length=50,
            start_id=34,
            device=device,
            mask_ids=(33, 34),
            temperature=0.5,
            log_p=sample_log_p,
        )
        generated, valid_log_p = idx_to_char(
            output, train_dataset, sample_log_p.cpu().numpy()
        )
        generated_molecules.extend(generated)
        corresponding_log_p.extend(valid_log_p)

    end = time.time()
    print("Time elapsed {}".format(end - start))
    import ipdb

    ipdb.set_trace()
    df_gen = pd.DataFrame(
        {"gen_mol": generated_molecules, "log_p": corresponding_log_p}
    )
    df_gen.to_json("generated_mols_1000.json")
