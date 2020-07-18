import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import SMILESDataset, create_vocab, make_collate_fn
from src.utils import seed_everything

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the data file.")
    args = parser.parse_args()

    seed_everything(13)

    df = pd.read_csv(args.data)
    train, test_val = train_test_split(df, test_size=0.1)
    test, val = train_test_split(test_val, test_size=0.5)

    vocab = create_vocab(df)

    train_dataset = SMILESDataset(train, vocab)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        collate_fn=make_collate_fn(padding_values=train_dataset.padding_values),
    )

    val_dataset = SMILESDataset(val, vocab)
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        collate_fn=make_collate_fn(padding_values=train_dataset.padding_values),
    )

    test_dataset = SMILESDataset(test, vocab)
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        collate_fn=make_collate_fn(padding_values=train_dataset.padding_values),
    )

    import ipdb

    ipdb.set_trace()
