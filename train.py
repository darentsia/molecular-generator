import argparse

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import SMILESDataset, create_vocab, make_collate_fn
from src.training import configure_optimizer, configure_scheduler
from src.transformer import Transformer
from src.utils import seed_everything

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the data file.")
    parser.add_argument(
        "--lr", type=int, default=0.01, help="Learning rate."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size."
    )
    parser.add_argument(
        "--accumulation",
        type=int,
        default=1,
        help="Number of accumulation steps.",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Number of warmup epochs."
    )

    args = parser.parse_args()

    seed_everything(13)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    print("Using device:", device)
    print()

    # Additional info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print(
            "Allocated:",
            round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1),
            "GB",
        )
        print(
            "Cached:   ",
            round(torch.cuda.memory_cached(0) / 1024 ** 3, 1),
            "GB",
        )

    df = pd.read_csv(args.data)
    train, test_val = train_test_split(df, test_size=0.1)
    test, val = train_test_split(test_val, test_size=0.5)

    vocab = create_vocab(df)
    print("Vocab size: ", len(vocab))

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

    model = Transformer(
        vocab_size=len(vocab)+1,
        dmodel=512,
        nhead=8,
        decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        num_positions=1024,
    )
    model.to(device)

    optimizer = configure_optimizer(model.named_parameters(), args.lr)
    scheduler = configure_scheduler(
        optimizer,
        training_steps=(
            args.epochs
            * len(train_dataset)
            / (args.batch_size * args.accumulation)
        ),
        warmup=args.warmup,
    )
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    for batch in train_loader:
        smiles = batch["smiles"].to(device)
        log_p = batch["logP"].to(device)
        mask = batch["mask"].to(device)
        target_sequence_length = batch["seq_len"].to(device)

        output = model(
            output_ids=smiles, target_sequence_length=target_sequence_length
        )
        import ipdb; ipdb.set_trace()
        loss = criterion(input=output["logits"].permute(0, 2, 1), target=smiles) * mask
