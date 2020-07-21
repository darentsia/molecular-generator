import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SMILESDataset, create_vocab, make_collate_fn
from src.training import configure_optimizer, configure_scheduler
from src.transformer import Transformer
from src.utils import seed_everything


def strip_molecule(molecule, end_tok):
    head, *tails = molecule.split(end_tok)
    return head[1:]


def idx_to_char(output, train_dataset, sample_log_p):
    molecules = output["output_ids"].cpu().numpy()
    char_molecules = []
    valid_log_p = []
    for idxes, log_p in zip(molecules, sample_log_p):
        if train_dataset.end_idx not in idxes:
            continue
        molecule = "".join(train_dataset.idx_to_char[idx] for idx in idxes)
        molecule = strip_molecule(molecule, end_tok=train_dataset.end_tok)
        char_molecules.append(molecule)
        valid_log_p.append(log_p)
    return char_molecules, valid_log_p


def to_markdown(output):
    df = pd.DataFrame({"Molecules": output})
    markdown_str = df.to_markdown()
    return markdown_str


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the data file.")
    parser.add_argument("--lr", type=int, default=0.0001, help="Learning rate.")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size."
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
    parser.add_argument("--out_suff", type=str, help="Suffix added to the saved model name.")

    args = parser.parse_args()

    seed_everything(13)
    writer = SummaryWriter("runs/test_run_logP_warmup15")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
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
            "Cached: ", round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), "GB",
        )

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
    test, val = train_test_split(test_val, test_size=0.5)

    print(f"Train samples: {len(train)}, Test samples: {len(test)}")

    vocab = create_vocab(df)
    print("Vocab size: ", len(vocab))

    train_dataset = SMILESDataset(train, vocab, mean_log_p, std_log_p)
    print("With special tokens", train_dataset.vocab_size)
    print(train_dataset.char_to_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=make_collate_fn(padding_values=train_dataset.padding_values),
        shuffle=True,
        drop_last=True,
    )

    val_dataset = SMILESDataset(val, vocab, mean_log_p, std_log_p)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=make_collate_fn(padding_values=train_dataset.padding_values),
    )

    test_dataset = SMILESDataset(test, vocab, mean_log_p, std_log_p)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=make_collate_fn(padding_values=train_dataset.padding_values),
    )

    model = Transformer(
        vocab_size=train_dataset.vocab_size,
        dmodel=512,  # 512
        nhead=8,
        decoder_layers=6,  # 6
        dim_feedforward=1024,  # 1024
        dropout=0.1,
        num_positions=1024,
        n_conditional_channels=1,
    )
    model.to(device)

    optimizer = configure_optimizer(model.named_parameters(), args.lr)
    steps_per_epoch = len(train_dataset) / (args.batch_size * args.accumulation)
    scheduler = configure_scheduler(
        optimizer,
        training_steps=(args.epochs * steps_per_epoch),
        warmup=args.warmup * steps_per_epoch,
    )
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    best_val_loss = 1000
    generated_molecules = defaultdict()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, total=len(train_loader), ncols=80):
            optimizer.zero_grad()

            smiles = batch["smiles"].to(device)
            log_p = batch["logP"].to(device)
            mask = batch["mask"].to(device)
            target_sequence_length = batch["seq_len"].to(device)

            output = model(
                output_ids=smiles,
                target_sequence_length=target_sequence_length,
                log_p=log_p,
            )
            loss = torch.mean(
                torch.sum(
                    criterion(
                        input=output["logits"].permute(0, 2, 1)[:, :, :-1],
                        target=smiles[:, 1:],
                    )
                    * mask[:, :-1],
                    axis=1,
                )
                / target_sequence_length
            )
            epoch_loss += loss.item() / len(train_dataset) * len(smiles)
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch} loss: {epoch_loss}")
        model.eval()
        sample_log_p, _ = gmm.sample(4)
        sample_log_p = (
            torch.from_numpy(sample_log_p).squeeze(1).float().to(device)
        )
        output = model.generate(
            batch_size=4,
            max_target_sequence_length=50,
            start_id=train_dataset.start_idx,
            device=device,
            mask_ids=(train_dataset.pad_idx, train_dataset.start_idx),
            temperature=0.5,
            log_p=sample_log_p,
        )
        generated = idx_to_char(output, train_dataset)
        print(generated)
        generated_molecules[epoch] = generated

        writer.add_text("Generated_molecule", to_markdown(generated), epoch)
        writer.add_scalar("train_loss", epoch_loss, epoch)

        # Validation
        with torch.no_grad():
            val_loss = 0
            for batch in tqdm(val_loader, total=len(val_loader), ncols=80):

                smiles = batch["smiles"].to(device)
                log_p = batch["logP"].to(device)
                mask = batch["mask"].to(device)
                target_sequence_length = batch["seq_len"].to(device)

                output = model(
                    output_ids=smiles,
                    target_sequence_length=target_sequence_length,
                    log_p=log_p,
                )
                loss = torch.mean(
                    torch.sum(
                        criterion(
                            input=output["logits"].permute(0, 2, 1)[:, :, :-1],
                            target=smiles[:, 1:],
                        )
                        * mask[:, :-1],
                        axis=1,
                    )
                    / target_sequence_length
                )
                val_loss += loss.item() / len(val_dataset) * len(smiles)
            print(f"Epoch {epoch} validation loss: {val_loss}")
            writer.add_scalar("val_loss", val_loss, epoch)

            # Model saving 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"best_model_{args.out_suff}.pt")
                
    import ipdb; ipdb.set_trace()
    generated_molecules_df = pd.DataFrame(generated_molecules)
    generated_molecules_df.to_json("generated_molecules.json")
    import ipdb; ipdb.set_trace()
