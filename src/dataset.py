import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


def create_vocab(df):
    smiles = df["SMILES"].tolist()
    unique_chars = set("".join(smiles))
    vocab = {char: idx for idx, char in enumerate(sorted(unique_chars))}
    return vocab


class SMILESDataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab
        self.pad_tok = len(vocab)

    @property
    def padding_values(self):
        return {"smiles": self.pad_tok, "mask": 0}

    def convert(self, string):
        idxes = [self.vocab[char] for char in string]
        return np.array(idxes)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = self.convert(row.SMILES)
        log_p = row.Kow
        return {"smiles": smiles, "logP": log_p, "mask": np.ones(len(smiles))}

    def __len__(self):
        return self.df.shape[0]


def make_collate_fn(padding_values):
    def _collate_fn(batch):

        for name, padding_value in padding_values.items():

            lengths = [len(sample[name]) for sample in batch]
            max_length = max(lengths)

            for n, size in enumerate(lengths):
                p = max_length - size
                if p:
                    pad_width = [(0, p)] + [(0, 0)] * (batch[n][name].ndim - 1)
                    if padding_value == "edge":
                        batch[n][name] = np.pad(batch[n][name], pad_width, mode="edge")
                    else:
                        batch[n][name] = np.pad(
                            batch[n][name],
                            pad_width,
                            mode="constant",
                            constant_values=padding_value,
                        )

        return default_collate(batch)

    return _collate_fn
