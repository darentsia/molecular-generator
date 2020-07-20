import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


PAD_TOK = "X"
START_TOK = "<"
END_TOK = ">"


def create_vocab(df):
    smiles = df["SMILES"].tolist()
    unique_chars = set("".join(smiles))
    vocab = {char: idx for idx, char in enumerate(sorted(unique_chars))}

    vocab[PAD_TOK] = len(vocab)
    vocab[START_TOK] = len(vocab)
    vocab[END_TOK] = len(vocab)

    return vocab


class SMILESDataset(Dataset):

    def __init__(self, df, vocab):
        self.df = df
        self.char_to_idx = vocab
        self.vocab_size = len(self.char_to_idx)

        self.pad_tok = PAD_TOK
        self.start_tok = START_TOK
        self.end_tok = END_TOK

        self.pad_idx = self.char_to_idx[PAD_TOK]
        self.start_idx = self.char_to_idx[START_TOK]
        self.end_idx = self.char_to_idx[END_TOK]

        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

    @property
    def padding_values(self):
        return {"smiles": self.pad_idx, "mask": 0}

    def to_idx(self, string):
        idxes = (
            [self.start_idx]
            + [self.char_to_idx[char] for char in string]
            + [self.end_idx]
        )
        return np.array(idxes)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = self.to_idx(row.SMILES)
        log_p = row.Kow
        return {
            "smiles": smiles,
            "logP": log_p,
            "seq_len": len(smiles),
            "mask": np.ones(len(smiles)),
        }

    def __len__(self):
        return self.df.shape[0]

    def to_chars(self, idxes):
        return np.array([self.idx_to_char[idx] for idx in idxes])


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
