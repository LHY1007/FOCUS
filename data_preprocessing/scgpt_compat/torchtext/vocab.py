"""Minimal torchtext vocabulary interface for the pinned scGPT-spatial source.
Only token bookkeeping is used during pretrained embedding inference.
"""
class Vocab:
    def __init__(self, vocab):
        self._tokens = list(vocab._tokens) if isinstance(vocab, Vocab) else list(vocab)
        self._stoi = {token: i for i, token in enumerate(self._tokens)}
        self._default = None
        self.vocab = self

    def __len__(self):
        return len(self._tokens)

    def __contains__(self, token):
        return token in self._stoi

    def __getitem__(self, token):
        return self._stoi.get(token, self._default)

    def __call__(self, tokens):
        return [self[token] for token in tokens]

    def set_default_index(self, index):
        self._default = index

    def insert_token(self, token, index):
        if index != len(self._tokens):
            raise ValueError(f'Nonconsecutive vocabulary index: {index}')
        self._tokens.append(token)
        self._stoi[token] = index

    def append_token(self, token):
        self.insert_token(token, len(self))

    def get_stoi(self):
        return dict(self._stoi)


def vocab(ordered_dict, min_freq=1):
    return Vocab([token for token, count in ordered_dict.items() if count >= min_freq])
