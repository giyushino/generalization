"""
naive tokenization
TODO:
implement multiprocessing to
tokenize batches at the same time


i need to refactor to support the question 

"""

import torch

class AdditionTokenizer:
    # should we enforce max seq length within the tokenizer?
    def __init__(self, vocab: dict[str, int], eos_id, padding_id: int):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.eos_id = eos_id
        self.padding_id = padding_id
    
    def pad(self, tokenized: list[list[int]]) -> None:
        # update in place? not sure how i feel
        # about this design choice
        max_list_len = len(max(tokenized, key=len))

        for tokenized_sample in tokenized:
            tokenized_sample[:0] = [self.padding_id] * (max_list_len - len(tokenized_sample))

        return

    def encode(self, inputs: list[str]) -> torch.Tensor:
        # not sure about enforcing list
        # might be better to accept str
        encoded = []

        for untokenized in inputs:
            tokenized = []
            for letter in untokenized:
                tokenized.append(self.vocab[letter])
            tokenized.append(self.eos_id)
            encoded.append(tokenized)
        
        self.pad(encoded)

        return torch.tensor(encoded)

    def decode(self, inputs: list[list[int]]) -> list[str]:
        decoded = []

        for tokenized in inputs:
            detokenized = ""
            for token in tokenized:
                detokenized += self.inv_vocab[token]
            decoded.append(detokenized)

        return decoded


if __name__ == "__main__":
    vocab = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "+": 10,
        "=": 11,
        "<eos>": 12,
        "<pad>": 13,
    }

    tokenizer_config = {
        "vocab": vocab,
        "eos_id": 12,
        "padding_id": 13
    }

    tokenizer = AdditionTokenizer(**tokenizer_config)
    example = ["123+123=", "1+1="]
    print(tokenizer.encode(example))

