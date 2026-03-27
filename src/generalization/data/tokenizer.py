"""
naive tokenization
TODO:
implement multiprocessing to
tokenize batches at the same time
"""

class AdditionTokenizer:
    def __init__(self, vocab: dict[str, int], padding_id: int):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.padding_id = padding_id
    
    def pad(self, tokenized: list[list[int]]) -> list[list[int]]:
        # update in place? not sure how i feel
        # about this design choice
        max_list_len = len(max(tokenized, key=len))

        for tokenized_sample in tokenized:
            tokenized_sample.extend([self.padding_id] * (max_list_len - len(tokenized_sample)))

        return

    def encode(self, inputs: list[str]) -> list[list[int]]:
        # not sure about enforcing list
        # might be better to accept str
        encoded = []
        for untokenized in inputs:
            tokenized = []
            for letter in untokenized:
                tokenized.append(self.vocab[letter])
            encoded.append(tokenized)
        
        self.pad(encoded)
        return encoded

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
        "padding_id": 13
    }

    tokenizer = AdditionTokenizer(**tokenizer_config)
    example = ["123+123=", "1+1="]
    print(tokenizer.encode(example))
