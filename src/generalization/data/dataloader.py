"""
todo if we ever want to use more complex
datasets and resume training
"""

from generalization.data.dataset import AdditionDataset
from generalization.data.tokenizer import AdditionTokenizer


class AdditionDataloader():
    def __init__(self, tokenizer: AdditionTokenizer, dataset: AdditionDataset, batch_size: int, mode: str = "train"):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.batch_size = batch_size

        self.idx = 0 - batch_size
        self.batch = {}
        self.mode = mode

    def __iter__(self):
        self.idx = 0 - self.batch_size

        return self

    def __next__(self):
        if self.idx >= self.num_samples - self.batch_size:
            raise StopIteration
        else:
            self.idx += self.batch_size
            if self.mode == "train":
                batch = self.dataset[self.idx: min(self.idx + self.batch_size, self.num_samples)]
            else:
                batch = [self.dataset[idx]["question"] for idx in range(self.idx, min(self.idx + self.batch_size, self.num_samples))] 

            self.batch = {
                "original": batch,
                "encoded": self.tokenizer.encode(batch)
            }

            return self.batch


            



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

    dataset_config = {
        "num_samples": 100,
        "num_digits": [4, 5],
        "seed": 42,
        "mode": "eval"
    }

    tokenizer = AdditionTokenizer(**tokenizer_config)
    dataset = AdditionDataset(**dataset_config)
    dataloader = AdditionDataloader(tokenizer, dataset, 2, "eval")
    
    for batch in dataloader:
        print(batch["original"])
        print(batch["encoded"])
        break
