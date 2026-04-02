import os
import time

import torch
import torch.nn.functional as F

from generalization.data.dataloader import AdditionDataloader
from generalization.data.dataset import AdditionDataset
from generalization.models.models import AdditionTransformer
from generalization.data.tokenizer import AdditionTokenizer


class AdditionTrainer():
    def __init__(self, epochs: int, learning_rate: int,
                 batch_size: int, model_config: dict, 
                 tokenizer_config: dict, dataset_config: dict,
                 device: str
    ):
        t0 = time.perf_counter()
        self.device = device
        self.model = AdditionTransformer(**model_config).to(self.device)
        self.tokenizer = AdditionTokenizer(**tokenizer_config)
        self.dataset = AdditionDataset(**dataset_config)
        self.dataloader = AdditionDataloader(self.tokenizer, self.dataset, batch_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
         
        self.epochs = epochs
        self.lr = learning_rate
        t1 = time.perf_counter()
        
        print(f"model loaded in {(t1 - t0):.4f} seconds")

    def loss(self, input: torch.Tensor, logits: torch.Tensor):
        B, S, V = logits.shape
        # we do this to remove the last logit, we don't have
        # a ground truth for that
        shift_logits = logits[:, :-1, :]
        # since we have no context on the first token
        # we should just remove it
        shift_labels = input[:, 1:]


        return F.cross_entropy(
            shift_logits.reshape(-1, V),
            shift_labels.reshape(-1),
            ignore_index=self.tokenizer.padding_id,
        )

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for index, batch in enumerate(self.dataloader):
                tokenized_batch = batch["encoded"].to(self.device)
                attention_mask = tokenized_batch != self.tokenizer.padding_id
                print(f"step: {index}/{100} || epoch: {epoch}")
                self.optimizer.zero_grad()
                output = self.model(tokenized_batch, attention_mask)
                loss = self.loss(tokenized_batch, output)
                print(loss) 
                loss.backward()
                self.optimizer.step()

            print(f"{loss=}")

        torch.save(self.model.state_dict(), "/home/allan/nvim/generalization/checkpoints/generalized/model.pth")
   
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
        "padding_id": 13,
    }
    model_config = {
        "num_layers": 10,
        "num_heads": 4,
        "emb_dim": 728,
        "ffn_mult": 4,
        "vocab_size": 14,
        "max_seq_len": 30
    }
    dataset_config = {
        "num_samples": 500_000,
        "num_digits": [1, 5],
        "seed": 42
    }
    trainer_config = {
        "epochs": 1,
        "learning_rate": 0.01,
        "batch_size": 100,
        "device": "cuda",
        "model_config": model_config,
        "tokenizer_config": tokenizer_config,
        "dataset_config": dataset_config
    }
    trainer = AdditionTrainer(**trainer_config)
    trainer.train()
