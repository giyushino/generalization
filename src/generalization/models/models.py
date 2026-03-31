#env: generalization
"""
TODO: implement rotary embeddings later

maybe implement contious batching? not sure
how we can fetch the data early, might
need to do some mp shenanigans
"""
import enum
import time
    
import torch
import torch.nn as nn

from generalization.data.tokenizer import AdditionTokenizer
from generalization.models.architecture import TransformerBlock

class AdditionTransformer(nn.Module):
    """
    Will be used for n by n digit addition
    """ 
    def __init__(self, num_layers: int, num_heads: int, emb_dim: int, ffn_mult: int, vocab_size: int, max_seq_len: int):
        super().__init__()
        t0 = time.perf_counter()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_seq_len, emb_dim)

        self.blocks = nn.ModuleList(
            TransformerBlock(num_heads, emb_dim, ffn_mult)
            for _ in range(num_layers)
        )

        self.norm = nn.LayerNorm(emb_dim)
        self.output = nn.Linear(emb_dim, vocab_size)
        t1 = time.perf_counter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape

        # embed with positional information
        positions = torch.arange(S, device=x.device)
        x = self.token_emb(x) + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x)
        
        # (batch_size, seq_lenth, vocab_size)
        return self.output(self.norm(x))

class AdditionPredicter():
    def __init__(self, model_config: dict, tokenizer_config: dict, device):
        self.model = AdditionTransformer(**model_config)
        self.tokenizer = AdditionTokenizer(**tokenizer_config)
        self.device = device

    def generate(self, input: list[str]) -> list[str]:
        # this is really bad but it's 4am and i'm on
        # a bus, so some ops will not be vectorized

        tokenized_input = self.tokenizer.encode(input).to(self.device)

        logits = self.model(tokenized_input)
        next_tokens = logits[:, -1, :].argmax(dim=-1)

class VisionTransfomer(nn.Module):
    # todo
    pass


if __name__ == "__main__":
    addition_config = {
        "num_layers": 10,
        "num_heads": 4,
        "emb_dim": 728,
        "ffn_mult": 4,
        "vocab_size": 14,
        "max_seq_len": 20
    }

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

#    model = AdditionTransformer(**addition_config)
#    rand_tensor = torch.randint(0, 13, (2, 10)) 
#    output = model(rand_tensor)

    model = AdditionPredicter(addition_config, tokenizer_config, device="cuda")
    rand_tensor = torch.randint(0, 13, (2, 10))
    test = ["84810+15592=100402", "84810+15593=100403"]

    model.generate(test)
