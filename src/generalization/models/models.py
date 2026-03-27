#env: generalization
"""
TODO: implement rotary embeddings later
"""
    
import torch
import torch.nn as nn

from generalization.models.architecture import TransformerBlock

class AdditionTransformer(nn.Module):
    """
    Will be used for n by n digit addition
    """ 
    def __init__(self, num_layers: int, num_heads: int, emb_dim: int, ffn_mult: int, vocab_size: int, max_seq_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_seq_len, emb_dim)

        self.blocks = nn.ModuleList(
            TransformerBlock(num_heads, emb_dim, ffn_mult)
            for _ in range(num_layers)
        )

        self.norm = nn.LayerNorm(emb_dim)
        self.output = nn.Linear(emb_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape

        # embed with positional information
        positions = torch.arange(S, device=x.device)
        x = self.token_emb(x) + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x)
        
        # (batch_size, seq_lenth, vocab_size)
        return self.output(self.norm(x))
        

class VisionTransfomer(nn.Module):
    # todo
    pass


if __name__ == "__main__":
    addition_config = {
        "num_layers": 10,
        "num_heads": 4,
        "emb_dim": 728,
        "ffn_mult": 4,
        "vocab_size": 13,
        "max_seq_len": 20
    }

    model = AdditionTransformer(**addition_config)
    rand_tensor = torch.randint(0, 13, (2, 10)) 
    output = model(rand_tensor)
    print(output)
