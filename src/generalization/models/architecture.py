#env: generalization
"""
PyTorch provides torch.nn.Transformer,
but it would be relatively interesting
to write our own transformer from
scratch. In theory we should achieve
parity with the native implementation

Later I want to write a megafused kernel to
do all of this one one pass (either CUDA or triton)
"""

import math
import torch
import torch.nn as nn
import torch.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, emb_dim: int):
        super().__init__()
        # for mha we split the embedding dimension across heads
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_head"
        
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = self.emb_dim // self.num_heads

        
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # reshape input Tensor for multi-head attention
        # x is (batch_size, seq_length, emb_dim)
        B, S, D = x.shape
        assert D == self.emb_dim
        
        # this is so that each head sees a part
        # of each sample
        # some optimization thoughts:
        return x.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # combine outputs from each head into original shape
        B, _, S, _ = x.shape
    
        # order of operations is reverse from split_heads
        return x.transpose(1, 2).reshape(B, S, self.emb_dim)

    def scaled_cross_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Q matrix contains what information each token needs
        # K matrix contains what information each token has
        # V matrix contains what information each token provides

        # Q and K are (batch_size, num_heads, seq_length, head_dim)
        # for matrix mult, we'll need to take the transpose of K, ie
        # we swap head_dim and seq_length
        # divide by sqrt to reduce variance in downstream operations


        # produces (batch_size, num_heads, seq_length, seq_length)
        # torch.matmul automatically broadcasts over the first two dims
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) 
        attn_probs = torch.softmax(attn_scores, dim=-1)
       
        # this final matrix mult mixes information between the embeddings
        # of a token with the tokens it attended to
        return torch.matmul(attn_probs, V)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.split_heads(self.q_proj(x)) 
        K = self.split_heads(self.k_proj(x)) 
        V = self.split_heads(self.v_proj(x)) 

        attn_output = self.scaled_cross_attention(Q, K, V)

        return self.out_proj(self.combine_heads(attn_output))
        

class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int, emb_dim: int, ffn_mult: int):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, emb_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        # maybe we make param to change the
        # activation function?
        ffn_dim = emb_dim * ffn_mult
        assert ffn_dim.is_integer(), "emb_dim * ffn_mult must result in an integer"

        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # can't do += in pytorch since this
        # updates values in place, preventing
        # gradients from flowing properly
        x = x + self.mha(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

if __name__ == "__main__": 
    config = {
        "num_heads": 4,
        "emb_dim": 728,
        "ffn_mult": 4
    }
    
    mha = TransformerBlock(**config)
    rand_Tensor = torch.rand((2, 10, 728))
    output = mha(rand_Tensor)
    print(output)

