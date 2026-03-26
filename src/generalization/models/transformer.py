#env: generalization
"""
PyTorch provides torch.nn.Transformer,
but it would be relatively interesting
to write our own transformer implementation
from scratch. In theory we should achieve
parity with the native implementation
"""

import torch
import torch.nn as nn
import torch.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_dim):
        super().__init__
        # for mha we split the embedding dimension across heads
        assert emb_dim % num_heads == 0 
        
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = self.emb_dim / self.num_heads


        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
    
    def split_heads(self, x):
        # reshape input tensor for multi-head attention
        # x is (batch_size, seq_length, emb_dim)
        B, S, D = x.shape()
        assert D == self.emb_dim
        
        # this is so that each head sees a part
        # of each sample
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        # combine outputs from each head into original shape
        B, _, S, _ = x.shape()
        
        # order of operations is reverse from split_heads
        return x.transpose(1, 2).contigious().view(B, S, self.emb_dim)

    def scaled_cross_attention(self, Q, K, V):

        

        
    


