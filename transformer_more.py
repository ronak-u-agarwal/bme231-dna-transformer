#%%
'''
Transformer coded from pytorch without using attention/transformer modules
'''

#%%
import torch
import torch.nn.functional as F
import torch.nn as nn
import math


#%% Layer classes
# Make key, query, value classes? or maybe just linear layers
# then make an attention head class
# then make a multi-headed self-attention layer class
# then make a fully connected MLP class


class Embedding_Layer(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim,
                                      padding_idx=pad_idx)

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 300):
        super().__init__()
        # Precompute the PE matrix once
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # shape (max_len,1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )  # shape (embed_dim/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # shape (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, embed_dim)
        seq_len = x.size(1)
        out = x + self.pe[:, :seq_len]
        return out


class Head(nn.Module):
    # This should call the key/query matrices on everything, take the dot product, and softmax?
    # The output of the forward pass should be a list of kq-dim vectors that correspond to the head's "vote"
    # for how embeddings should be altered
    def __init__(self, embed_dim, kq_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.kq_dim = kq_dim
        self.key = nn.Linear(self.embed_dim, self.kq_dim)
        self.query = nn.Linear(self.embed_dim, self.kq_dim)
        self.value = nn.Linear(self.embed_dim, self.kq_dim)

    def forward(self, x: torch.Tensor, causal_mask, pad_mask) -> torch.Tensor:
        B, L, _ = x.shape
        # x: (genes_in_batch, seq_len, emb_dim) = (4k, 216, 16)
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)

        # K,Q,V: (batch, seq_len, qv_dim) = (4k, 216, 8)
        attention = torch.matmul(Q, K.transpose(-2, -1))  # I think means top half should be -inf?
        attention = attention / math.sqrt(self.kq_dim)

        c_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)
        p_mask = ~pad_mask.unsqueeze(1).expand(B, L, L)

        full_mask = c_mask | p_mask

        # set all masked-out scores to -inf before softmax
        attention = attention.masked_fill(full_mask, float('-inf'))

        # now softmax the attention weights to get good averages
        attn = torch.softmax(attention, dim=-1)
        out = torch.matmul(attn, V)
        return out

    # Class attention: make multiple heads, also make a stapled value matrix that


# up-projects the averages
class Multihead_Attention(nn.Module):
    def __init__(self, embed_dim, heads, kq_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.heads = heads
        self.kq_dim = kq_dim
        self.attn_heads = nn.ModuleList([
            Head(embed_dim, kq_dim) for _ in range(heads)
        ])
        self.bigV = nn.Linear(self.heads * self.kq_dim, self.embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor, causal_mask, pad_mask) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        # h(x): (batch, seq_len, kq_dim)
        head_outs = [h(x, causal_mask, pad_mask) for h in self.attn_heads]
        # concatenate on the last dim → (batch, seq_len, heads*kq_dim)
        stapled = torch.cat(head_outs, dim=-1)
        # 3) project back to embed_dim
        attn_out = self.bigV(stapled)
        out = self.norm(x + self.dropout(attn_out))
        return out


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, causal_mask, pad_mask):
        mlp_out = self.layer2(F.relu(self.layer(x)))
        out = self.norm(x + self.dropout(mlp_out))
        return out


class CodonTransformer(nn.Module):
    def __init__(self,
                 vocab_size: int = 22,
                 embed_dim: int = 16,
                 kq_dim: int = 8,
                 num_heads: int = 4,  # number of heads per multi-head attention block
                 hidden_dim: int = 24,
                 num_layers: int = 2,  # number of attn --> mlp rounds
                 pad_idx: int = 21,
                 max_len: int = 250):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.kq_dim = kq_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        # 1) Token embedding + pad handling
        self.embed = Embedding_Layer(vocab_size, embed_dim, pad_idx)
        # 2) Positional encoding
        self.pos_enc = PositionalEncoding(embed_dim, max_len)

        # 3) Stack of Transformer “encoder” blocks
        layers = []
        for _ in range(num_layers):
            layers.append(Multihead_Attention(embed_dim=embed_dim, kq_dim=kq_dim, heads=num_heads))
            layers.append(MLP(embed_dim, hidden_dim, embed_dim))

        self.layers = nn.ModuleList(layers)

        # 4) Final projection back to codon‑vocab size
        self.output = nn.Linear(embed_dim, vocab_size)
        self.final_dropout = nn.Dropout(0.1)

    def forward(self, tokens: torch.LongTensor):
        # tokens: (B, L)
        pad_mask = tokens != self.pad_idx  # (B, L)
        x = self.embed(tokens)  # (B, L, D)
        x = self.pos_enc(x)  # (B, L, D)

        # build a single causal mask once
        causal_mask = torch.triu(
            torch.ones(tokens.size(1), tokens.size(1), device=tokens.device),
            diagonal=1
        ).bool()  # (L, L)

        # pass through all layers
        for layer in self.layers:
            x = layer(x, causal_mask, pad_mask)

        logits = self.output(self.final_dropout(x))  # (B, L, V)
        return logits  # note that these aren't actually logits, but will be once we take the loss



#%%