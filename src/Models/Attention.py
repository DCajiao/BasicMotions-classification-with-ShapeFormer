import torch
import torch.nn as nn
from einops import rearrange
import pandas as pd


class Attention(nn.Module):
    """
    Standard multi-head self-attention mechanism.

    This module computes scaled dot‑product attention using a projection of the
    input sequence into query (Q), key (K), and value (V) tensors. The attention
    scores are computed as QKᵀ / sqrt(d_head), normalized with softmax, and used
    to aggregate the values.

    Args:
        emb_size (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability applied after attention.

    Attributes:
        key (nn.Linear): Linear projection for keys.
        value (nn.Linear): Linear projection for values.
        query (nn.Linear): Linear projection for queries.
        num_heads (int): Number of attention heads.
        scale (float): Scaling factor = emb_size^(-0.5).
        attn (Tensor or None): Stores last computed attention map.
    """

    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)
        self.attn = None

    def forward(self, x):
        """
        Forward pass for multi-head self-attention.

        Args:
            x (Tensor): Input tensor of shape (B, L, D)
                where:
                    B = batch size
                    L = sequence length
                    D = embedding dimension

        Returns:
            Tensor: Output tensor of shape (B, L, D) after attention and layer normalization.
        """

        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len,
                                self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len,
                                  self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len,
                                  self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale
        # attn shape (seq_len, seq_len)
        attn = nn.functional.softmax(attn, dim=-1)

        # import matplotlib.pyplot as plt
        # plt.plot(x[0, :, 0].detach().cpu().numpy())
        # plt.show()
        self.attn = attn

        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out

    def get_att(self):
        """
        Returns the most recently computed attention matrix.

        Returns:
            Tensor or None: Attention weights of shape (B, H, L, L),
                            where H = number of heads.
        """
        return self.attn


class Attention_Rel_Scl(nn.Module):
    """
    Multi-head attention with relative positional biases (scalar form).

    Implements relative position encoding by adding learned bias values to the
    attention scores, based on pairwise relative distances between sequence indices.

    Args:
        emb_size (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        seq_len (int): Maximum sequence length.
        dropout (float): Dropout probability.
    """

    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.seq_len - 1), num_heads))
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        """
        Forward pass for relative-bias attention.

        Args:
            x (Tensor): Input tensor of shape (B, L, D)

        Returns:
            Tensor: Output tensor of shape (B, L, D) after attention and layer normalization.
        """
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len,
                                self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len,
                                  self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len,
                                  self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale
        # attn shape (seq_len, seq_len)
        attn = nn.functional.softmax(attn, dim=-1)

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, 8))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=1 * self.seq_len, w=1 * self.seq_len)
        attn = attn + relative_bias

        # distance_pd = pd.DataFrame(relative_bias[0,0,:,:].cpu().detach().numpy())
        # distance_pd.to_csv('scalar_position_distance.csv')

        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out


class Attention_Rel_Vec(nn.Module):
    """
    Multi-head attention using vector-based relative position representations.

    Each relative distance between positions i and j is encoded by a learnable
    vector, enabling more expressive positional interactions than scalar biases.

    Args:
        emb_size (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        seq_len (int): Maximum sequence length.
        dropout (float): Dropout probability.
    """

    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.Er = nn.Parameter(torch.randn(
            self.seq_len, int(emb_size/num_heads)))

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.seq_len, self.seq_len))
            .unsqueeze(0).unsqueeze(0)
        )

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        """
        Forward pass for relative vector-based attention.

        Args:
            x (Tensor): Input tensor of shape (B, L, D)

        Returns:
            Tensor: Output tensor of shape (B, L, D) after attention and layer normalization.
        """
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len,
                                self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len,
                                  self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len,
                                  self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        QEr = torch.matmul(q, self.Er.transpose(0, 1))
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, self.num_heads, seq_len, seq_len)

        attn = torch.matmul(q, k)
        # attn shape (seq_len, seq_len)
        attn = (attn + Srel) * self.scale

        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out

    def skew(self, QEr):
        """
        Applies the skewing operation used in Transformer-XL style relative attention.

        Converts relative embeddings into a form aligned with attention matrix ordering.

        Args:
            QEr (Tensor): Tensor of shape (B, H, L, L) representing Q * Eᵀ.

        Returns:
            Tensor: Skewed tensor of shape (B, H, L, L).
        """
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = nn.functional.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


class ShapeAttention(nn.Module):
    """
    Cross-attention mechanism designed for ShapeFormer.

    Computes attention where queries are derived from sequence x, while keys and
    values come from an external sequence s. This allows the model to attend to
    shape tokens or extracted shapelets separately from the raw sequence.

    Args:
        emb_size (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """

    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x, s=None, d=None):
        """
        Forward pass for shape-based cross-attention.

        Args:
            x (Tensor): Query input of shape (B, L, D).
            s (Tensor): Key/value input of shape (B, L, D).
            d (Tensor, optional): Unused placeholder for optional extra context.

        Returns:
            Tensor: Updated representation of shape (B, L, D) after cross-attention.
        """

        batch_size, seq_len, _ = x.shape
        k = self.key(s).reshape(batch_size, seq_len,
                                self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(s).reshape(batch_size, seq_len,
                                  self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len,
                                  self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale
        # attn shape (seq_len, seq_len)
        attn = nn.functional.softmax(attn, dim=-1)

        # import matplotlib.pyplot as plt
        # plt.plot(x[0, :, 0].detach().cpu().numpy())
        # plt.show()

        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out
