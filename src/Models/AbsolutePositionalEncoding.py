import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd


class tAPE(nn.Module):
    """
    Temporal Absolute Positional Encoding with sine and cosine functions, scaled by (d_model / max_len).

    This encoding method injects temporal position information into the input embeddings.
    It scales the sinusoidal pattern based on embedding size and sequence length.

    Args:
        d_model (int): Embedding dimensionality.
        dropout (float): Dropout rate applied after adding positional encoding. Default is 0.1.
        max_len (int): Maximum sequence length expected. Default is 1024.
        scale_factor (float): Factor to scale the positional encodings. Default is 1.0.
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        """
        Add temporal absolute positional encoding to input tensor.

        Args:
            x (Tensor): Input tensor of shape (sequence_length, batch_size, d_model).

        Returns:
            Tensor: Positional encoded tensor of same shape as input.
        """
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)


class AbsolutePositionalEncoding(nn.Module):
    """
    Classic absolute positional encoding using sinusoidal functions (without temporal scaling).

    Uses the original Transformer positional encoding approach with sine on even indices
    and cosine on odd indices.

    Args:
        d_model (int): Embedding dimensionality.
        dropout (float): Dropout rate applied after adding positional encoding. Default is 0.1.
        max_len (int): Maximum sequence length expected. Default is 1024.
        scale_factor (float): Factor to scale the positional encodings. Default is 1.0.
    """


    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(AbsolutePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        """
        Add absolute positional encoding to input tensor.

        Args:
            x (Tensor): Input tensor of shape (sequence_length, batch_size, d_model).

        Returns:
            Tensor: Positional encoded tensor of same shape as input.
        """
        x = x + self.pe
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

        # distance = torch.matmul(self.pe, self.pe[10])
        # import matplotlib.pyplot as plt

        # plt.plot(distance.detach().numpy())
        # plt.show()

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe
        # distance = torch.matmul(self.pe, self.pe.transpose(1,0))
        # distance_pd = pd.DataFrame(distance.cpu().detach().numpy())
        # distance_pd.to_csv('learn_position_distance.csv')
        return self.dropout(x)