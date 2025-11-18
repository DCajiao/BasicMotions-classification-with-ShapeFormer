import numpy as np
from torch import nn
import torch
from Models.AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
from Models.Attention import Attention, Attention_Rel_Scl, Attention_Rel_Vec
from Models.position_shapelet import PPSN
from Shapelet.auto_pisd import auto_piss_extractor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.


    Args:
    model (nn.Module): The PyTorch model.


    Returns:
    int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Permute(nn.Module):
    """
    Layer that permutes the input tensor dimensions from (B, T, D) to (T, B, D).
    """

    def forward(self, x):
        return x.permute(1, 0, 2)


def model_factory(config):
    """
    Factory method to instantiate a PPSN or Shapeformer model based on configuration.


    Args:
    config (dict): Configuration dictionary containing model parameters.


    Returns:
    nn.Module: Instantiated model.
    """
    if config['Net_Type'][0] == "PPSN":
        model = PPSN(shapelets_info=config['shapelets_info'], shapelets=config['shapelets'],
                     len_ts=config['len_ts'], num_classes=config['num_labels'],
                     sge=config['sge'], window_size=config['window_size'])
        config['shapelets'] = None
    elif config['Net_Type'][0] == "Shapeformer":
        model = Shapeformer(config, num_classes=config['num_labels'])
    return model


class ShapeBlock(nn.Module):
    """
    Computes complexity-aware shapelet embedding for a specific segment of a time series.


    Args:
    shapelet_info (list): Metadata about the shapelet (position, dimension, etc).
    shapelet (list or array): Shapelet values.
    shape_embed_dim (int): Dimension of the shapelet embedding.
    window_size (int): Tolerance for alignment.
    len_ts (int): Length of the time series.
    norm (float): Complexity-invariance normalization factor.
    max_ci (float): Maximum complexity-invariant correction factor.
    """

    def __init__(self, shapelet_info=None, shapelet=None, shape_embed_dim=32, window_size=50, len_ts=100, norm=1000, max_ci=3):
        super(ShapeBlock, self).__init__()
        self.dim = shapelet_info[5]
        self.shape_embed_dim = shape_embed_dim
        self.shapelet = torch.nn.Parameter(torch.tensor(
            shapelet, dtype=torch.float32), requires_grad=True)
        # window_size = 0
        self.window_size = window_size
        self.norm = norm
        self.kernel_size = shapelet.shape[-1]
        self.weight = shapelet_info[3]

        self.ci_shapelet = np.sqrt(
            np.sum((shapelet[1:] - shapelet[:-1])**2)) + 1/norm
        self.max_ci = max_ci

        self.sp = shapelet_info[1]
        self.ep = shapelet_info[2]

        self.start_position = int(shapelet_info[1] - window_size)
        self.start_position = self.start_position if self.start_position >= 0 else 0
        self.end_position = int(shapelet_info[2] + window_size)
        self.end_position = self.end_position if self.end_position < len_ts else len_ts

        self.l1 = nn.Linear(self.kernel_size, shape_embed_dim)
        self.l2 = nn.Linear(self.kernel_size, shape_embed_dim)

    def forward(self, x):
        """
        Forward pass to extract and embed the best-matching segment to the shapelet.


        Args:
        x (Tensor): Input tensor of shape (B, C, T).


        Returns:
        Tensor: Embedded shapelet representation (B, 1, shape_embed_dim).
        """
        pis = x[:, self.dim, self.start_position:self.end_position]
        ci_pis = torch.square(torch.subtract(pis[:, 1:], pis[:, :-1]))

        pis = pis.unfold(1, self.kernel_size, 1).contiguous()
        pis = pis.view(-1, self.kernel_size)

        ci_pis = ci_pis.unfold(1, self.kernel_size - 1, 1).contiguous()
        ci_pis = ci_pis.view(-1, self.kernel_size - 1)
        ci_pis = torch.sum(ci_pis, dim=1) + (1 / self.norm)

        ci_shapelet_vec = torch.ones(ci_pis.size(
            0), device=x.device, requires_grad=False)*self.ci_shapelet
        max_ci = torch.max(ci_pis, ci_shapelet_vec)
        min_ci = torch.min(ci_pis, ci_shapelet_vec)
        ci_dist = max_ci / min_ci
        ci_dist[ci_dist > self.max_ci] = self.max_ci
        dist1 = torch.sum(torch.square(pis - self.shapelet), 1)
        dist1 = dist1 * ci_dist
        dist1 = dist1 / self.shapelet.size(-1)
        dist1 = dist1.view(x.size(0), -1)

        # soft-minimum
        index = torch.argmin(dist1, dim=1)
        pis = pis.view(x.size(0), -1, self.kernel_size)
        device = x.device
        batch_idx = torch.arange(int(x.size(0)), device=device).long()
        out = pis[batch_idx, index.long()]
        out = self.l1(out)

        out_s = self.l2(self.shapelet.unsqueeze(0))

        out = out - out_s

        return out.view(x.shape[0], 1, -1)


class Shapeformer(nn.Module):
    """
    Initializes the Shapeformer model, which combines local convolutional–attention
    features with global shapelet-based representations.

    The model extracts:
    - Local features using convolutional blocks, positional encodings, and self-attention.
    - Global features through multiple ShapeBlock modules that embed shapelet descriptors.
    - Positional information for shapelets (dimension, start, end) encoded via one-hot embeddings.

    Args:
        config (dict): Dictionary containing all architectural and training hyperparameters.
        num_classes (int): Number of output classes for classification.
    """

    def __init__(self, config, num_classes):
        super().__init__()
        # Shapelet Query  ---------------------------------------------------------
        self.shapelet_info = config['shapelets_info']
        self.shapelet_info = torch.IntTensor(self.shapelet_info)
        self.shapelets = config['shapelets']

        self.sw = torch.nn.Parameter(torch.tensor(
            config['shapelets_info'][:, 3]).float(), requires_grad=True)

        # Local Information
        self.len_w = config['len_w']
        self.pad_w = self.len_w - config['len_ts'] % self.len_w
        self.pad_w = 0 if self.pad_w == self.len_w else self.pad_w
        self.height = config['ts_dim']
        self.weight = int(np.ceil(config['len_ts'] / self.len_w))

        list_d = []
        list_p = []
        for d in range(self.height):
            for p in range(self.weight):
                list_d.append(d)
                list_p.append(p)

        list_ed = position_embedding(torch.tensor(list_d))
        list_ep = position_embedding(torch.tensor(list_p))
        self.local_pos_embedding = torch.cat((list_ed, list_ep), dim=1)

        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        dim_ff = config['dim_ff']
        num_heads = config['num_heads']
        local_pos_dim = config['local_pos_dim']
        local_embed_dim = config['local_embed_dim']
        local_emb_size = local_embed_dim
        self.local_emb_size = local_emb_size

        self.local_layer = nn.Linear(self.len_w, local_embed_dim)
        self.embed_layer = nn.Sequential(nn.Conv2d(1, local_emb_size * 1, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(local_emb_size * 1),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(local_emb_size * 1, local_emb_size,
                      kernel_size=[channel_size, 1], padding='valid'),
            nn.BatchNorm2d(local_emb_size),
            nn.GELU())
        self.Fix_Position = LearnablePositionalEncoding(
            local_emb_size, dropout=config['dropout'], max_len=seq_len)
        self.local_pos_layer = nn.Linear(
            self.local_pos_embedding.shape[-1], local_pos_dim)
        self.local_ln1 = nn.LayerNorm(local_emb_size, eps=1e-5)
        self.local_ln2 = nn.LayerNorm(local_emb_size, eps=1e-5)
        self.local_attention_layer = Attention(
            local_emb_size, num_heads, config['dropout'])
        self.local_ff = nn.Sequential(
            nn.Linear(local_emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, local_emb_size),
            nn.Dropout(config['dropout']))
        self.local_gap = nn.AdaptiveAvgPool1d(1)
        self.local_flatten = nn.Flatten()

        # Global Information
        self.shape_blocks = nn.ModuleList([
            ShapeBlock(shapelet_info=self.shapelet_info[i], shapelet=self.shapelets[i],
                       shape_embed_dim=config['shape_embed_dim'], len_ts=config["len_ts"])
            for i in range(len(self.shapelet_info))])

        self.shapelet_info = config['shapelets_info']
        self.shapelet_info = torch.FloatTensor(self.shapelet_info)
        self.position = torch.index_select(
            self.shapelet_info, 1, torch.tensor([5, 1, 2]))
        # 1hot pos embedding
        self.d_position = self.position_embedding(self.position[:, 0])
        self.s_position = self.position_embedding(self.position[:, 1])
        self.e_position = self.position_embedding(self.position[:, 2])

        self.d_pos_embedding = nn.Linear(
            self.d_position.shape[1], config['pos_embed_dim'])
        self.s_pos_embedding = nn.Linear(
            self.s_position.shape[1], config['pos_embed_dim'])
        self.e_pos_embedding = nn.Linear(
            self.e_position.shape[1], config['pos_embed_dim'])

        # Parameters Initialization -----------------------------------------------
        emb_size = config['shape_embed_dim']

        self.LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)
        self.attention_layer = Attention(
            emb_size, num_heads, config['dropout'])

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size + local_emb_size, num_classes)
        self.out2 = nn.Linear(emb_size, num_classes)

        # Merge Layer----------------------------------------------------------
        self.local_merge = nn.Linear(local_emb_size, int(local_emb_size / 2))

    def position_embedding(self, position_list):
        """
        Converts a list of discrete position indices into one-hot positional encodings.

        Args:
            position_list (Tensor): 1D tensor of integer position indices.

        Returns:
            Tensor: A 2D tensor of shape (num_positions, num_unique_positions),
                where each row is a one-hot vector encoding the corresponding position.
        """

        max_d = position_list.max() + 1
        identity_matrix = torch.eye(int(max_d))
        d_position = identity_matrix[position_list.to(dtype=torch.long)]
        return d_position

    def forward(self, x, ep):
        """
        Forward pass of the Shapeformer model.

        This method processes the input time series through:
            - A local feature extraction branch using convolutions, learnable positional
            encodings, self-attention, and feed-forward layers.
            - A global branch composed of multiple shapelet-based embedding blocks, combined
            with learned positional embeddings for dimension, start, and end indices.

        The outputs of both branches are concatenated and passed through a linear classifier.

        Args:
            x (Tensor): Input time series of shape (batch_size, channels, sequence_length).
            ep (int): Epoch index or training stage indicator (used for certain shapelet behaviors).

        Returns:
            Tensor: Logits of shape (batch_size, num_classes).
        """

        local_x = x.unsqueeze(1)
        local_x = self.embed_layer(local_x)
        local_x = self.embed_layer2(local_x).squeeze(2)
        local_x = local_x.permute(0, 2, 1)
        x_src_pos = self.Fix_Position(local_x)
        local_att = local_x + self.local_attention_layer(x_src_pos)
        local_att = self.local_ln1(local_att)
        local_out = local_att + self.local_ff(local_att)
        local_out = self.local_ln2(local_out)
        local_out = local_out.permute(0, 2, 1)
        local_out = self.local_gap(local_out)
        local_out = self.local_flatten(local_out)

        # Global information
        global_x = None
        for block in self.shape_blocks:
            if global_x is None:
                global_x = block(x)
            else:
                global_x = torch.cat((global_x, block(x)), dim=1)
        if self.d_position.device != x.device:
            self.d_position = self.d_position.to(x.device)
            self.s_position = self.s_position.to(x.device)
            self.e_position = self.e_position.to(x.device)

        d_pos = self.d_position.repeat(x.shape[0], 1, 1)
        s_pos = self.s_position.repeat(x.shape[0], 1, 1)
        e_pos = self.e_position.repeat(x.shape[0], 1, 1)

        d_pos_emb = self.d_pos_embedding(d_pos)
        s_pos_emb = self.s_pos_embedding(s_pos)
        e_pos_emb = self.e_pos_embedding(e_pos)

        global_x = global_x + d_pos_emb + s_pos_emb + e_pos_emb
        global_att = global_x + self.attention_layer(global_x)
        global_att = global_att * self.sw.unsqueeze(0).unsqueeze(2)
        global_att = self.LayerNorm1(global_att)  # Choosing LN and BN
        global_out = global_att + self.FeedForward(global_att)
        global_out = self.LayerNorm2(global_out)  # Choosing LN and BN
        global_out = global_out * self.sw.unsqueeze(0).unsqueeze(2)
        global_out = global_out[:, 0, :]

        out = torch.cat((global_out, local_out), dim=1)
        out = self.out(out)

        return out


def position_embedding(position_list):
    """
    Creates a one-hot positional encoding for a list of integer positions.

    Args:
        position_list (Tensor): 1D tensor containing discrete position indices.

    Returns:
        Tensor: One-hot encoded positional matrix where each row corresponds to a position.
    """

    max_d = position_list.max() + 1
    identity_matrix = torch.eye(int(max_d))
    d_position = identity_matrix[position_list.to(dtype=torch.long)]
    return d_position


if __name__ == '__main__':
    print()
