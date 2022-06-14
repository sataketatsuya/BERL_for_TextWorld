import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim, Tensor
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ActorNetwork(nn.Module):
    def __init__(self, device, config, critic,
                d_model=512, nhead=8, d_hid=2048, nlayers=6, dropout=0.1,
                chkpt_dir='/tmp/ppo'):
        super(ActorNetwork, self).__init__()
        torch.manual_seed(42)  # For reproducibility
        self.device = device
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        input_size   = config['model']['max_seq_length']
        hidden_size  = config['model']['hidden_size']
        alpha  = config['training']['optimizer']['alpha']

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(input_size, d_model)
        self.d_model = d_model

        self.state_gru    = nn.GRU(d_model, d_model)
        self.hidden_size  = hidden_size
        self.state_hidden = None

        # Critic to determine a value for the current state
        self.critic = critic

        # Scorer for the commands
        self.att_cmd = nn.Sequential(
                nn.Linear(d_model*2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)

    def forward(self, state, commands, **kwargs):
        # Transformer Encoder for context state
        state_src = self.encoder(state) * math.sqrt(self.d_model)
        state_src = self.pos_encoder(state_src)
        state_output = self.transformer_encoder(state_src) # torch.Size([1, 1, 512])

        if self.state_hidden is None:
            self.state_hidden = torch.zeros(1, 1, self.d_model, device=self.device)

        state_output, self.state_hidden = self.state_gru(state_output, self.state_hidden) # torch.Size([1, 1, 512]) torch.Size([1, 1, 512])
        observation_hidden = self.state_hidden.squeeze()

        value = self.critic(state_output).squeeze().item()

        # Transformer Encoder for commands
        cmd_src = self.encoder(commands.permute(1, 0)) * math.sqrt(self.d_model)
        cmd_src = self.pos_encoder(cmd_src)
        cmd_output = self.transformer_encoder(cmd_src) # torch.Size([cmd_len, 1, 512])

        # Concatenate the observed state and command encodings.
        observation_hidden = torch.stack([observation_hidden] * commands.size(1)) # torch.Size([66, 512])
        cmd_selector_input = torch.cat([cmd_output.squeeze(1), observation_hidden], dim=-1) # torch.Size([66, 1024])

        # compute a score for each of the commands
        scores = self.att_cmd(cmd_selector_input).squeeze() # torch.Size([66])
        if len(scores.shape) == 0:
            # if only one admissible_command
            scores = scores.unsqueeze(0)
        prob = F.softmax(scores, dim=0)
        dist = Categorical(prob)

        return dist, value

    def reset_hidden(self):
        self.state_hidden = None

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.pe[:x.size(0)]
        return self.dropout(x)


class CriticNetwork(nn.Module):
    def __init__(self, device, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='/tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.device = device
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
