"""Recurrent flood-prediction models with fixed physical constants.

Design change from the original repo
-------------------------------------
The original code declared the physics *loss weights* and *thresholds* as
``nn.Parameter``s and optimized them with the same objective they regularize.
Because the physics penalties are non-negative and enter the loss additively,
gradient descent drives the weights (and thresholds) in the penalty-shrinking
direction, annealing the constraints away (see README "What was wrong").

Here the physical constants (``thresh_rain``, ``thresh_flood``, ``q_zero_norm``)
are **non-trainable buffers** -- they encode domain knowledge and must stay put.
The strength of the physics term is controlled outside the model (a fixed weight
or a Lagrange multiplier updated by dual *ascent*; see :mod:`src.train`).
"""

import torch
import torch.nn as nn


class _PhysicsInformedRNN(nn.Module):
    """Shared implementation for the LSTM / GRU variants."""

    rnn_cls = None  # set by subclass

    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_dim,
        output_dim,
        init_rain,
        init_flood,
        q_zero_norm=0.0,
        dropout=0.0,
        precip_idx=0,
        w_eff_idx=None,
        weff_hi=0.0,
        weff_low=0.0,
        api_idx=None,
        api_hi=0.0,
        api_lo=0.0,
        recession_k=0.0,
    ):
        super().__init__()
        self.rnn = self.rnn_cls(
            input_dim,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=dropout if layer_dim > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.precip_idx = precip_idx
        # Column of the degree-day effective-input feature (None if absent).
        self.w_eff_idx = w_eff_idx
        # Column of the antecedent-moisture (API) feature (None if absent).
        self.api_idx = api_idx

        # Fixed physical constants (NOT trained). register_buffer keeps them on
        # the right device and in state_dict without exposing them to the optimizer.
        self.register_buffer("thresh_rain", torch.tensor(float(init_rain)))
        self.register_buffer("thresh_flood", torch.tensor(float(init_flood)))
        self.register_buffer("q_zero_norm", torch.tensor(float(q_zero_norm)))
        self.register_buffer("weff_hi", torch.tensor(float(weff_hi)))
        self.register_buffer("weff_low", torch.tensor(float(weff_low)))
        # Antecedent-wet / antecedent-dry levels (normalized API quantiles).
        self.register_buffer("api_hi", torch.tensor(float(api_hi)))
        self.register_buffer("api_lo", torch.tensor(float(api_lo)))
        # Master recession constant (physical Q_{t+1}=k*Q_t), 0 if unused.
        self.register_buffer("recession_k", torch.tensor(float(recession_k)))

    def _init_hidden(self, batch, device):
        return torch.zeros(self.layer_dim, batch, self.hidden_dim, device=device)

    def forward(self, x):
        raise NotImplementedError


class PhysicsInformedLSTM(_PhysicsInformedRNN):
    rnn_cls = nn.LSTM

    def forward(self, x):
        h0 = self._init_hidden(x.size(0), x.device)
        c0 = self._init_hidden(x.size(0), x.device)
        out, _ = self.rnn(x, (h0, c0))
        return self.fc(out)


class PhysicsInformedGRU(_PhysicsInformedRNN):
    rnn_cls = nn.GRU

    def forward(self, x):
        h0 = self._init_hidden(x.size(0), x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out)


def build_model(model_type, input_dim, hidden_dim, layer_dim, output_dim, meta, dropout=0.0):
    """Factory that wires a model to the dataset metadata."""
    cls = PhysicsInformedLSTM if model_type.upper() == "LSTM" else PhysicsInformedGRU
    return cls(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        output_dim=output_dim,
        init_rain=meta["init_rain"],
        init_flood=meta["init_flood"],
        q_zero_norm=meta["q_zero_norm"],
        dropout=dropout,
        precip_idx=meta.get("precip_idx", 0),
        w_eff_idx=meta.get("w_eff_idx"),
        weff_hi=meta.get("init_weff_hi", 0.0),
        weff_low=meta.get("weff_low", 0.0),
        api_idx=meta.get("api_idx"),
        api_hi=meta.get("init_api_hi", 0.0),
        api_lo=meta.get("init_api_lo", 0.0),
        recession_k=meta.get("recession_k", 0.0),
    )
