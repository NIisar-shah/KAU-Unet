"""
Kolmogorov-Arnold linear layer with B-spline parameterization.

Implements the layer mapping:
    y = W_base @ silu(x) + W_spline @ B(tanh(x))

where B(.) are B-spline basis functions of given order on a uniform grid.
The tanh bounding is the methodological contribution of the present
work and addresses the out-of-grid muting failure mode described in
Section 3.2.3 of the manuscript. Set use_tanh=False to ablate this
modification (Ablation A in Section 4.X of the manuscript).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        use_tanh=True,  # ABLATION A flag
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.use_tanh = use_tanh

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h
             + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features,
                         grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight,
                                 a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features,
                            self.out_features) - 1 / 2)
                * self.scale_noise / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline
                 if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order:-self.spline_order],
                    noise))
        if self.enable_standalone_scale_spline:
            nn.init.kaiming_uniform_(self.spline_scaler,
                                     a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x):
        assert x.dim() >= 2 and x.size(-1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k + 1)])
                / (grid[:, k:-1] - grid[:, :-(k + 1)])
                * bases[..., :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)])
                * bases[..., 1:]
            )
        return bases

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x):
        # Base path always uses the unbounded input (gradient flows
        # through SiLU even for tokens in the saturated tanh region).
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # Spline path: tanh-bound unless ablated.
        if self.use_tanh:
            x_spline = torch.tanh(x)
        else:
            x_spline = x

        spline_output = F.linear(
            self.b_splines(x_spline).view(*x.shape[:-1], -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output