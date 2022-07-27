
import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from typing import Optional, Dict

from modules.flows.spline.cubic import cubic_spline
from modules.flows.spline.linear_rational import rational_linear_spline
from modules.flows.spline.quadratic_rational import rational_quadratic_spline


# -----------------------------------------------------------------------------------------
class IdentityLayer(nn.Module):
    def forward(self, x: Tensor, **kwargs):
        return x


# -----------------------------------------------------------------------------------------
class AffineCouplingLayer(nn.Module):

    def __init__(self, coupling: str, transform_net: nn.Module, params: Dict, split_dim=1, clamp=None):
        super(AffineCouplingLayer, self).__init__()

        assert coupling in ['additive', 'affine', 'affineEx']
        assert split_dim in [-1, 1, 2, 3]
        self.coupling  = coupling
        self.dim = split_dim

        self.bias_net = transform_net(**params)

        if self.coupling == 'affine':
            self.scale_net = transform_net(**params)
        elif self.coupling == 'affineEx':
            params_t = params
            params_t['in_channel'] = params['out_channel']
            params_t['out_channel'] = params['in_channel']
            self.g1 = transform_net(**params_t)
            self.g2 = transform_net(**params)
            self.g3 = transform_net(**params)

        self.clamping = clamp or IdentityLayer()

    def forward(self, x: Tensor, c: Tensor=None, **kwargs):

        h1, h2 = self.channel_split(x)

        if self.coupling == 'affine':
            scale = self.clamping(self.scale_net(h1, c, **kwargs))
            bias  = self.bias_net(h1, c, **kwargs)

            h2 = (h2 - bias) * torch.exp(-scale)
            log_det_J = -scale.flatten(start_dim=1).sum(1)
        elif self.coupling == 'additive':
            bias = self.bias_net(h1, c, **kwargs)
            h2 = h2 - bias
            log_det_J = None
        elif self.coupling == 'affineEx':
            scale = self.clamping(self.g2(h1, c, **kwargs))
            bias  = self.g3(h1, c, **kwargs)
    
            h1 = h1 + self.g1(h2)
            h2 = torch.exp(scale) * h2 + bias
            log_det_J = scale.flatten(start_dim=1).sum(1)
        else:
            raise NotImplementedError()

        x = self.channel_cat(h1, h2)
        return x, log_det_J

    def inverse(self, z: Tensor, c: Tensor=None, **kwargs):
        
        h1, h2 = self.channel_split(z)

        if self.coupling == 'affine':
            scale = self.clamping(self.scale_net(h1, c, **kwargs))
            bias  =  self.bias_net(h1, c, **kwargs)

            h2 = h2 * torch.exp(scale) + bias
            log_det_J = scale.flatten(start_dim=1).sum(1)
        elif self.coupling == 'additive':
            bias = self.bias_net(h1, c, **kwargs)
            h2 = h2 + bias
            log_det_J = None
        elif self.coupling == 'affineEx':
            scale = self.clamping(self.g2(h1, c, **kwargs))
            bias  = self.g3(h1, c, **kwargs)

            h2 = (h2 - bias) * torch.exp(-scale)
            h1 = h1 - self.g1(h2)
            log_det_J = -scale.flatten(start_dim=1).sum(1)
        else:
            raise NotImplementedError()

        z = self.channel_cat(h1, h2)
        return z, log_det_J

    def channel_split(self, x: Tensor):
        return torch.chunk(x, 2, dim=self.dim)

    def channel_cat(self, h1: Tensor, h2: Tensor):
        return torch.cat([h1, h2], dim=self.dim)



# -----------------------------------------------------------------------------------------
class AffineSpatialCouplingLayer(AffineCouplingLayer):

    def __init__(self, coupling, transform_net, params, is_even, split_dim, clamp=None):
        super().__init__(coupling, transform_net, params, split_dim=split_dim, clamp=clamp)
        self.is_even = is_even

    def channel_split(self, x: Tensor):
        if self.is_even:
            return torch.split(x, [1, 2], dim=self.dim)
        else:
            return torch.split(x, [2, 1], dim=self.dim)


# -----------------------------------------------------------------------------------------
class AffineInjectorLayer(AffineCouplingLayer):

    def __init__(self, coupling, transform_net, params, split_dim, clamp=None):
        super().__init__(coupling, transform_net, params, split_dim=split_dim, clamp=clamp)

    def forward(self, x: Tensor, c: Tensor, **kwargs):
        log_det_J = None

        if self.coupling == 'additive':
            x = x - self.bias_net(c)
        if self.coupling == 'affine':
            scale = self.clamping(self.scale_net(c, **kwargs))
            bias  = self.bias_net(c, **kwargs)

            x = (x - bias) * torch.exp(-scale)
            log_det_J = -torch.sum(torch.flatten(scale, start_dim=1), dim=1)

        return x, log_det_J

    def inverse(self, z: Tensor, c: Tensor, **kwargs):
        log_det_J = None

        if self.coupling == 'additive':
            z = z + self.bias_net(c)
        if self.coupling == 'affine':
            scale = self.clamping(self.scale_net(c, **kwargs))
            bias  = self.bias_net(c, **kwargs)
            z = z * torch.exp(scale) + bias
            log_det_J = torch.sum(torch.flatten(scale, start_dim=1), dim=1)
        return z, log_det_J
# -----------------------------------------------------------------------------------------




# -----------------------------------------------------------------------------------------
class SplineCouplingLayer(nn.Module):
    """Spline Coupling layer. A drop in replacement for Affine Coupling layer."""

    def __init__(self, spline, transform_net, params):
        super(SplineCouplingLayer, self).__init__()

        assert spline in ['cubic', 'quadratic', 'linear-rational']
        self.spline = spline
        self.num_bins = 64
        self.hidden_channel = params['hidden_channel']
        self.tails = 'linear'
        self.tail_bound = 5
        self.min_bin_width  = 0.001
        self.min_bin_height = 0.001
        self.min_derivative = 0.001
        self.dim = 1

        dim_multiplier = None
        if spline == 'cubic':
            dim_multiplier = self.num_bins * 2 + 2
        if spline == 'quadratic':
            dim_multiplier = self.num_bins * 3 - 1
        if spline == 'linear-rational':
            dim_multiplier = self.num_bins * 4 - 1

        params['out_channel'] *= dim_multiplier
        self.transform_net = transform_net(**params)

    def forward(self, x: Tensor, u: Optional[Tensor]=None):
        """
        x: [B, C1, N, M]
        u: [B, C2, N, M]
        """

        h1, h2 = self.channel_split(x)
        h2 = h2.permute(0, 2, 3, 1)  # [B, N, C // 2]
        params = self.transform_net(h1, u)
        params = params.reshape(h2.shape + (-1,))
        h2, log_det_J = self.piecewise_cdf(h2, params, inverse=False)
        h2 = h2.permute(0, 3, 1, 2)

        x = self.channel_cat(h1, h2)
        log_det_J = torch.sum(torch.flatten(log_det_J, start_dim=1), dim=1)
        return x, log_det_J

    def inverse(self, z: Tensor, u: Optional[Tensor]=None):

        h1, h2 = self.channel_split(z)
        h2 = h2.permute(0, 2, 3, 1)
        params = self.transform_net(h1, u)
        params = params.reshape(h2.shape + (-1,))
        h2, _ = self.piecewise_cdf(h2, params, inverse=True)
        h2 = h2.permute(0, 3, 1, 2)

        z = self.channel_cat(h1, h2)
        return z

    def piecewise_cdf(self, h2: Tensor, params: Tensor, inverse=False):
        
        unnormalized_width  = params[..., :self.num_bins]
        unnormalized_height = params[..., self.num_bins: 2 * self.num_bins]

        # Scaled down inputs to the softmax
        unnormalized_width  /= np.sqrt(self.hidden_channel)
        unnormalized_height /= np.sqrt(self.hidden_channel)

        if self.spline == 'cubic':
            unnorm_derivative_left  = params[..., 2 * self.num_bins].unsqueeze(-1)
            unnorm_derivative_right = params[..., 2 * self.num_bins + 1].unsqueeze(-1)

            return cubic_spline(h2,
                unnormalized_width, unnormalized_height, unnorm_derivative_left, unnorm_derivative_right,
                inverse, self.tails, self.tail_bound, self.num_bins,
                self.min_bin_width, self.min_bin_height)

        if self.spline == 'quadratic':
            unnormalized_derivatives = params[..., 2 * self.num_bins:]

            return rational_quadratic_spline(h2,
                unnormalized_width, unnormalized_height, unnormalized_derivatives,
                inverse, self.tails, self.tail_bound, self.num_bins,
                self.min_bin_width, self.min_bin_height, self.min_derivative)

        if self.spline == 'linear-rational':
            unnormalized_lambdas     = params[..., 2 * self.num_bins: 3 * self.num_bins]
            unnormalized_derivatives = params[..., 3 * self.num_bins:]

            return rational_linear_spline(h2,
                unnormalized_width, unnormalized_height, unnormalized_derivatives, unnormalized_lambdas,
                inverse, self.tails, self.tail_bound, self.num_bins,
                self.min_bin_width, self.min_bin_height, self.min_derivative)
    
    def channel_split(self, x: Tensor):
        return torch.chunk(x, 2, dim=self.dim)

    def channel_cat(self, h1: Tensor, h2: Tensor):
        return torch.cat([h1, h2], dim=self.dim)
# -----------------------------------------------------------------------------------------
