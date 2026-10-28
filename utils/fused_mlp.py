"""Native bias-free tiny-cuda-nn MLP with explicit input padding."""
import importlib
import torch
from torch import nn
from torch.nn import functional as F


class FusedMLP(nn.Module):
    def __init__(self, inputs, width, outputs, seed=1337):
        super().__init__()
        self.inputs = inputs
        self.width = width
        self.outputs = outputs
        self.padded_inputs = ((inputs+15)//16)*16
        tcnn = importlib.import_module('tinycudann')
        try:
            self.network = tcnn.Network(self.padded_inputs, outputs, {
                'otype': 'FullyFusedMLP', 'activation': 'ReLU',
                'output_activation': 'None', 'n_neurons': width,
                'n_hidden_layers': 2}, seed=seed)
        except RuntimeError as error:
            raise RuntimeError('FullyFusedMLP unavailable; rebuild with bash scripts/install_tcnn.sh') from error
        self.network.jit_fusion = False
        expected = width*self.padded_inputs+width*width+((outputs+15)//16)*16*width
        if self.network.params.numel() != expected:
            raise RuntimeError('Unsupported FullyFusedMLP parameter layout')

    def _input(self, x):
        if not x.is_cuda or x.device != self.network.params.device:
            raise ValueError('FullyFusedMLP requires inputs and parameters on the same CUDA device')
        return F.pad(x.float(), (0, self.padded_inputs-self.inputs)).contiguous()

    def forward(self, x):
        return self.network(self._input(x)).float()

    @torch.no_grad()
    def activity(self, x):
        """Reconstruct hidden activations for diagnostics; render with native TCNN.

        TCNN does not expose hidden states. FP16 Torch accumulation can differ
        from fused kernels near zero, so report this as a reconstructed probe.
        """
        params = self.network.params.detach().to(self.network.dtype)
        a = self.width*self.padded_inputs
        b = a+self.width*self.width
        first = F.linear(self._input(x).to(params.dtype), params[:a].reshape(self.width, self.padded_inputs))
        second = F.linear(F.relu(first), params[a:b].reshape(self.width, self.width))
        return first, second, F.relu(second), self(x)

    def hidden_gradient_norm(self):
        grad = self.network.params.grad
        if grad is None:
            return 0.0
        a = self.width*self.padded_inputs
        b = a+self.width*self.width
        return float(grad[:a].norm())+float(grad[a:b].norm())
