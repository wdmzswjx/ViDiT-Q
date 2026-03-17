import torch
import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.quarot.quarot_utils import random_hadamard_matrix, matmul_hadU_cuda

class QuarotQuantizedLinear(QuantizedLinear):
    """
    the base quantized linear layer,
    adpot the static weight quantization,
    and the dynamic activation quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device: None,
        quant_config: dict,
        fp_module: torch.nn.Linear,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, quant_config, fp_module)

        self.rotation_matrix = None   # init so could be load in quant_params

    def get_rotation_matrix(self):
        device = self.fp_module.weight.device
        self.rotation_matrix = random_hadamard_matrix(self.in_features, device)

    def update_quantized_weight_rotated(self):
        device = self.fp_module.weight.device
        if self.rotation_matrix.device != device:
            self.rotation_matrix = self.rotation_matrix.to(device)
        self.w_quantizer.init_done = False   # unset the init done to overwrite quant_params
        self.weight.data = self.w_quantizer(torch.matmul(self.fp_module.weight.data.to(device).double(), self.rotation_matrix).float())
        self.w_quantizer.init_done = True

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        input shape: [B,N_token,C]
        """
        if not self.quant_mode:  # use the FP
            return self.fp_module(x, *args, **kwargs)
        else:
            # reshape X into [G, -1] 
            B, N_token, C = x.shape
            x = torch.matmul(x.double(), self.rotation_matrix).to(dtype=x.dtype)
            x = x.reshape([B*N_token,-1])

            # quantize activation
            x = self.a_quantizer(x)
            x = x.reshape([B, N_token, C])

            y = F.linear(x, self.weight.to(x.dtype), self.bias, *args, **kwargs)

            return y



