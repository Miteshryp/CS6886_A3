# ==============================================================================
# 2. FROM-SCRATCH QUANTIZATION COMPONENTS
# ==============================================================================

import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

from utils import *

# def calculate_qparams(tensor):
#     q_min, q_max = 0, 255 # Using UINT8 for activations as is common
#     t_min, t_max = tensor.min(), tensor.max()
#     scale = (t_max - t_min) / (q_max - q_min)
#     # Ensure scale is not zero
#     if scale == 0.0:
#         scale = 1e-8
#     zero_point = q_min - t_min / scale
#     return scale.item(), int(zero_point)

# This is a building block for Quantization
"""
NOTE: For FP4 and FP8, we are forced to simulate the effect of
quantization since pytorch does not have native support for
FP4 and FP8 data types.
"""
class UnifiedQuantizedLayer(nn.Module):
    def __init__(self, original_layer, weight_bits=8, activation_bits=8, act_qparams=None):
        super().__init__()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits

        
        # Copy necessary attributes from original layer
        if isinstance(original_layer, nn.Conv2d):
            self.stride, self.padding, self.dilation, self.groups = \
                original_layer.stride, original_layer.padding, original_layer.dilation, original_layer.groups
        self.bias = original_layer.bias

        # Quantize Weights based on mode
        weight_fp32 = original_layer.weight.data
   
        if self.weight_bits == 16:
            # Native FP16 is supported in torch
            self.quantized_weight = weight_fp32.to(torch.float16)
        
        elif self.weight_bits in [8, 4]:
            # Simlulating low precision FP
            self.q_levels = 2**self.weight_bits
            min_val, max_val = weight_fp32.min(), weight_fp32.max()
            scale = (max_val - min_val) / (self.q_levels - 1)
            
            self.register_buffer('weight_min_val', min_val)
            self.register_buffer('weight_scale', scale)
            
            indices = torch.round((weight_fp32 - min_val) / scale)
            
            # Store as uint8 to save space
            # NOTE: Because we are storing both 4 bit and 8 bit as uint8,
            # the space occupied by them both is same.
            # But we will see some difference in accuracy
            self.weight_indices = indices.to(torch.uint8) 
        else:
            raise ValueError("Unsupported num_bits or mode")

    def forward(self, x):
        # Dequantize and compute based on mode
        
        if self.weight_bits == 16:
            # FP16 Forward
            dq_weight = self.quantized_weight
            dq_x = x.to(torch.float16)
            dq_bias = self.bias.to(torch.float16) if self.bias is not None else None
        
        elif self.weight_bits in [8, 4]:
            # Simulated FP8/FP4 Forward pass
            dq_weight = self.weight_min_val + self.weight_indices.float() * self.weight_scale
            
            # Simulate activation quantization on the fly 
            act_min, act_max = x.min(), x.max()
            act_scale = (act_max - act_min) / (self.q_levels - 1)
            x_indices = torch.round((x - act_min) / act_scale)
            dq_x = act_min + x_indices * act_scale
            dq_bias = self.bias
            
            
        # Quantizing Activations
        if self.activation_bits == 32:
            dq_x = x
        elif self.activation_bits == 16:
            dq_x = x.to(torch.float16)
        else: # FP8/4 simulation
            q_levels = 2**self.activation_bits
            act_min, act_max = x.min(), x.max()
            act_scale = (act_max - act_min) / (q_levels - 1)
            if act_scale == 0.0: act_scale = 1e-8
            x_indices = torch.round((x - act_min) / act_scale)
            dq_x = act_min + x_indices * act_scale
            
        
        op_dtype = torch.float32 if dq_x.dtype == torch.float32 or dq_weight.dtype == torch.float32 else torch.float16
        
        op_input = dq_x.to(op_dtype)
        op_weight = dq_weight.to(op_dtype)
        op_bias = self.bias.to(op_dtype) if self.bias is not None else None


        # Perform the operation
        if hasattr(self, 'groups'): # Conv2d
            output = F.conv2d(op_input, op_weight, dq_bias, self.stride, self.padding, self.dilation, self.groups)
        else: # Linear
            output = F.linear(op_input, op_weight, op_bias)
        
        return output.to(torch.float32)


def manual_quantize(model, weight_bits=8, activation_bits=8, is_integer_quant=False):
    print(f"\n Manually Quantizing to FP{weight_bits}")
    q_model = copy.deepcopy(model)
    q_model.eval()
    
    # FP16 is simple, no calibration needed
    if weight_bits == 16:
        for name, module in q_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                quantized_layer = UnifiedQuantizedLayer(module, weight_bits, activation_bits=activation_bits)
                parent, child_name = find_parent_module_and_name_final(q_model, module)
                if parent: setattr(parent, child_name, quantized_layer)
        return q_model
    
    
    # For FP4 and FP8, UnifiedQuantizedLayer calibration is needed
    for name, module in q_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            act_qparams = None
            quantized_layer = UnifiedQuantizedLayer(module, weight_bits, activation_bits, act_qparams)
            parent, child_name = find_parent_module_and_name_final(q_model, module)
            if parent: setattr(parent, child_name, quantized_layer)
            
    print("Quantization setup complete.")
    return q_model