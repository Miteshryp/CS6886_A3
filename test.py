import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_cifar10
from utils import *
from mobilenet import *
import copy

from utils import *
from quantize import *

import argparse
import wandb

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantization')
    parser.add_argument('--weight_quant_bits',type=int,default=8,help='Bits to Quantize the weights')
    parser.add_argument('--activation_quant_bits',type=int,default=8,help='Bits to Quantize the weights')
    
    args = parser.parse_args()
    
    # WandB Setup
    run = wandb.init(
        project="cs6886-assignment3-mobilenet", # Give your project a name
        config={
            "weight_quant_bits": args.weight_quant_bits,
            "activation_quant_bits": args.activation_quant_bits,
            "optimizer": "SGD",
            "learning_rate": 0.01,
        }
    )
    
    weight_bits = wandb.config.weight_quant_bits
    activation_bits = wandb.config.activation_quant_bits
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_model = MobileNetV2(num_classes=10).to(device)
    original_model.load_state_dict(torch.load('./checkpoints/mobilenetv2_cifar10.pth',weights_only=True, map_location=device_name))
    original_model.to(device)
    
    
    model_to_prune = copy.deepcopy(original_model)
    train_loader, test_loader = get_cifar10(batchsize=2)
    
    top_1, top_5 = evaluate_top_1_and_5(original_model, test_loader, device)
    
    # Monitoring original model accuracy for comparative analysis
    original_test_acc = evaluate(original_model, test_loader, device)
    print(f"Original Test Acc={original_test_acc:.2f}%")
    original_model_size = get_model_size(original_model)

    print_model_size(original_model, "Original")
    
    # Finding all blocks in MobileNet-v2 that can be pruned
    # These are 1x1 convolutions which have high channel redundancy
    prunable_blocks = find_prunable_blocks(model_to_prune, 96)
    
    # Apply pruning on all layers selected based on threshold criteria
    if prunable_blocks:        
        for target_block in prunable_blocks:
            prune_inverted_residual_block(target_block, amount=0.5)
    print_model_size(model_to_prune, "Pruned")
    
    # Testing Pruned model accuracy
    pruned_test_acc = evaluate(model_to_prune, test_loader, device)
    print(f"Pruned Test Acc={pruned_test_acc:.2f}%")
    
    # Quantization
    quantized_pruned_model = manual_quantize(model_to_prune, weight_bits=weight_bits, activation_bits=activation_bits)
    print_model_size(quantized_pruned_model, f"Quantized ({weight_bits} bits)")
    
    # Evaluating Pruned + Quantised model
    quantised_model_size = get_model_size(quantized_pruned_model)
    quantized_accuracy = evaluate(quantized_pruned_model, test_loader, device)
    print(f"Quantized + Pruned Test Acc={quantized_accuracy:.2f}%")
    
    compression_ratio = original_model_size / quantised_model_size
    
    
    # Logging to WandB for Parallel Coordinate Graph
    wandb.log({
        "weight_quant_bits": weight_bits,
        "activation_quant_bits": activation_bits,
        "quantized_acc": quantized_accuracy,
        "model_size_mb": quantised_model_size,
        "compression_ratio": compression_ratio
    })
    
    wandb.finish()
    

