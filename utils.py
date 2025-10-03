import torch 
import torch.nn as nn
import os
from mobilenet import *

def evaluate(model, testloader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
        final_acc = 100 * correct / total
    return final_acc


def evaluate_top_1_and_5(model, testloader, device):
    model.eval()
    top1_correct, top5_correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Top-1 accuracy
            _, preds = outputs.topk(1, dim=1, largest=True, sorted=True)
            top1_correct += (preds.squeeze() == targets).sum().item()

            # Top-5 accuracy
            _, top5_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
            top5_correct += sum([targets[i] in top5_preds[i] for i in range(len(targets))])

            total += targets.size(0)

    top1_acc = 100 * top1_correct / total
    top5_acc = 100 * top5_correct / total
    
    print(f"Final Test Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Final Test Top-5 Accuracy: {top5_acc:.2f}%")

    return top1_acc, top5_acc



def print_model_size(model, label=""):
    """Prints the size of a model in MB."""
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    print(f"{label} Model size: {size_mb:.2f} MB")
    os.remove("temp.p")
    
def get_model_size(model):
    """Returns the size of a model in MB."""
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    # print(f"{label} Model size: {size_mb:.2f} MB")
    os.remove("temp.p")
    return size_mb
    

def find_parent_module_and_name_final(model, target_module):
    """
    Finds the parent of a target module by robustly matching its parameters.
    This is resilient to the use of copy.deepcopy().
    """
    # Iterate through all potential parent modules in the model
    for parent_name, parent_module in model.named_modules():
        # Iterate through the direct children of the potential parent
        for child_name, child_module in parent_module.named_children():
            
            # Use our new robust comparison function instead of a fragile 'is' check
            if are_modules_equal(child_module, target_module):
                # If we find an exact match based on parameters, we've found our layer
                return parent_module, child_name
    
    # If no match is found after checking the entire model
    return None, None


def find_prunable_blocks(model, min_channels=32):
    """
    Finds InvertedResidual blocks that are suitable for internal pruning.
    Suitable blocks are those with an expansion layer.
    """
    prunable_blocks = []
    print("\n--- Searching for prunable InvertedResidual blocks ---")
    for module in model.features:
        if isinstance(module, InvertedResidual) and len(module.conv) == 8:
             # len(module.conv) == 8 means it has an expansion layer
             if module.conv[0].in_channels >= min_channels:
                 prunable_blocks.append(module)
    
    print(f"Found {len(prunable_blocks)} prunable blocks.")
    return prunable_blocks


def prune_inverted_residual_block(block, amount=0.5):
    """
    Performs safe, internal pruning on a single InvertedResidual block.
    """
    if not (isinstance(block, InvertedResidual) and len(block.conv) == 8):
        print("Warning: Skipping block, as it is not a prunable InvertedResidual block.")
        return

    # 1. Identify all the layers we need to modify within the block
    expansion_conv = block.conv[0]
    expansion_bn = block.conv[1]
    depthwise_conv = block.conv[3]
    depthwise_bn = block.conv[4]
    projection_conv = block.conv[6]
    
    print(f"\n--- Pruning block with input channels: {expansion_conv.in_channels} ---")

    # 2. Score the filters of the expansion layer
    weights = expansion_conv.weight.data
    l1_norms = torch.sum(torch.abs(weights), dim=(1, 2, 3))
    
    # 3. Get the indices of the expanded channels to KEEP
    num_channels_to_keep = int(l1_norms.shape[0] * (1 - amount))
    keep_indices, _ = torch.sort(torch.topk(l1_norms, num_channels_to_keep).indices)

    # 4. Create the new, smaller layers
    new_expansion_conv = nn.Conv2d(expansion_conv.in_channels, num_channels_to_keep, 1, bias=False)
    new_expansion_bn = nn.BatchNorm2d(num_channels_to_keep)
    new_depthwise_conv = nn.Conv2d(num_channels_to_keep, num_channels_to_keep, 3, 
                                   stride=depthwise_conv.stride, padding=1, 
                                   groups=num_channels_to_keep, bias=False)
    new_depthwise_bn = nn.BatchNorm2d(num_channels_to_keep)
    new_projection_conv = nn.Conv2d(num_channels_to_keep, projection_conv.out_channels, 1, bias=False)

    # 5. Carefully copy the weights for the preserved channels
    new_expansion_conv.weight.data = expansion_conv.weight.data[keep_indices, :, :, :]
    new_expansion_bn.weight.data = expansion_bn.weight.data[keep_indices]
    new_expansion_bn.bias.data = expansion_bn.bias.data[keep_indices]
    new_expansion_bn.running_mean.data = expansion_bn.running_mean.data[keep_indices]
    new_expansion_bn.running_var.data = expansion_bn.running_var.data[keep_indices]

    new_depthwise_conv.weight.data = depthwise_conv.weight.data[keep_indices, :, :, :]
    new_depthwise_bn.weight.data = depthwise_bn.weight.data[keep_indices]
    new_depthwise_bn.bias.data = depthwise_bn.bias.data[keep_indices]
    new_depthwise_bn.running_mean.data = depthwise_bn.running_mean.data[keep_indices]
    new_depthwise_bn.running_var.data = depthwise_bn.running_var.data[keep_indices]

    new_projection_conv.weight.data = projection_conv.weight.data[:, keep_indices, :, :]

    # 6. Replace the old layers with the new layers in the block's sequential module
    block.conv[0] = new_expansion_conv
    block.conv[1] = new_expansion_bn
    block.conv[3] = new_depthwise_conv
    block.conv[4] = new_depthwise_bn
    block.conv[6] = new_projection_conv
    
    print(f"Successfully pruned block. Expanded channels reduced to {num_channels_to_keep}.")


def are_modules_equal(mod1, mod2):
    """
    Helper function to robustly compare two modules by checking if all their
    parameters are identical.
    """
    # First, check if they are the same type of layer
    if type(mod1) != type(mod2):
        return False
    
    # Compare their state dictionaries, which hold all parameters
    sd1 = mod1.state_dict()
    sd2 = mod2.state_dict()
    
    # Check if they have the same set of parameters
    if sd1.keys() != sd2.keys():
        return False
    
    # Check if each parameter tensor is identical
    for key in sd1.keys():
        if not torch.equal(sd1[key], sd2[key]):
            return False
            
    # If all checks pass, the modules are considered equal
    return True