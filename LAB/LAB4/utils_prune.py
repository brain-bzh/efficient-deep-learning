import torch
import torch.nn.utils.prune as prune
import torch.nn as nn

def apply_global_pruning(model, amount=0.2):
    """Applies global unstructured pruning based on L1 norm."""
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)

def apply_structured_pruning(model, amount=0.2):
    """Applies structured pruning to convolutional filters."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)

def apply_thinet_pruning(model, amount=0.2):
    """ThiNet-style pruning based on feature map norms."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)

def remove_pruning(model):
    """Removes pruning masks to finalize model structure."""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight_orig'):
            prune.remove(module, 'weight')
