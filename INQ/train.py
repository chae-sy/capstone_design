import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# -------- INQ Quantization Function --------
def quantize_to_pow2(tensor):
    """Quantize weights to the nearest power of 2."""
    tensor_sign = torch.sign(tensor)
    tensor_abs = tensor.abs()
    tensor_log2 = tensor_abs.log2()
    tensor_log2_rounded = tensor_log2.round()
    quantized_abs = (2 ** tensor_log2_rounded)
    return tensor_sign * quantized_abs

# -------- INQ Mask Update Function --------
def update_fixed_mask(weight, fixed_mask, prune_percent):
    """Fix a percentage of weights (magnitude based)."""
    # Only unfixed weights are considered
    unfixed_weights = weight[fixed_mask == 0]
    k = int(prune_percent * unfixed_weights.numel())
    if k == 0:
        return fixed_mask  # nothing to fix
    
    # Find threshold
    threshold = unfixed_weights.abs().kthvalue(k).values.item()
    
    # Update mask: fix weights larger than threshold
    new_fixed = (weight.abs() >= threshold) & (fixed_mask == 0)
    fixed_mask = fixed_mask | new_fixed
    return fixed_mask

# -------- Training Loop --------
def train(model, device, trainloader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # After optimizer step, apply INQ
        model.apply_inq()

        running_loss += loss.item()
        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {running_loss/100:.4f}')
            running_loss = 0.0

    avg_loss = total_loss / num_batches
    return avg_loss  # <<< return average train loss for this epoch


# -------- Evaluation --------
def test(model, device, testloader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    avg_loss = total_loss / num_batches
    print(f'Test Accuracy: {acc:.2f}%')
    return avg_loss, acc


