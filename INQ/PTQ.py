# Reload model
model = AlexNetCIFAR().to(device)
model.load_state_dict(torch.load("alexnet_fp32.pth"))
model.eval()  # Very important: eval mode disables Dropout, BatchNorm stats updates

# --------- PTQ Quantization Functions ---------
def quantize_static(x, bits=8):
    """Simple uniform quantization."""
    levels = 2 ** bits
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (levels - 1)
    x_quant = torch.round((x - x_min) / scale) * scale + x_min
    return x_quant

def quantize_model_weights(model):
    """Quantize all weights in the model."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data = quantize_static(param.data, bits=8)
    return model

# --------- Quantization Wrapper for Activations ---------
class QuantizedAlexNetCIFAR(nn.Module):
    def __init__(self, fp32_model):
        super(QuantizedAlexNetCIFAR, self).__init__()
        self.features = fp32_model.features
        self.classifier = fp32_model.classifier

    def forward(self, x):
        x = quantize_static(x, bits=8)  # Quantize input
        x = self.features(x)
        x = quantize_static(x, bits=8)  # Quantize features output
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
def main():
    # Apply weight quantization
    model = quantize_model_weights(model)

    # Wrap for activation quantization
    model = QuantizedAlexNetCIFAR(model)
    model.to(device)
    acc = test(model, device, testloader)
    print(f"Test Accuracy after PTQ (W8A8): {acc:.2f}%")

if __name__ == '__main__':
    main()