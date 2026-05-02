# GPU Run Check
import torch
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
print(f"Supported Architectures: {torch.cuda.get_arch_list()}")