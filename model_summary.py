import sys
sys.path.append("/home/meijian/Time_series_big_model/toto-main/toto2/toto2") 

import torch
from model import Toto2Model

SIZE = "4m"  # 4m | 22m | 313m | 1B | 2.5B
CHECKPOINT = f"/toto-main/Toto-2.0-4m"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Toto2Model.from_pretrained(CHECKPOINT, map_location=device)
model = model.to(device).eval()

print(f"Loaded {CHECKPOINT}: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Patch size: {model.config.patch_size}")

# Print model architecture
print("\n" + "="*80)
print("MODEL ARCHITECTURE")
print("="*80)
print(model)

# Print model parameters
print("\n" + "="*80)
print("MODEL PARAMETERS")
print("="*80)
total_params = 0
for name, param in model.named_parameters():
    num_params = param.numel()
    total_params += num_params
    print(f"{name:60s} | shape: {str(list(param.shape)):20s} | params: {num_params:>12,}")
print("-"*80)
print(f"{'TOTAL':60s} | {'':20s} | params: {total_params:>12,}")

# Print model layers summary (hierarchical)
print("\n" + "="*80)
print("MODEL LAYERS SUMMARY")
print("="*80)
print(f"{'Module':60s} | {'Type':35s} | {'Params':>12s}")
print("-"*110)
for name, module in model.named_modules():
    if name == "":
        continue
    mod_params = sum(p.numel() for p in module.parameters(recurse=False))
    mod_type = type(module).__name__
    print(f"{name:60s} | {mod_type:35s} | {mod_params:>12,}")

# Write model details to file
model_details_file = "/home/meijian/Time_series_big_model/toto-main/toto2/model_details.txt"
with open(model_details_file, "w") as f:
    f.write("="*80 + "\n")
    f.write("MODEL DETAILS REPORT\n")
    f.write("="*80 + "\n\n")
    
    # Basic info
    f.write("BASIC INFO\n")
    f.write("-"*80 + "\n")
    f.write(f"Checkpoint: {CHECKPOINT}\n")
    f.write(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    f.write(f"Patch size: {model.config.patch_size}\n")
    f.write(f"Device: {device}\n\n")
    
    # Model architecture
    f.write("="*80 + "\n")
    f.write("MODEL ARCHITECTURE\n")
    f.write("="*80 + "\n")
    f.write(str(model) + "\n\n")
    
    # Model parameters
    f.write("="*80 + "\n")
    f.write("MODEL PARAMETERS\n")
    f.write("="*80 + "\n")
    f.write(f"{'Parameter Name':60s} | {'Shape':20s} | {'Params':>12s}\n")
    f.write("-"*80 + "\n")
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        f.write(f"{name:60s} | {str(list(param.shape)):20s} | {num_params:>12,}\n")
    f.write("-"*80 + "\n")
    f.write(f"{'TOTAL':60s} | {'':20s} | {total_params:>12,}\n\n")
    
    # Model layers summary
    f.write("="*80 + "\n")
    f.write("MODEL LAYERS SUMMARY\n")
    f.write("="*80 + "\n")
    f.write(f"{'Module':60s} | {'Type':35s} | {'Params':>12s}\n")
    f.write("-"*110 + "\n")
    for name, module in model.named_modules():
        if name == "":
            continue
        mod_params = sum(p.numel() for p in module.parameters(recurse=False))
        mod_type = type(module).__name__
        f.write(f"{name:60s} | {mod_type:35s} | {mod_params:>12,}\n")

print(f"\nModel details written to: {model_details_file}")
