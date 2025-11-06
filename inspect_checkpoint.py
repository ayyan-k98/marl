"""
Check what's stored in the checkpoint
"""
import torch

checkpoint_path = "./checkpoints/fcn_final.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("=" * 80)
print("CHECKPOINT CONTENTS")
print("=" * 80)
print(f"\nKeys in checkpoint:")
for key in checkpoint.keys():
    if not key.endswith('_state_dict'):
        print(f"  {key}: {checkpoint[key]}")
    else:
        print(f"  {key}: <state dict>")

print("\n" + "=" * 80)
print("CRITICAL INFO")
print("=" * 80)

# Check for probabilistic setting
if 'config' in checkpoint or 'use_probabilistic' in checkpoint or 'probabilistic' in checkpoint:
    print("\n✓ Checkpoint contains environment config")
else:
    print("\n❌ Checkpoint does NOT store environment config")
    print("   We cannot determine if it was trained with probabilistic mode")

print("\n" + "=" * 80)
