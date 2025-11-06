"""
Diagnostic: Check if network representations are truly grid-size invariant
"""
import torch
import numpy as np
from fcn_agent import FCNAgent

print("=" * 80)
print("NETWORK GRID-SIZE INVARIANCE DIAGNOSTIC")
print("=" * 80)
print()

# Load checkpoint
checkpoint_path = "./checkpoints/fcn_final.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Test on different grid sizes with SAME semantic input
grid_sizes = [20, 25, 30, 35, 40]

print("\nCreating semantically identical inputs at different resolutions...")
print("(Empty grid with agent at center)\n")

results = {}

for grid_size in grid_sizes:
    # Create agent
    agent = FCNAgent(grid_size=grid_size, input_channels=6, device=device)
    agent.load(checkpoint_path)
    agent.policy_net.eval()
    
    # Create a simple test input: empty grid with agent at center
    batch_size = 1
    test_input = torch.zeros(batch_size, 6, grid_size, grid_size, device=device)
    
    # Add agent at center
    center = grid_size // 2
    test_input[0, 2, center, center] = 1.0  # Agent position channel
    
    # Add some visited cells around agent (proportional to grid size)
    visit_radius = max(1, grid_size // 10)
    for dy in range(-visit_radius, visit_radius+1):
        for dx in range(-visit_radius, visit_radius+1):
            y, x = center + dy, center + dx
            if 0 <= y < grid_size and 0 <= x < grid_size:
                test_input[0, 0, y, x] = 1.0  # Visited
                test_input[0, 1, y, x] = 0.8  # Coverage
    
    # Forward pass
    with torch.no_grad():
        q_values = agent.policy_net(test_input)
    
    # Get action probabilities (softmax over Q-values)
    action_probs = torch.softmax(q_values, dim=-1)[0].cpu().numpy()
    
    results[grid_size] = {
        'q_values': q_values[0].cpu().numpy(),
        'action_probs': action_probs,
        'best_action': int(q_values[0].argmax())
    }
    
    print(f"Grid {grid_size}×{grid_size}:")
    print(f"  Q-values: [{', '.join([f'{q:.3f}' for q in results[grid_size]['q_values']])}]")
    print(f"  Best action: {results[grid_size]['best_action']}")
    print(f"  Action probs: [{', '.join([f'{p:.3f}' for p in action_probs])}]")
    print()

print("=" * 80)
print("ANALYSIS")
print("=" * 80)

# Check consistency of best actions
best_actions = [results[sz]['best_action'] for sz in grid_sizes]
best_action_consistency = len(set(best_actions))

print(f"\nBest actions: {best_actions}")
print(f"Unique best actions: {best_action_consistency}")

if best_action_consistency == 1:
    print("✓ Network chooses SAME action across all grid sizes")
else:
    print("✗ Network chooses DIFFERENT actions at different grid sizes!")
    print("  This indicates spatial softmax is NOT providing invariance")

# Check Q-value correlation
print("\nQ-value correlation between grid sizes:")
baseline_q = results[20]['q_values']
for grid_size in [25, 30, 35, 40]:
    test_q = results[grid_size]['q_values']
    correlation = np.corrcoef(baseline_q, test_q)[0, 1]
    print(f"  20×20 vs {grid_size}×{grid_size}: {correlation:.3f}")
    if correlation < 0.7:
        print(f"    ⚠️ LOW CORRELATION! Network sees different patterns")

# Check action probability distribution
print("\nAction probability variance across grid sizes:")
action_names = ['N', 'E', 'S', 'W', 'NE', 'NW', 'SE', 'SW', 'STAY']
for action_idx in range(9):
    probs = [results[sz]['action_probs'][action_idx] for sz in grid_sizes]
    variance = np.var(probs)
    print(f"  {action_names[action_idx]:4s}: var={variance:.4f}  [{', '.join([f'{p:.2f}' for p in probs])}]")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if best_action_consistency == 1:
    print("\n✓ Spatial softmax IS providing action-level invariance")
    print("  The network chooses the same action regardless of grid size")
    print()
    print("  Performance degradation is likely due to:")
    print("  1. Different coverage dynamics at larger scales")
    print("  2. Accumulation of small errors over longer episodes")
    print("  3. Sensor range not perfectly scaled")
    print("  4. Environment differences (map generation, obstacles)")
else:
    print("\n✗ Spatial softmax is NOT providing invariance")
    print("  The network sees different spatial patterns at different scales")
    print()
    print("  Possible causes:")
    print("  1. Coordinate grids not being scaled correctly")
    print("  2. Global features breaking invariance")
    print("  3. Network architecture issue")
    print("  4. Training data distribution mismatch")

print("=" * 80)
