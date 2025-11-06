"""
Test if agent produces consistent results with fixed random seed
This verifies the checkpoint is stable and environment is deterministic
"""
import numpy as np
import torch
import random
from fcn_agent import FCNAgent
from environment import CoverageEnvironment
from config import config

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

print("=" * 80)
print("DETERMINISTIC BEHAVIOR TEST")
print("=" * 80)
print("Running same episode 5 times with fixed seed")
print("Results should be IDENTICAL if environment is deterministic")
print("=" * 80)
print()

checkpoint_path = "./checkpoints/fcn_final.pt"

results = []

for run in range(5):
    print(f"\n--- Run {run+1}/5 ---")
    
    # Set seed before EVERYTHING
    set_seed(42)
    
    # Create fresh agent and environment
    agent = FCNAgent(grid_size=20, input_channels=6)
    agent.load(checkpoint_path)
    agent.set_epsilon(0.1)
    agent.policy_net.eval()
    
    dummy_occupancy = np.zeros((20, 20), dtype=np.float32)
    
    # Create environment
    env = CoverageEnvironment(grid_size=20, map_type='empty')
    state = env.reset()
    
    # Run episode
    max_steps = 150
    step_count = 0
    
    for step in range(max_steps):
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, env.world_state, agent_occupancy=dummy_occupancy, valid_actions=valid_actions)
        state, reward, done, info = env.step(action)
        step_count += 1
        
        if done:
            break
    
    final_coverage = info.get('coverage_pct', 0.0)
    results.append({
        'coverage': final_coverage,
        'steps': step_count,
        'robot_pos': env.robot_state.position
    })
    
    print(f"Coverage: {final_coverage:.4f}")
    print(f"Steps: {step_count}")
    print(f"Final position: {env.robot_state.position}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

coverages = [r['coverage'] for r in results]
steps = [r['steps'] for r in results]

print(f"\nCoverage values:")
for i, cov in enumerate(coverages):
    print(f"  Run {i+1}: {cov:.6f}")

print(f"\nStep counts:")
for i, s in enumerate(steps):
    print(f"  Run {i+1}: {s}")

# Check if identical
all_same = len(set(coverages)) == 1
all_steps_same = len(set(steps)) == 1

print(f"\n{'='*80}")
if all_same and all_steps_same:
    print("✅ PASS: Results are IDENTICAL across all runs")
    print("   Environment and agent are deterministic")
    print("   Checkpoint is stable")
else:
    print("❌ FAIL: Results differ between runs!")
    print(f"   Coverage variance: {np.std(coverages):.6f}")
    print(f"   Step variance: {np.std(steps):.2f}")
    print()
    print("   This suggests:")
    print("   - Environment has non-deterministic elements")
    print("   - Agent uses non-seeded randomness")
    print("   - OR epsilon-greedy exploration is active")
    
print("=" * 80)
