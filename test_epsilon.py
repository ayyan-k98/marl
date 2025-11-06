"""
Test agent with different epsilon values
Maybe epsilon=0.1 (10% random) is hurting performance
"""
import numpy as np
import torch
import random
from fcn_agent import FCNAgent
from environment import CoverageEnvironment
from config import config

print("=" * 80)
print("EPSILON SENSITIVITY TEST")
print("=" * 80)
print("Testing how epsilon affects coverage performance")
print("=" * 80)
print()

checkpoint_path = "./checkpoints/fcn_final.pt"

# Test different epsilon values
epsilons = [0.0, 0.05, 0.1, 0.2]

for eps in epsilons:
    print(f"\n{'='*80}")
    print(f"TESTING WITH EPSILON = {eps:.2f}")
    print(f"{'='*80}")
    
    coverages = []
    
    for ep in range(5):
        # Set seed for reproducibility
        random.seed(42 + ep)
        np.random.seed(42 + ep)
        torch.manual_seed(42 + ep)
        
        agent = FCNAgent(grid_size=20, input_channels=6)
        agent.load(checkpoint_path)
        agent.set_epsilon(eps)
        agent.policy_net.eval()
        
        dummy_occupancy = np.zeros((20, 20), dtype=np.float32)
        
        env = CoverageEnvironment(grid_size=20, map_type='empty')
        state = env.reset()
        
        max_steps = 150
        
        for step in range(max_steps):
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, env.world_state, agent_occupancy=dummy_occupancy, valid_actions=valid_actions)
            state, reward, done, info = env.step(action)
            
            if done:
                break
        
        final_coverage = info.get('coverage_pct', 0.0)
        coverages.append(final_coverage)
    
    avg = np.mean(coverages)
    std = np.std(coverages)
    
    print(f"\nResults (5 episodes):")
    for i, cov in enumerate(coverages):
        print(f"  Episode {i+1}: {cov:.1%}")
    print(f"\nAverage: {avg:.1%} ± {std:.1%}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nIf epsilon=0.0 gives best performance:")
print("  → Network has learned good policy, exploration hurts")
print("\nIf epsilon=0.1 gives best performance:")
print("  → Network needs exploration to escape local optima")
print("\nIf all epsilons give similar low performance:")
print("  → Network hasn't learned effective coverage behavior")
print("=" * 80)
