"""
Test with probabilistic mode enabled vs disabled
Maybe the checkpoint was trained with probabilistic mode?
"""
import numpy as np
import torch
import random
from fcn_agent import FCNAgent
from environment import CoverageEnvironment
from config import config

print("=" * 80)
print("PROBABILISTIC MODE TEST")
print("=" * 80)
print()

checkpoint_path = "./checkpoints/fcn_final.pt"

for use_prob in [False, True]:
    # Set probabilistic mode
    config.USE_PROBABILISTIC_ENV = use_prob
    
    mode_name = "PROBABILISTIC" if use_prob else "BINARY"
    print(f"\n{'='*80}")
    print(f"TESTING WITH {mode_name} COVERAGE MODE")
    print(f"{'='*80}")
    
    coverages = []
    
    for ep in range(5):
        # Set seed
        random.seed(42 + ep)
        np.random.seed(42 + ep)
        torch.manual_seed(42 + ep)
        
        agent = FCNAgent(grid_size=20, input_channels=6)
        agent.load(checkpoint_path)
        agent.set_epsilon(0.1)
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
    
    print(f"\nResults (5 episodes, 150 steps):")
    for i, cov in enumerate(coverages):
        print(f"  Episode {i+1}: {cov:.1%}")
    print(f"\nAverage: {avg:.1%} ± {std:.1%}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("\nIf PROBABILISTIC mode gives much better results (~70%+):")
print("  → Checkpoint was trained with --probabilistic flag")
print("  → Training validation used probabilistic mode") 
print("  → You MUST use probabilistic mode for testing")
print("\nIf BINARY mode matches PROBABILISTIC:")
print("  → Mode doesn't matter much")
print("\nIf both give low performance (~20-30%):")
print("  → Checkpoint performance is genuinely low")
print("  → Training logs may have been incorrect")
print("=" * 80)
