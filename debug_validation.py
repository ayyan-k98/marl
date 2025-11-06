"""
Debug script to test checkpoint using EXACT training validation code
"""
import numpy as np
from fcn_agent import FCNAgent
from environment import CoverageEnvironment
from config import config

# CRITICAL: Enable probabilistic mode (checkpoint was trained with --probabilistic)
config.USE_PROBABILISTIC_ENV = True

# Load checkpoint
checkpoint_path = "./checkpoints/fcn_final.pt"

agent = FCNAgent(grid_size=20, input_channels=6)
agent.load(checkpoint_path)

# Set epsilon to 0.1 like training validation
agent.set_epsilon(0.1)

# Create dummy occupancy
dummy_occupancy = np.zeros((20, 20), dtype=np.float32)

# Test on empty maps (8 episodes like validation)
coverages = []
for ep in range(8):
    env = CoverageEnvironment(grid_size=20, map_type='empty')
    state = env.reset()
    
    # Use VALIDATION_MAX_STEPS (150) like training validation
    max_steps = config.VALIDATION_MAX_STEPS if not config.FAST_VALIDATION else config.MAX_EPISODE_STEPS
    
    for step in range(max_steps):
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, env.world_state, agent_occupancy=dummy_occupancy, valid_actions=valid_actions)
        state, reward, done, info = env.step(action)
        
        if done:
            break
    
    final_coverage = info.get('coverage_pct', 0.0)
    coverages.append(final_coverage)
    print(f"Episode {ep+1}: {final_coverage:.1%}")

print(f"\nAverage: {np.mean(coverages):.1%}")
print(f"Expected from training: ~85.7%")
