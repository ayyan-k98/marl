"""
Test with FULL episode length (350 steps) instead of validation steps (150)
Maybe training logs were from training episodes, not validation?
"""
import numpy as np
from fcn_agent import FCNAgent
from environment import CoverageEnvironment
from config import config

# Load checkpoint
checkpoint_path = "./checkpoints/fcn_final.pt"

agent = FCNAgent(grid_size=20, input_channels=6)
agent.load(checkpoint_path)

# Set epsilon to 0.1 like training validation
agent.set_epsilon(0.1)

# Create dummy occupancy
dummy_occupancy = np.zeros((20, 20), dtype=np.float32)

print("Testing with FULL episode length (350 steps)...")
print("=" * 80)

# Test on empty maps (8 episodes)
coverages = []
for ep in range(8):
    env = CoverageEnvironment(grid_size=20, map_type='empty')
    state = env.reset()
    
    # Use FULL training episode length (350 steps)
    max_steps = config.MAX_EPISODE_STEPS
    
    for step in range(max_steps):
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, env.world_state, agent_occupancy=dummy_occupancy, valid_actions=valid_actions)
        state, reward, done, info = env.step(action)
        
        if done:
            break
    
    final_coverage = info.get('coverage_pct', 0.0)
    coverages.append(final_coverage)
    print(f"Episode {ep+1}: {final_coverage:.1%} (steps: {step+1}/{max_steps})")

print("=" * 80)
print(f"\nAverage with 350 steps: {np.mean(coverages):.1%}")
print(f"Average with 150 steps: ~30.6% (from previous test)")
print(f"Training claimed: ~85.7%")
print(f"\nDifference: {np.mean(coverages)*100 - 85.7:.1f} percentage points")
