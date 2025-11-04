"""Quick test of single-agent training with action masking."""

from environment import CoverageEnvironment
from fcn_agent import FCNAgent
import numpy as np

print("Testing single-agent with action masking...")

# Create environment and agent
env = CoverageEnvironment(grid_size=40)
state = env.reset()
agent = FCNAgent(input_channels=6, grid_size=40, learning_rate=1e-4)

# Create dummy 6th channel
dummy_occ = np.zeros((40, 40), dtype=np.float32)

# Test action selection with masking
valid = env.get_valid_actions()
grid = agent._encode_state(state, env.world_state, dummy_occ)
action = agent.select_action_from_tensor(grid, valid_actions=valid)

print(f"✓ Selected action: {action}")
print(f"✓ Action is valid: {valid[action]}")
assert valid[action], "Agent selected invalid action!"

# Test environment step
state, reward, done, info = env.step(action)
collision = info.get('collision', False)

print(f"✓ Step completed: reward={reward:.3f}, collision={collision}")
assert not collision, "Collision occurred with action masking!"

print("\n✓ Single-agent with action masking works correctly!")
