"""
Test script to verify action masking implementation.
"""

import numpy as np
from environment import CoverageEnvironment
from multi_agent_env import MultiAgentCoverageEnv
import config

def test_single_agent_masking():
    """Test action masking in single-agent environment."""
    print("Testing single-agent action masking...")
    
    env = CoverageEnvironment(grid_size=40)
    env.reset()
    
    valid = env.get_valid_actions()
    
    print(f"  ✓ Valid actions shape: {valid.shape}")
    print(f"  ✓ Valid actions type: {type(valid)}")
    print(f"  ✓ Valid actions dtype: {valid.dtype}")
    print(f"  ✓ Number of valid actions: {np.sum(valid)}/{len(valid)}")
    
    # Test that at least some actions are valid
    assert np.sum(valid) > 0, "No valid actions found!"
    
    # Test that array is boolean
    assert valid.dtype == bool, "Valid actions should be boolean array!"
    
    print("  ✓ Single-agent masking works correctly!")
    return True

def test_multi_agent_masking():
    """Test action masking in multi-agent environment."""
    print("\nTesting multi-agent action masking...")
    
    env = MultiAgentCoverageEnv(num_agents=4, grid_size=40)
    observations = env.reset()
    
    for agent_id in range(4):
        valid = env.get_valid_actions(agent_id)
        
        print(f"  Agent {agent_id}:")
        print(f"    ✓ Valid actions shape: {valid.shape}")
        print(f"    ✓ Number of valid actions: {np.sum(valid)}/{len(valid)}")
        
        # Test that at least some actions are valid
        assert np.sum(valid) > 0, f"No valid actions for agent {agent_id}!"
        
        # Test that array is boolean
        assert valid.dtype == bool, f"Valid actions for agent {agent_id} should be boolean!"
    
    print("  ✓ Multi-agent masking works correctly!")
    return True

def test_masking_prevents_wall_collisions():
    """Test that masking correctly identifies wall collisions."""
    print("\nTesting wall collision detection...")
    
    env = CoverageEnvironment(grid_size=10)
    env.reset()
    
    # Move agent to corner (0, 0)
    env.robot_state.position = (0, 0)
    
    valid = env.get_valid_actions()
    
    print(f"  Position: (0, 0)")
    print(f"  Valid actions: {np.where(valid)[0].tolist()}")
    
    # At (0,0), actions that move into negative x or y should be invalid
    # ACTION_DELTAS: N(0,-1), NE(1,-1), E(1,0), SE(1,1), S(0,1), SW(-1,1), W(-1,0), NW(-1,-1), STAY(0,0)
    # From (0,0): N, NE, NW invalid (y<0); W, SW invalid (x<0)
    # Valid: E(2), SE(3), S(4), STAY(8)
    
    # Check that actions moving into negative coordinates are invalid
    assert not valid[0], "N: Moving to (0,-1) from (0,0) should be invalid!"
    assert not valid[1], "NE: Moving to (1,-1) from (0,0) should be invalid!"
    assert not valid[6], "W: Moving to (-1,0) from (0,0) should be invalid!"
    assert not valid[7], "NW: Moving to (-1,-1) from (0,0) should be invalid!"
    assert not valid[5], "SW: Moving to (-1,1) from (0,0) should be invalid!"
    
    # Check that valid movements work
    assert valid[2], "E: Moving to (1,0) from (0,0) should be valid!"
    assert valid[3], "SE: Moving to (1,1) from (0,0) should be valid!"
    assert valid[4], "S: Moving to (0,1) from (0,0) should be valid!"
    assert valid[8], "STAY: Staying at (0,0) should be valid!"
    
    print("  ✓ Wall collision detection works correctly!")
    return True

def test_fcn_agent_integration():
    """Test that FCN agent can use action masking."""
    print("\nTesting FCN agent integration...")
    
    from fcn_agent import FCNAgent
    
    env = CoverageEnvironment(grid_size=40)
    robot_state = env.reset()
    
    agent = FCNAgent(
        input_channels=6,
        grid_size=40,
        learning_rate=1e-4
    )
    
    # Test action selection with masking
    valid_actions = env.get_valid_actions()
    
    # Create dummy agent_occupancy (6th channel) - all zeros
    agent_occupancy = np.zeros((40, 40), dtype=np.float32)
    
    action = agent.select_action(
        robot_state,
        env.world_state,
        epsilon=0.0,  # Greedy
        agent_occupancy=agent_occupancy,
        valid_actions=valid_actions
    )
    
    print(f"  ✓ Selected action: {action}")
    print(f"  ✓ Action is valid: {valid_actions[action]}")
    
    # Verify selected action is valid
    assert valid_actions[action], "Agent selected invalid action!"
    
    print("  ✓ FCN agent integration works correctly!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("ACTION MASKING IMPLEMENTATION TEST")
    print("=" * 60)
    
    try:
        test_single_agent_masking()
        test_multi_agent_masking()
        test_masking_prevents_wall_collisions()
        test_fcn_agent_integration()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("Action masking is working correctly!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED!")
        print(f"Error: {e}")
        print("=" * 60)
        raise
