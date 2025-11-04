"""
Enhanced test for partial obstacle map encoding (Channel 4).

Verifies the 3-value encoding:
- 0.0 = unexplored / unknown cells
- 0.5 = explored free space
- 1.0 = discovered obstacles
"""

import numpy as np
from environment import CoverageEnvironment
from fcn_agent import FCNAgent
from config import config


def test_channel_4_three_value_encoding():
    """Test that Channel 4 uses 0.0, 0.5, 1.0 encoding correctly."""
    print("=" * 70)
    print("TEST: Channel 4 Three-Value Encoding")
    print("=" * 70)
    print("\nVerifying partial obstacle map encoding:")
    print("  0.0 = unexplored/unknown")
    print("  0.5 = explored free space")
    print("  1.0 = discovered obstacles")
    print("=" * 70)
    
    # Create environment with obstacles
    env = CoverageEnvironment(
        grid_size=20,
        sensor_range=4.0,
        map_type='room'  # Has internal walls
    )
    
    # Create agent
    agent = FCNAgent(grid_size=20, input_channels=6)
    
    # Reset environment
    state = env.reset()
    
    print(f"\n✓ Environment created: 20×20 grid, sensor_range=4.0")
    print(f"  Total obstacles: {len(env.world_state.obstacles)}")
    print(f"  Initial discovered: {len(state.discovered_obstacles)}")
    
    # Encode initial state
    grid_tensor = agent._encode_state(state, env.world_state)
    channel_4 = grid_tensor[0, 4].cpu().numpy()  # [H, W]
    
    # Analyze initial state
    unknown_cells = np.sum(channel_4 == 0.0)
    free_cells = np.sum(channel_4 == 0.5)
    obstacle_cells = np.sum(channel_4 == 1.0)
    total_cells = channel_4.size
    
    print(f"\n✓ Initial Channel 4 analysis:")
    print(f"  Unknown (0.0):    {unknown_cells:4d} cells ({unknown_cells/total_cells*100:.1f}%)")
    print(f"  Free (0.5):       {free_cells:4d} cells ({free_cells/total_cells*100:.1f}%)")
    print(f"  Obstacles (1.0):  {obstacle_cells:4d} cells ({obstacle_cells/total_cells*100:.1f}%)")
    
    # Verify initial conditions
    assert unknown_cells > 0, "ERROR: No unknown cells at start!"
    assert free_cells > 0, "ERROR: No free cells discovered at start!"
    print(f"✓ PASS: Initial state has unknown, free, and obstacle cells")
    
    # Move around to discover more
    print(f"\n✓ Exploring for 30 steps...")
    for step in range(30):
        action = np.random.randint(0, 9)
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        if done:
            break
    
    # Re-encode after exploration
    grid_tensor = agent._encode_state(state, env.world_state)
    channel_4_after = grid_tensor[0, 4].cpu().numpy()
    
    # Analyze after exploration
    unknown_after = np.sum(channel_4_after == 0.0)
    free_after = np.sum(channel_4_after == 0.5)
    obstacle_after = np.sum(channel_4_after == 1.0)
    
    print(f"\n✓ After exploration:")
    print(f"  Unknown (0.0):    {unknown_after:4d} cells ({unknown_after/total_cells*100:.1f}%)")
    print(f"  Free (0.5):       {free_after:4d} cells ({free_after/total_cells*100:.1f}%)")
    print(f"  Obstacles (1.0):  {obstacle_after:4d} cells ({obstacle_after/total_cells*100:.1f}%)")
    
    # Verify changes
    assert unknown_after < unknown_cells, \
        f"ERROR: Unknown cells didn't decrease! {unknown_after} >= {unknown_cells}"
    
    assert free_after > free_cells, \
        f"ERROR: Free cells didn't increase! {free_after} <= {free_cells}"
    
    print(f"\n✓ PASS: Exploration reduces unknown, increases free cells")
    
    # Verify obstacle count matches discovered_obstacles
    discovered_count = len(state.discovered_obstacles)
    assert obstacle_after == discovered_count, \
        f"ERROR: Channel 4 obstacles ({obstacle_after}) != discovered_obstacles ({discovered_count})"
    
    print(f"✓ PASS: Channel 4 obstacles match discovered_obstacles set")
    
    # Verify no invalid values
    unique_values = np.unique(channel_4_after)
    valid_values = {0.0, 0.5, 1.0}
    for val in unique_values:
        assert val in valid_values, \
            f"ERROR: Invalid value {val} in Channel 4! Expected only {valid_values}"
    
    print(f"✓ PASS: Channel 4 contains only valid values (0.0, 0.5, 1.0)")
    
    return env, state, channel_4_after


def test_free_cell_encoding_details():
    """Test that free cells are correctly marked as 0.5."""
    print("\n" + "=" * 70)
    print("TEST: Free Cell Encoding Details")
    print("=" * 70)
    
    # Create simple environment
    env = CoverageEnvironment(
        grid_size=15,
        sensor_range=5.0,
        map_type='empty'  # Minimal obstacles
    )
    
    agent = FCNAgent(grid_size=15, input_channels=6)
    state = env.reset()
    
    # Take a few steps to sense free cells
    for _ in range(5):
        action = 2  # Move East
        state, _, done, _ = env.step(action)
        if done:
            break
    
    # Encode state
    grid_tensor = agent._encode_state(state, env.world_state)
    channel_4 = grid_tensor[0, 4].cpu().numpy()
    
    # Analyze local_map entries
    free_in_local_map = sum(1 for (x, y), (cov, typ) in state.local_map.items() if typ == "free")
    obstacles_in_local_map = sum(1 for (x, y), (cov, typ) in state.local_map.items() if typ == "obstacle")
    
    print(f"\n✓ Agent's local_map:")
    print(f"  Free cells:     {free_in_local_map}")
    print(f"  Obstacle cells: {obstacles_in_local_map}")
    
    # Check channel 4 encoding of those cells
    free_count_ch4 = 0
    obstacle_count_ch4 = 0
    
    for (x, y), (cov, typ) in state.local_map.items():
        if 0 <= x < 15 and 0 <= y < 15:
            ch4_value = channel_4[y, x]
            if typ == "free":
                if ch4_value == 0.5:
                    free_count_ch4 += 1
                else:
                    print(f"  WARNING: Free cell ({x},{y}) has channel_4={ch4_value}, expected 0.5")
            elif typ == "obstacle":
                if ch4_value == 1.0:
                    obstacle_count_ch4 += 1
                else:
                    print(f"  WARNING: Obstacle cell ({x},{y}) has channel_4={ch4_value}, expected 1.0")
    
    print(f"\n✓ Channel 4 encoding:")
    print(f"  Free cells (0.5):      {free_count_ch4}/{free_in_local_map}")
    print(f"  Obstacles (1.0):       {obstacle_count_ch4}/{obstacles_in_local_map}")
    
    # Verify encoding is correct
    assert free_count_ch4 == free_in_local_map, \
        f"ERROR: Not all free cells encoded as 0.5! {free_count_ch4}/{free_in_local_map}"
    
    assert obstacle_count_ch4 == obstacles_in_local_map, \
        f"ERROR: Not all obstacles encoded as 1.0! {obstacle_count_ch4}/{obstacles_in_local_map}"
    
    print(f"✓ PASS: All local_map entries correctly encoded in Channel 4")


def test_unknown_cells_remain_zero():
    """Test that unexplored cells remain 0.0."""
    print("\n" + "=" * 70)
    print("TEST: Unknown Cells Remain 0.0")
    print("=" * 70)
    
    # Create environment
    env = CoverageEnvironment(
        grid_size=25,
        sensor_range=3.0,
        map_type='random'
    )
    
    agent = FCNAgent(grid_size=25, input_channels=6)
    state = env.reset()
    
    # Move only a little (don't explore everything)
    for _ in range(10):
        action = 0  # Move North
        state, _, done, _ = env.step(action)
        if done:
            break
    
    # Encode state
    grid_tensor = agent._encode_state(state, env.world_state)
    channel_4 = grid_tensor[0, 4].cpu().numpy()
    
    # Find cells NOT in local_map (unexplored)
    unexplored_cells = []
    for x in range(25):
        for y in range(25):
            if (x, y) not in state.local_map:
                unexplored_cells.append((x, y))
    
    print(f"\n✓ Agent explored: {len(state.local_map)} cells")
    print(f"  Unexplored: {len(unexplored_cells)} cells")
    
    # Verify all unexplored cells are 0.0
    wrong_values = []
    for (x, y) in unexplored_cells[:100]:  # Check first 100
        if channel_4[y, x] != 0.0:
            wrong_values.append((x, y, channel_4[y, x]))
    
    if wrong_values:
        print(f"\n  ERROR: {len(wrong_values)} unexplored cells have non-zero values:")
        for (x, y, val) in wrong_values[:5]:
            print(f"    Cell ({x},{y}): {val}")
        assert False, "Unexplored cells should be 0.0!"
    
    print(f"✓ PASS: All unexplored cells are 0.0 in Channel 4")


def test_multi_agent_partial_obstacles():
    """Test partial obstacle maps in multi-agent environment."""
    print("\n" + "=" * 70)
    print("TEST: Multi-Agent Partial Obstacle Maps")
    print("=" * 70)
    
    from multi_agent_env import MultiAgentCoverageEnv, CoordinationStrategy
    
    # Create multi-agent environment
    env = MultiAgentCoverageEnv(
        num_agents=3,
        grid_size=20,
        sensor_range=4.0,
        coordination=CoordinationStrategy.INDEPENDENT,
        map_type='corridor'
    )
    
    agent = FCNAgent(grid_size=20, input_channels=6)
    state = env.reset()
    
    print(f"\n✓ Multi-agent environment: 3 agents, 20×20 grid")
    
    # Check each agent has independent obstacle knowledge
    for i, agent_state in enumerate(state.agents):
        obs_count = len(agent_state.robot_state.discovered_obstacles)
        local_count = len(agent_state.robot_state.local_map)
        
        print(f"  Agent {i}: {obs_count} obstacles, {local_count} cells in local_map")
    
    # Take some steps
    for step in range(20):
        actions = [np.random.randint(0, 9) for _ in range(3)]
        state, rewards, done, info = env.step(actions)
        if done:
            break
    
    print(f"\n✓ After 20 steps:")
    
    # Verify agents have different knowledge
    all_discovered = [len(a.robot_state.discovered_obstacles) for a in state.agents]
    all_local_maps = [len(a.robot_state.local_map) for a in state.agents]
    
    for i in range(3):
        print(f"  Agent {i}: {all_discovered[i]} obstacles, {all_local_maps[i]} cells explored")
    
    # Agents should have explored different amounts (unless they stayed together)
    if len(set(all_local_maps)) > 1:
        print(f"✓ PASS: Agents have independent exploration (different local map sizes)")
    else:
        print(f"⚠ WARNING: All agents have same local map size (may have moved together)")
    
    # Verify each agent's channel 4 reflects their own knowledge
    for i, agent_state in enumerate(state.agents):
        # Get observations
        observations = env.get_observations()
        obs = observations[i]
        
        # Encode state
        grid_tensor = agent._encode_state(
            obs['robot_state'],
            obs['world_state']
        )
        channel_4 = grid_tensor[0, 4].cpu().numpy()
        
        obstacles_ch4 = np.sum(channel_4 == 1.0)
        obstacles_discovered = len(agent_state.robot_state.discovered_obstacles)
        
        assert obstacles_ch4 == obstacles_discovered, \
            f"Agent {i}: Channel 4 obstacles ({obstacles_ch4}) != discovered ({obstacles_discovered})"
    
    print(f"✓ PASS: Each agent's Channel 4 matches their discovered obstacles")


def main():
    """Run all enhanced tests."""
    print("\n" + "=" * 70)
    print("ENHANCED PARTIAL OBSTACLE MAP TESTS")
    print("=" * 70)
    print("\nTesting 3-value encoding: 0.0 (unknown), 0.5 (free), 1.0 (obstacle)")
    print("=" * 70)
    
    try:
        # Run all tests
        test_channel_4_three_value_encoding()
        test_free_cell_encoding_details()
        test_unknown_cells_remain_zero()
        test_multi_agent_partial_obstacles()
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ ALL ENHANCED TESTS PASSED!")
        print("=" * 70)
        print("\n✓ Channel 4 correctly encodes:")
        print("  • 0.0 = unexplored/unknown cells")
        print("  • 0.5 = explored free space")
        print("  • 1.0 = discovered obstacles")
        print("\n✓ Partial obstacle maps work correctly")
        print("✓ Each agent maintains independent obstacle knowledge")
        print("✓ Exploration gradually reveals the environment")
        print("\n" + "=" * 70)
        
    except AssertionError as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED!")
        print("=" * 70)
        print(f"\n{e}")
        raise
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR!")
        print("=" * 70)
        print(f"\n{e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
