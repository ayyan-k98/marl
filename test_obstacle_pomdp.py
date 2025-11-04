"""
Test script to verify POMDP obstacle discovery.

Verifies that:
1. Obstacles are NOT visible globally (only within sensor range)
2. Obstacles are permanently remembered once discovered
3. Channel 4 only shows discovered obstacles
"""

import numpy as np
from environment import CoverageEnvironment
from fcn_agent import FCNAgent
from config import config


def test_obstacle_visibility():
    """Test that obstacles are only visible when sensed."""
    print("=" * 70)
    print("TEST 1: Obstacle Visibility (POMDP)")
    print("=" * 70)
    
    # Create environment with obstacles
    env = CoverageEnvironment(
        grid_size=20,
        sensor_range=4.0,
        map_type='room'  # Has walls/obstacles
    )
    
    # Reset environment
    state = env.reset()
    
    # Check initial obstacle knowledge
    initial_obstacles = len(state.discovered_obstacles)
    total_obstacles = len(env.world_state.obstacles)
    
    print(f"\n✓ Environment created with {total_obstacles} obstacles")
    print(f"✓ Agent initially knows about {initial_obstacles} obstacles")
    
    # Agent should only know about nearby obstacles (within sensor range)
    assert initial_obstacles < total_obstacles, \
        f"ERROR: Agent knows ALL obstacles ({initial_obstacles}/{total_obstacles})!"
    
    print(f"✓ PASS: Agent has partial knowledge ({initial_obstacles}/{total_obstacles})")
    
    return env, state


def test_obstacle_persistence():
    """Test that discovered obstacles remain known."""
    print("\n" + "=" * 70)
    print("TEST 2: Obstacle Memory Persistence")
    print("=" * 70)
    
    env, state = test_obstacle_visibility()
    
    # Remember initial obstacle count
    obstacles_at_start = len(state.discovered_obstacles)
    
    # Move around to discover more obstacles
    for step in range(20):
        action = 2  # Move East
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        if done:
            break
    
    obstacles_after_exploration = len(state.discovered_obstacles)
    
    print(f"\n✓ After 20 steps of exploration:")
    print(f"  Initial: {obstacles_at_start} obstacles known")
    print(f"  Final: {obstacles_after_exploration} obstacles known")
    
    # Should discover more obstacles
    assert obstacles_after_exploration >= obstacles_at_start, \
        "ERROR: Obstacle memory not persisting!"
    
    print(f"✓ PASS: Discovered {obstacles_after_exploration - obstacles_at_start} new obstacles")
    
    return env, state


def test_channel_4_encoding():
    """Test that Channel 4 only shows discovered obstacles."""
    print("\n" + "=" * 70)
    print("TEST 3: Channel 4 Encoding (FCN Agent)")
    print("=" * 70)
    
    # Create agent
    agent = FCNAgent(grid_size=20, input_channels=6)
    
    # Create environment
    env = CoverageEnvironment(
        grid_size=20,
        sensor_range=4.0,
        map_type='room'
    )
    state = env.reset()
    
    # Encode state
    grid_tensor = agent._encode_state(state, env.world_state)
    
    # Extract Channel 4 (obstacles)
    channel_4 = grid_tensor[0, 4].cpu().numpy()  # [H, W]
    
    # Count obstacles in channel 4
    obstacles_in_channel = int(channel_4.sum())
    obstacles_discovered = len(state.discovered_obstacles)
    total_obstacles = len(env.world_state.obstacles)
    
    print(f"\n✓ Channel 4 analysis:")
    print(f"  Obstacles in Channel 4: {obstacles_in_channel}")
    print(f"  Discovered obstacles: {obstacles_discovered}")
    print(f"  Total obstacles (ground truth): {total_obstacles}")
    
    # Channel 4 should match discovered obstacles
    assert obstacles_in_channel == obstacles_discovered, \
        f"ERROR: Channel 4 mismatch! {obstacles_in_channel} != {obstacles_discovered}"
    
    # Channel 4 should NOT have all obstacles
    assert obstacles_in_channel < total_obstacles, \
        f"ERROR: Channel 4 has ALL obstacles! {obstacles_in_channel} == {total_obstacles}"
    
    print(f"✓ PASS: Channel 4 shows only discovered obstacles ({obstacles_in_channel}/{total_obstacles})")
    
    # Test persistence after movement
    print(f"\n✓ Testing obstacle memory after movement...")
    
    for step in range(10):
        action = 1  # Move NE
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        if done:
            break
    
    # Re-encode state
    grid_tensor = agent._encode_state(state, env.world_state)
    channel_4 = grid_tensor[0, 4].cpu().numpy()
    
    obstacles_in_channel_after = int(channel_4.sum())
    obstacles_discovered_after = len(state.discovered_obstacles)
    
    print(f"  After 10 steps:")
    print(f"    Obstacles in Channel 4: {obstacles_in_channel_after}")
    print(f"    Discovered obstacles: {obstacles_discovered_after}")
    
    assert obstacles_in_channel_after == obstacles_discovered_after, \
        "ERROR: Channel 4 doesn't match discovered obstacles!"
    
    assert obstacles_in_channel_after >= obstacles_in_channel, \
        "ERROR: Obstacle memory lost!"
    
    print(f"✓ PASS: Obstacle memory persists correctly")


def test_exploration_discovers_obstacles():
    """Test that exploration gradually reveals obstacles."""
    print("\n" + "=" * 70)
    print("TEST 4: Obstacle Discovery Through Exploration")
    print("=" * 70)
    
    # Create environment with complex obstacles
    env = CoverageEnvironment(
        grid_size=20,
        sensor_range=4.0,
        map_type='corridor'
    )
    state = env.reset()
    
    # Track obstacle discovery over time
    discovery_timeline = []
    
    for step in range(100):
        # Random exploration
        action = np.random.randint(0, 9)
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        obstacles_known = len(state.discovered_obstacles)
        discovery_timeline.append(obstacles_known)
        
        if step % 20 == 0:
            percentage = (obstacles_known / len(env.world_state.obstacles)) * 100
            print(f"  Step {step:3d}: {obstacles_known:3d} obstacles known ({percentage:5.1f}%)")
        
        if done:
            break
    
    # Verify discovery increases over time
    initial_knowledge = discovery_timeline[0]
    final_knowledge = discovery_timeline[-1]
    
    print(f"\n✓ Obstacle discovery summary:")
    print(f"  Initial: {initial_knowledge} obstacles")
    print(f"  Final: {final_knowledge} obstacles")
    print(f"  Discovered: {final_knowledge - initial_knowledge} new obstacles")
    
    assert final_knowledge > initial_knowledge, \
        "ERROR: No obstacle discovery during exploration!"
    
    print(f"✓ PASS: Exploration successfully discovers new obstacles")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TESTING POMDP OBSTACLE DISCOVERY")
    print("=" * 70)
    print("\nVerifying that obstacles are NOT globally visible...")
    print("Agents must explore to discover obstacles within sensor range.")
    print("=" * 70)
    
    try:
        # Run tests
        test_obstacle_visibility()
        test_obstacle_persistence()
        test_channel_4_encoding()
        test_exploration_discovers_obstacles()
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n✓ Obstacles are POMDP (partially observable)")
        print("✓ Obstacles discovered through ray-casting")
        print("✓ Obstacle memory persists correctly")
        print("✓ Channel 4 shows only discovered obstacles")
        print("✓ Exploration reveals more obstacles over time")
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
        raise


if __name__ == "__main__":
    main()
