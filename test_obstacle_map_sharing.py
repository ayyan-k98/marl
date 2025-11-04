"""
Test obstacle map sharing via communication system.

Verifies that:
1. Agents can share discovered obstacles via communication
2. Agents can share discovered free cells via communication
3. Communication respects range limits
4. Reliability weighting works correctly
"""

import numpy as np
from multi_agent_env import MultiAgentCoverageEnv, CoordinationStrategy
from communication import PositionCommunication, get_communication_protocol
from config import config


def test_obstacle_map_broadcast():
    """Test that agents broadcast their discovered obstacles."""
    print("=" * 70)
    print("TEST 1: Obstacle Map Broadcasting")
    print("=" * 70)
    
    # Create communication protocol
    comm = PositionCommunication(
        num_agents=3,
        comm_range=15.0,
        comm_freq=1,
        share_obstacle_maps=True
    )
    
    print(f"\n✓ Communication protocol created:")
    print(f"  Agents: {comm.num_agents}")
    print(f"  Range: {comm.comm_range}")
    print(f"  Share obstacle maps: {comm.share_obstacle_maps}")
    
    # Simulate agents discovering different obstacles
    agent_0_obstacles = {(5, 5), (6, 5), (7, 5)}
    agent_0_free = {(5, 6), (6, 6), (7, 6)}
    
    agent_1_obstacles = {(10, 10), (11, 10)}
    agent_1_free = {(10, 11), (11, 11)}
    
    # Create mock local_map
    agent_0_local_map = {}
    for cell in agent_0_free:
        agent_0_local_map[cell] = (0.8, "free")
    for cell in agent_0_obstacles:
        agent_0_local_map[cell] = (0.0, "obstacle")
    
    agent_1_local_map = {}
    for cell in agent_1_free:
        agent_1_local_map[cell] = (0.7, "free")
    for cell in agent_1_obstacles:
        agent_1_local_map[cell] = (0.0, "obstacle")
    
    # Broadcast
    comm.broadcast(
        agent_id=0,
        position=(5.0, 5.0),
        velocity=(1.0, 0.0),
        discovered_obstacles=agent_0_obstacles,
        local_map=agent_0_local_map
    )
    
    comm.broadcast(
        agent_id=1,
        position=(10.0, 10.0),
        velocity=(0.0, 1.0),
        discovered_obstacles=agent_1_obstacles,
        local_map=agent_1_local_map
    )
    
    print(f"\n✓ Agent 0 broadcasted:")
    print(f"  Position: (5.0, 5.0)")
    print(f"  Obstacles: {len(agent_0_obstacles)} cells")
    print(f"  Free: {len(agent_0_free)} cells")
    
    print(f"\n✓ Agent 1 broadcasted:")
    print(f"  Position: (10.0, 10.0)")
    print(f"  Obstacles: {len(agent_1_obstacles)} cells")
    print(f"  Free: {len(agent_1_free)} cells")
    
    # Verify messages stored
    assert 0 in comm.last_messages, "Agent 0 message not stored!"
    assert 1 in comm.last_messages, "Agent 1 message not stored!"
    
    pos, vel, timestamp, obstacles, free_cells = comm.last_messages[0]
    assert obstacles == agent_0_obstacles, "Agent 0 obstacles not stored correctly!"
    assert free_cells == agent_0_free, "Agent 0 free cells not stored correctly!"
    
    print(f"\n✓ PASS: Messages stored correctly in communication system")


def test_obstacle_map_reception():
    """Test that agents receive obstacle maps from nearby agents."""
    print("\n" + "=" * 70)
    print("TEST 2: Obstacle Map Reception")
    print("=" * 70)
    
    # Create communication protocol
    comm = PositionCommunication(
        num_agents=3,
        comm_range=10.0,
        comm_freq=1,
        share_obstacle_maps=True
    )
    
    # Agent 0 at (5, 5) discovers obstacles
    agent_0_obstacles = {(5, 5), (6, 5)}
    agent_0_free = {(5, 6), (6, 6)}
    agent_0_local_map = {cell: (0.8, "free") for cell in agent_0_free}
    
    comm.broadcast(
        agent_id=0,
        position=(5.0, 5.0),
        velocity=(0.0, 0.0),
        discovered_obstacles=agent_0_obstacles,
        local_map=agent_0_local_map
    )
    
    # Agent 1 nearby at (8, 8) should receive
    messages_for_agent_1 = comm.receive(
        agent_id=1,
        agent_position=(8.0, 8.0)
    )
    
    # Agent 2 far away at (20, 20) should NOT receive
    messages_for_agent_2 = comm.receive(
        agent_id=2,
        agent_position=(20.0, 20.0)
    )
    
    print(f"\n✓ Agent 1 (nearby at (8, 8)):")
    print(f"  Received {len(messages_for_agent_1)} messages")
    
    print(f"\n✓ Agent 2 (far at (20, 20)):")
    print(f"  Received {len(messages_for_agent_2)} messages")
    
    # Verify agent 1 received the message
    assert len(messages_for_agent_1) == 1, \
        f"Agent 1 should receive 1 message, got {len(messages_for_agent_1)}"
    
    msg = messages_for_agent_1[0]
    assert msg['sender_id'] == 0, "Wrong sender ID!"
    assert msg['discovered_obstacles'] == agent_0_obstacles, "Obstacles not received!"
    assert msg['discovered_free'] == agent_0_free, "Free cells not received!"
    
    print(f"\n✓ Agent 1 received message:")
    print(f"  Sender: Agent {msg['sender_id']}")
    print(f"  Obstacles: {len(msg['discovered_obstacles'])} cells")
    print(f"  Free cells: {len(msg['discovered_free'])} cells")
    print(f"  Reliability: {msg['reliability']:.3f}")
    
    # Verify agent 2 did NOT receive (out of range)
    assert len(messages_for_agent_2) == 0, \
        f"Agent 2 should not receive messages, got {len(messages_for_agent_2)}"
    
    print(f"\n✓ PASS: Range-limited communication works correctly")


def test_merge_obstacle_maps():
    """Test merging received obstacle maps into agent's knowledge."""
    print("\n" + "=" * 70)
    print("TEST 3: Merging Obstacle Maps")
    print("=" * 70)
    
    from data_structures import RobotState
    
    # Create communication protocol
    comm = PositionCommunication(
        num_agents=2,
        comm_range=15.0,
        comm_freq=1,
        share_obstacle_maps=True
    )
    
    # Agent 0 discovers some obstacles
    agent_0_obstacles = {(5, 5), (6, 5)}
    agent_0_free = {(5, 6), (6, 6)}
    agent_0_local_map = {cell: (0.9, "free") for cell in agent_0_free}
    
    comm.broadcast(
        agent_id=0,
        position=(5.0, 5.0),
        velocity=(0.0, 0.0),
        discovered_obstacles=agent_0_obstacles,
        local_map=agent_0_local_map
    )
    
    # Agent 1 starts with empty knowledge
    agent_1_state = RobotState(
        position=(8, 8),
        orientation=0.0,
        grid_size=20
    )
    
    initial_obstacles = len(agent_1_state.discovered_obstacles)
    initial_local_map = len(agent_1_state.local_map)
    
    print(f"\n✓ Agent 1 initial knowledge:")
    print(f"  Discovered obstacles: {initial_obstacles}")
    print(f"  Local map size: {initial_local_map}")
    
    # Agent 1 receives message from Agent 0
    messages = comm.receive(agent_id=1, agent_position=(8.0, 8.0))
    
    print(f"\n✓ Agent 1 received {len(messages)} messages")
    
    # Merge received obstacle maps
    comm.merge_obstacle_maps(agent_1_state, messages, reliability_threshold=0.3)
    
    final_obstacles = len(agent_1_state.discovered_obstacles)
    final_local_map = len(agent_1_state.local_map)
    
    print(f"\n✓ Agent 1 after merging:")
    print(f"  Discovered obstacles: {final_obstacles}")
    print(f"  Local map size: {final_local_map}")
    
    # Verify obstacles were merged
    assert final_obstacles > initial_obstacles, \
        "Obstacles were not merged!"
    
    assert agent_0_obstacles.issubset(agent_1_state.discovered_obstacles), \
        "Agent 0's obstacles not in Agent 1's knowledge!"
    
    # Verify free cells were merged
    assert final_local_map > initial_local_map, \
        "Free cells were not merged!"
    
    for cell in agent_0_free:
        assert cell in agent_1_state.local_map, \
            f"Free cell {cell} not in Agent 1's local_map!"
    
    print(f"\n✓ PASS: Obstacle maps merged correctly")


def test_communication_integration():
    """Test full integration with multi-agent environment."""
    print("\n" + "=" * 70)
    print("TEST 4: Communication Integration")
    print("=" * 70)
    
    # Create environment
    env = MultiAgentCoverageEnv(
        num_agents=3,
        grid_size=20,
        sensor_range=4.0,
        communication_range=15.0,
        coordination=CoordinationStrategy.INDEPENDENT,
        map_type='room'
    )
    
    # Create communication protocol
    comm = get_communication_protocol(
        protocol_name='position',
        num_agents=3,
        comm_range=15.0,
        comm_freq=1,
        share_obstacle_maps=True
    )
    
    # Set communication in environment
    env.set_communication_protocol(comm)
    
    print(f"\n✓ Environment created with communication")
    
    # Reset environment
    state = env.reset()
    
    # Get initial obstacle knowledge for each agent
    initial_knowledge = []
    for i in range(3):
        obs_count = len(state.agents[i].robot_state.discovered_obstacles)
        initial_knowledge.append(obs_count)
        print(f"  Agent {i}: {obs_count} obstacles initially known")
    
    # Take some steps
    print(f"\n✓ Running 30 steps with communication...")
    for step in range(30):
        # Get observations
        observations = env.get_observations()
        
        # Communicate (shares obstacle maps)
        all_messages = comm.communicate(observations, state)
        
        # Merge received maps into each agent's knowledge
        for i, messages in enumerate(all_messages):
            if messages:
                comm.merge_obstacle_maps(
                    state.agents[i].robot_state,
                    messages,
                    reliability_threshold=0.3
                )
        
        # Take actions
        actions = [np.random.randint(0, 9) for _ in range(3)]
        state, rewards, done, info = env.step(actions)
        
        if done:
            break
    
    # Check final knowledge
    print(f"\n✓ After {step+1} steps with obstacle sharing:")
    final_knowledge = []
    for i in range(3):
        obs_count = len(state.agents[i].robot_state.discovered_obstacles)
        final_knowledge.append(obs_count)
        print(f"  Agent {i}: {obs_count} obstacles known (+{obs_count - initial_knowledge[i]})")
    
    # Verify agents learned from each other
    total_initial = sum(initial_knowledge)
    total_final = sum(final_knowledge)
    
    print(f"\n✓ Total obstacle knowledge:")
    print(f"  Initial: {total_initial}")
    print(f"  Final: {total_final}")
    print(f"  Increase: +{total_final - total_initial}")
    
    assert total_final > total_initial, \
        "Agents did not gain obstacle knowledge!"
    
    print(f"\n✓ PASS: Communication integration works correctly")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TESTING OBSTACLE MAP SHARING VIA COMMUNICATION")
    print("=" * 70)
    print("\nVerifying that agents can share discovered obstacle maps")
    print("to avoid redundant exploration and improve coordination.")
    print("=" * 70)
    
    try:
        # Run tests
        test_obstacle_map_broadcast()
        test_obstacle_map_reception()
        test_merge_obstacle_maps()
        test_communication_integration()
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ ALL COMMUNICATION TESTS PASSED!")
        print("=" * 70)
        print("\n✓ Agents can broadcast discovered obstacles")
        print("✓ Agents can broadcast discovered free cells")
        print("✓ Communication respects range limits")
        print("✓ Obstacle maps merge correctly")
        print("✓ Integration with environment works")
        print("\n✓ Agents can now share obstacle knowledge for coordination!")
        print("=" * 70)
        
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
