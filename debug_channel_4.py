"""
Debug script to inspect Channel 4 encoding issues.

Runs a few steps and prints detailed information about:
- discovered_obstacles
- local_map contents
- Channel 4 values
- Inconsistencies between data structures
"""

import numpy as np
from environment import CoverageEnvironment
from fcn_agent import FCNAgent
from config import config

def debug_channel_4():
    print("="*70)
    print("DEBUG: Channel 4 Encoding Issues")
    print("="*70)
    
    # Create environment and agent
    env = CoverageEnvironment(grid_size=20, sensor_range=4.0, map_type="random")
    agent = FCNAgent(grid_size=20, input_channels=6)
    
    # Reset environment
    state = env.reset()
    
    print(f"\n{'='*70}")
    print(f"STEP 0 (Initial State)")
    print(f"{'='*70}")
    print(f"Agent position: {state.position}")
    print(f"Discovered obstacles: {len(state.discovered_obstacles)}")
    print(f"Local map size: {len(state.local_map)}")
    
    # Count cell types in local_map
    free_count = sum(1 for (pos, (cov, ctype)) in state.local_map.items() if ctype == "free")
    obstacle_count = sum(1 for (pos, (cov, ctype)) in state.local_map.items() if ctype == "obstacle")
    
    print(f"  - Free cells in local_map: {free_count}")
    print(f"  - Obstacle cells in local_map: {obstacle_count}")
    
    # Check for inconsistencies
    print(f"\nChecking for inconsistencies...")
    
    # 1. Are all discovered_obstacles also in local_map with type="obstacle"?
    missing_from_local_map = []
    wrong_type_in_local_map = []
    
    for obs_pos in state.discovered_obstacles:
        if obs_pos not in state.local_map:
            missing_from_local_map.append(obs_pos)
        elif state.local_map[obs_pos][1] != "obstacle":
            wrong_type_in_local_map.append((obs_pos, state.local_map[obs_pos]))
    
    if missing_from_local_map:
        print(f"  ❌ {len(missing_from_local_map)} obstacles in discovered_obstacles but NOT in local_map!")
        print(f"     Examples: {missing_from_local_map[:3]}")
    else:
        print(f"  ✓ All discovered_obstacles are in local_map")
    
    if wrong_type_in_local_map:
        print(f"  ❌ {len(wrong_type_in_local_map)} obstacles marked as 'free' in local_map!")
        for pos, (cov, ctype) in wrong_type_in_local_map[:3]:
            print(f"     {pos}: coverage={cov}, type='{ctype}' (should be 'obstacle')")
    else:
        print(f"  ✓ All discovered_obstacles have correct type in local_map")
    
    # 2. Are there "obstacle" entries in local_map NOT in discovered_obstacles?
    extra_obstacles = []
    for pos, (cov, ctype) in state.local_map.items():
        if ctype == "obstacle" and pos not in state.discovered_obstacles:
            extra_obstacles.append(pos)
    
    if extra_obstacles:
        print(f"  ❌ {len(extra_obstacles)} obstacles in local_map but NOT in discovered_obstacles!")
        print(f"     Examples: {extra_obstacles[:3]}")
    else:
        print(f"  ✓ discovered_obstacles and local_map['obstacle'] are consistent")
    
    # Encode state and check Channel 4
    grid_tensor = agent._encode_state(state, env.world_state)
    channel_4 = grid_tensor[0, 4].cpu().numpy()  # [H, W]
    
    unknown = np.sum(channel_4 == 0.0)
    free = np.sum(channel_4 == 0.5)
    obstacles_ch4 = np.sum(channel_4 == 1.0)
    total = channel_4.size
    
    print(f"\nChannel 4 statistics:")
    print(f"  Unknown (0.0):   {unknown:4d} cells ({unknown/total*100:.1f}%)")
    print(f"  Free (0.5):      {free:4d} cells ({free/total*100:.1f}%)")
    print(f"  Obstacle (1.0):  {obstacles_ch4:4d} cells ({obstacles_ch4/total*100:.1f}%)")
    
    # Compare counts
    if obstacles_ch4 != len(state.discovered_obstacles):
        print(f"  ❌ Channel 4 has {obstacles_ch4} obstacles but discovered_obstacles has {len(state.discovered_obstacles)}!")
    else:
        print(f"  ✓ Channel 4 obstacle count matches discovered_obstacles")
    
    # Check specific positions
    print(f"\nSampling specific cells around agent {state.position}:")
    px, py = state.position
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            x, y = px + dx, py + dy
            if 0 <= x < 20 and 0 <= y < 20:
                ch4_val = channel_4[y, x]
                in_local = (x, y) in state.local_map
                if in_local:
                    cov, ctype = state.local_map[(x, y)]
                    local_str = f"local_map: cov={cov:.2f}, type='{ctype}'"
                else:
                    local_str = "local_map: NOT PRESENT"
                
                in_disc = (x, y) in state.discovered_obstacles
                
                print(f"  ({x:2d}, {y:2d}): ch4={ch4_val:.1f} | {local_str} | discovered={in_disc}")
    
    # Run 5 steps
    print(f"\n{'='*70}")
    print(f"Running 5 steps...")
    print(f"{'='*70}")
    
    for step in range(1, 6):
        action = agent.select_action(state, env.world_state, epsilon=0.5)
        state, reward, done, info = env.step(action)
        
        if step == 5:
            print(f"\n{'='*70}")
            print(f"STEP {step}")
            print(f"{'='*70}")
            print(f"Agent position: {state.position}")
            print(f"Discovered obstacles: {len(state.discovered_obstacles)}")
            print(f"Local map size: {len(state.local_map)}")
            
            free_count = sum(1 for (pos, (cov, ctype)) in state.local_map.items() if ctype == "free")
            obstacle_count = sum(1 for (pos, (cov, ctype)) in state.local_map.items() if ctype == "obstacle")
            
            print(f"  - Free cells in local_map: {free_count}")
            print(f"  - Obstacle cells in local_map: {obstacle_count}")
            
            # Encode and check
            grid_tensor = agent._encode_state(state, env.world_state)
            channel_4 = grid_tensor[0, 4].cpu().numpy()
            
            unknown = np.sum(channel_4 == 0.0)
            free = np.sum(channel_4 == 0.5)
            obstacles_ch4 = np.sum(channel_4 == 1.0)
            
            print(f"\nChannel 4 statistics:")
            print(f"  Unknown (0.0):   {unknown:4d} cells")
            print(f"  Free (0.5):      {free:4d} cells")
            print(f"  Obstacle (1.0):  {obstacles_ch4:4d} cells")
            
            if obstacles_ch4 != len(state.discovered_obstacles):
                print(f"  ❌ MISMATCH! Channel 4: {obstacles_ch4}, discovered_obstacles: {len(state.discovered_obstacles)}")
            else:
                print(f"  ✓ Counts match")
    
    print(f"\n{'='*70}")
    print(f"DEBUG COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    debug_channel_4()
