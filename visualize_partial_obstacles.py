"""
Visual demonstration of partial obstacle map (Channel 4).

Shows how Channel 4 evolves as the agent explores:
- Dark (0.0) = unknown
- Gray (0.5) = explored free space
- White (1.0) = discovered obstacles
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import CoverageEnvironment
from fcn_agent import FCNAgent
from config import config


def visualize_channel_4_evolution():
    """Visualize Channel 4 as agent explores."""
    print("=" * 70)
    print("VISUAL DEMONSTRATION: Partial Obstacle Map (Channel 4)")
    print("=" * 70)
    print("\nEncoding:")
    print("  ⬛ Dark (0.0)  = unexplored / unknown")
    print("  ◼️  Gray (0.5)  = explored free space")
    print("  ⬜ White (1.0) = discovered obstacles")
    print("=" * 70)
    
    # Create environment
    env = CoverageEnvironment(
        grid_size=20,
        sensor_range=4.0,
        map_type='room'
    )
    
    agent = FCNAgent(grid_size=20, input_channels=6)
    state = env.reset()
    
    # Snapshots at different steps
    snapshots = [0, 5, 15, 30]
    channel_4_snapshots = []
    agent_positions = []  # Track agent position at each snapshot
    
    step = 0
    for target_step in snapshots:
        # Run until target step
        while step < target_step:
            action = np.random.randint(0, 9)
            state, reward, done, info = env.step(action)
            step += 1
            if done:
                break
        
        # Capture channel 4 and agent position
        grid_tensor = agent._encode_state(state, env.world_state)
        channel_4 = grid_tensor[0, 4].cpu().numpy()
        channel_4_snapshots.append(channel_4.copy())
        agent_positions.append(state.position)  # Save current position
        
        # Print stats
        unknown = np.sum(channel_4 == 0.0)
        free = np.sum(channel_4 == 0.5)
        obstacles = np.sum(channel_4 == 1.0)
        total = channel_4.size
        
        print(f"\nStep {step}:")
        print(f"  Unknown (0.0):   {unknown:3d} cells ({unknown/total*100:5.1f}%)")
        print(f"  Free (0.5):      {free:3d} cells ({free/total*100:5.1f}%)")
        print(f"  Obstacles (1.0): {obstacles:3d} cells ({obstacles/total*100:5.1f}%)")
        print(f"  Explored: {(free + obstacles)/total*100:5.1f}%")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, (snapshot_step, channel_4, agent_pos) in enumerate(zip(snapshots, channel_4_snapshots, agent_positions)):
        ax = axes[idx]
        
        # Display channel 4 (use grayscale: 0=black, 0.5=gray, 1.0=white)
        im = ax.imshow(channel_4, cmap='gray', vmin=0.0, vmax=1.0, origin='lower')
        
        # Overlay agent position (use correct position for each snapshot)
        agent_x, agent_y = agent_pos
        ax.plot(agent_x, agent_y, 'r*', markersize=15, label='Agent')
        
        ax.set_title(f'Step {snapshot_step} - Channel 4 (Partial Obstacle Map)', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(False)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Value', rotation=270, labelpad=15)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(['Unknown\n(0.0)', 'Free\n(0.5)', 'Obstacle\n(1.0)'])
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'd:/pro/marl/so_far_best_fcn/essential/channel_4_evolution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Show plot
    plt.show()
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 70)
    print("\nKey Observations:")
    print("  • Unknown areas (black) decrease over time")
    print("  • Free space (gray) increases as agent explores")
    print("  • Obstacles (white) are permanently remembered")
    print("  • Agent must explore to discover environment structure")
    print("=" * 70)


def compare_full_vs_partial():
    """Compare full obstacle knowledge vs partial (side-by-side)."""
    print("\n" + "=" * 70)
    print("COMPARISON: Full vs Partial Obstacle Knowledge")
    print("=" * 70)
    
    # Create environment
    env = CoverageEnvironment(
        grid_size=20,
        sensor_range=4.0,
        map_type='corridor'
    )
    
    agent = FCNAgent(grid_size=20, input_channels=6)
    state = env.reset()
    
    # Explore a bit
    for _ in range(20):
        action = np.random.randint(0, 9)
        state, _, done, _ = env.step(action)
        if done:
            break
    
    # Get partial obstacle map (Channel 4)
    grid_tensor = agent._encode_state(state, env.world_state)
    partial_obstacles = grid_tensor[0, 4].cpu().numpy()
    
    # Get full obstacle map (ground truth)
    full_obstacles = np.zeros((20, 20))
    for (x, y) in env.world_state.obstacles:
        if 0 <= x < 20 and 0 <= y < 20:
            full_obstacles[y, x] = 1.0
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Full obstacles (what agent WOULD see if unrealistic)
    axes[0].imshow(full_obstacles, cmap='RdYlGn_r', vmin=0, vmax=1, origin='lower')
    axes[0].set_title('❌ UNREALISTIC: Full Obstacle Knowledge\n(Agent sees ALL obstacles from start)', fontsize=12)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(False)
    
    # Partial obstacles (what agent ACTUALLY sees - realistic)
    axes[1].imshow(partial_obstacles, cmap='gray', vmin=0, vmax=1, origin='lower')
    axes[1].set_title('✅ REALISTIC: Partial Obstacle Knowledge\n(Agent discovers obstacles via exploration)', fontsize=12)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(False)
    
    # Overlay agent position
    ax, ay = state.position
    axes[1].plot(ax, ay, 'r*', markersize=15, label='Agent')
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'd:/pro/marl/so_far_best_fcn/essential/full_vs_partial_obstacles.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison saved to: {output_path}")
    
    plt.show()
    
    # Statistics
    total_obstacles = len(env.world_state.obstacles)
    discovered_obstacles = len(state.discovered_obstacles)
    
    print(f"\n📊 Statistics:")
    print(f"  Total obstacles (ground truth): {total_obstacles}")
    print(f"  Discovered obstacles: {discovered_obstacles}")
    print(f"  Discovery rate: {discovered_obstacles/total_obstacles*100:.1f}%")
    print(f"\n  Agent must explore to discover remaining {total_obstacles - discovered_obstacles} obstacles!")
    
    print("\n" + "=" * 70)


def main():
    """Run visual demonstrations."""
    try:
        # Visualize channel 4 evolution
        visualize_channel_4_evolution()
        
        # Compare full vs partial
        compare_full_vs_partial()
        
        print("\n" + "=" * 70)
        print("✅ ALL VISUALIZATIONS COMPLETE")
        print("=" * 70)
        print("\nGenerated files:")
        print("  • channel_4_evolution.png")
        print("  • full_vs_partial_obstacles.png")
        print("\nThese visualizations demonstrate why partial observability is critical")
        print("for realistic reinforcement learning in exploration tasks.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
