"""
Quick FCN Testing Script - Grid-Size Invariance Validation

Tests a trained FCN checkpoint on different grid sizes to validate
spatial softmax grid-size invariance claims.

Usage:
    # Test single size
    python quick_test_fcn.py --checkpoint checkpoints/fcn_final.pt --grid-size 30 --test-episodes 20
    
    # Test multiple sizes (bash loop)
    for SIZE in 25 30 35 40 50; do
        python quick_test_fcn.py --checkpoint checkpoints/fcn_final.pt --grid-size $SIZE --test-episodes 20
    done
"""

import argparse
import os
import json
import time
from typing import Dict, List
import numpy as np
import torch

from fcn_agent import FCNAgent
from environment import CoverageEnvironment
from map_generator import MapGenerator
from config import config

# CRITICAL: Enable probabilistic mode (checkpoint was trained with --probabilistic)
config.USE_PROBABILISTIC_ENV = True


def test_checkpoint_on_grid_size(
    checkpoint_path: str,
    grid_size: int,
    test_episodes: int = 20,
    map_types: List[str] = None,
    epsilon: float = 0.0,
    verbose: bool = True,
    save_results: str = None
) -> Dict:
    """
    Test a trained checkpoint on a specific grid size.
    
    Args:
        checkpoint_path: Path to checkpoint file
        grid_size: Grid size to test on
        test_episodes: Number of test episodes per map type
        map_types: List of map types to test (default: ['empty', 'random', 'room'])
        epsilon: Exploration rate (0.0 = greedy, >0 = exploration)
        verbose: Print progress
        save_results: Path to save JSON results (optional)
    
    Returns:
        results: Dictionary with coverage statistics
    """
    
    if map_types is None:
        map_types = ['empty', 'random', 'room']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print("=" * 80)
        print(f"TESTING GRID-SIZE INVARIANCE")
        print("=" * 80)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Grid size: {grid_size}×{grid_size}")
        print(f"Test episodes: {test_episodes} per map type")
        print(f"Map types: {map_types}")
        print(f"Epsilon: {epsilon:.3f} ({'greedy' if epsilon == 0 else 'exploring'})")
        print(f"Device: {device}")
        print("=" * 80)
        print()
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        # Try to resolve relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, checkpoint_path)
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Also tried: {alt_path}\n"
                f"Current directory: {os.getcwd()}"
            )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get training grid size from checkpoint (if available)
    training_grid_size = checkpoint.get('grid_size', 20)  # Default to 20 if not saved
    
    if verbose:
        print(f"✓ Checkpoint loaded")
        print(f"  Training grid size: {training_grid_size}×{training_grid_size}")
        print(f"  Testing grid size: {grid_size}×{grid_size}")
        if grid_size != training_grid_size:
            size_change = (grid_size / training_grid_size - 1) * 100
            print(f"  Size change: {size_change:+.1f}% ({training_grid_size}→{grid_size})")
        print()
    
    # Create agent with TEST grid size
    # CRITICAL: Must match the number of channels the checkpoint was trained with!
    # We need to detect this from the checkpoint
    first_layer_shape = checkpoint['policy_net_state_dict']['encoder.0.weight'].shape
    conv_channels = first_layer_shape[1]  # Channels after CoordConv
    
    # If CoordConv is used (config.USE_COORDCONV=True), subtract 2 coord channels
    # to get the actual input_channels parameter
    trained_input_channels = conv_channels - 2 if config.USE_COORDCONV else conv_channels
    
    agent = FCNAgent(
        grid_size=grid_size,
        input_channels=trained_input_channels,
        device=device
    )
    
    # Load FULL agent state (including epsilon, optimizer, etc.)
    # This is critical - we need the trained epsilon value!
    agent.load(checkpoint_path)
    agent.policy_net.eval()
    
    if verbose:
        print(f"✓ Agent created for {grid_size}×{grid_size} grid")
        print(f"  Input channels: {agent.input_channels}")
        print(f"  Network parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
        print()
    
    # Scale sensor range proportionally (CRITICAL for fair comparison)
    # IMPORTANT: Training validation uses the CoverageEnvironment default sensor_range=3.0
    # NOT config.SENSOR_RANGE (4.0)! This was the bug causing the mismatch.
    # Scaling: 20×20 with range 3.0 → 30×30 with range 4.5 (same % of grid)
    training_sensor_range = 3.0  # CoverageEnvironment default (CONFIRMED)
    scaled_sensor_range = training_sensor_range * (grid_size / training_grid_size)
    
    if verbose:
        print(f"✓ Sensor range scaling:")
        print(f"  Training: {training_sensor_range:.1f} on {training_grid_size}×{training_grid_size} ({training_sensor_range/training_grid_size*100:.1f}% of grid)")
        print(f"  Testing: {scaled_sensor_range:.1f} on {grid_size}×{grid_size} ({scaled_sensor_range/grid_size*100:.1f}% of grid)")
        print()
    
    # Results storage
    results = {
        'checkpoint': checkpoint_path,
        'training_grid_size': training_grid_size,
        'test_grid_size': grid_size,
        'size_change_pct': (grid_size / training_grid_size - 1) * 100,
        'test_episodes': test_episodes,
        'epsilon': 0.1,  # Validation epsilon (matches training validation)
        'sensor_range': scaled_sensor_range,
        'map_types': {},
        'overall': {}
    }
    
    # Test each map type
    # Note: MapGenerator will be created per-episode with correct grid_size
    
    # Create dummy occupancy for single-agent testing (if network expects 6 channels)
    dummy_occupancy = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    # CRITICAL: Use validation epsilon (0.1) to match training validation
    # Training validation explicitly sets epsilon=0.1 (see train_fcn.py line 349)
    agent.set_epsilon(0.1)
    
    if verbose:
        print(f"✓ Test configuration:")
        print(f"  Agent epsilon: {agent.epsilon:.3f}")
        print(f"  Dummy occupancy shape: {dummy_occupancy.shape}")
        print(f"  Using action masking: Yes")
        print()
    
    for map_type in map_types:
        if verbose:
            print(f"Testing on {map_type} maps...")
        
        coverages = []
        rewards = []
        lengths = []
        
        for episode in range(test_episodes):
            # Create environment with SCALED sensor range
            env = CoverageEnvironment(
                grid_size=grid_size,
                sensor_range=scaled_sensor_range,
                map_type=map_type
            )
            # Scale steps by AREA (quadratic) to maintain coverage difficulty
            # This ensures larger grids have proportionally more steps to reach same coverage %
            # 20×20 with 350 steps → 30×30 with 787 steps (2.25x area, 2.25x steps)
            training_steps = config.MAX_EPISODE_STEPS  # 350 for training
            env.max_steps = int(training_steps * ((grid_size * grid_size) / (training_grid_size * training_grid_size)))
            
            # Reset
            env.reset()
            done = False
            episode_reward = 0
            step = 0
            
            # Run episode
            while not done:
                # Get valid actions (ACTION MASKING - critical for correct behavior!)
                valid_actions = env.get_valid_actions()
                
                # Select action (greedy or epsilon-greedy)
                if np.random.random() < epsilon:
                    # Random exploration from valid actions only
                    valid_indices = np.where(valid_actions)[0]
                    action = np.random.choice(valid_indices) if len(valid_indices) > 0 else 0
                else:
                    # Use agent with action masking and occupancy channel
                    # Don't pass epsilon parameter - use agent's internal epsilon
                    action = agent.select_action(
                        env.robot_state, 
                        env.world_state, 
                        agent_occupancy=dummy_occupancy,
                        valid_actions=valid_actions
                    )
                
                _, reward, done, info = env.step(action)
                episode_reward += reward
                step += 1
            
            # Record results
            coverage = info['coverage_pct']  # Already in 0-1 range despite the name
            coverages.append(coverage)
            rewards.append(episode_reward)
            lengths.append(step)
            
            if verbose and (episode + 1) % 5 == 0:
                avg_cov = np.mean(coverages)
                print(f"  Episode {episode + 1}/{test_episodes}: Coverage = {coverage:.1%} (avg: {avg_cov:.1%})")
        
        # Compute statistics
        results['map_types'][map_type] = {
            'mean_coverage': float(np.mean(coverages)),
            'std_coverage': float(np.std(coverages)),
            'min_coverage': float(np.min(coverages)),
            'max_coverage': float(np.max(coverages)),
            'mean_reward': float(np.mean(rewards)),
            'mean_length': float(np.mean(lengths)),
            'coverages': [float(c) for c in coverages]
        }
        
        if verbose:
            print(f"  Results: {np.mean(coverages)*100:.1f}% ± {np.std(coverages)*100:.1f}%")
            print(f"    Range: [{np.min(coverages)*100:.1f}%, {np.max(coverages)*100:.1f}%]")
            print()
    
    # Overall statistics (average across all map types)
    all_coverages = []
    for map_type in map_types:
        all_coverages.extend(results['map_types'][map_type]['coverages'])
    
    results['overall'] = {
        'mean_coverage': float(np.mean(all_coverages)),
        'std_coverage': float(np.std(all_coverages)),
        'min_coverage': float(np.min(all_coverages)),
        'max_coverage': float(np.max(all_coverages)),
    }
    
    # Print summary
    if verbose:
        print("=" * 80)
        print(f"RESULTS SUMMARY")
        print("=" * 80)
        print(f"Grid size: {grid_size}×{grid_size} (trained on {training_grid_size}×{training_grid_size})")
        print()
        
        for map_type in map_types:
            stats = results['map_types'][map_type]
            print(f"{map_type.capitalize():12s}: {stats['mean_coverage']*100:5.1f}% ± {stats['std_coverage']*100:4.1f}%")
        
        print(f"{'Overall':12s}: {results['overall']['mean_coverage']*100:5.1f}% ± {results['overall']['std_coverage']*100:4.1f}%")
        print("=" * 80)
        print()
    
    # Save results to JSON
    if save_results:
        os.makedirs(os.path.dirname(save_results) if os.path.dirname(save_results) else '.', exist_ok=True)
        with open(save_results, 'w') as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"✓ Results saved to {save_results}")
            print()
    
    return results


def compare_multiple_sizes(
    checkpoint_path: str,
    grid_sizes: List[int],
    test_episodes: int = 20,
    map_types: List[str] = None,
    epsilon: float = 0.0,
    output_dir: str = "results"
) -> Dict:
    """
    Test checkpoint on multiple grid sizes and compare results.
    
    Args:
        checkpoint_path: Path to checkpoint
        grid_sizes: List of grid sizes to test
        test_episodes: Episodes per size per map type
        map_types: Map types to test
        epsilon: Exploration rate
        output_dir: Directory to save results
    
    Returns:
        comparison: Dictionary with comparison statistics
    """
    
    print("=" * 80)
    print("GRID-SIZE INVARIANCE COMPARISON")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Grid sizes: {grid_sizes}")
    print(f"Test episodes: {test_episodes} per size per map type")
    print("=" * 80)
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for grid_size in grid_sizes:
        save_path = os.path.join(output_dir, f"results_size_{grid_size}.json")
        
        results = test_checkpoint_on_grid_size(
            checkpoint_path=checkpoint_path,
            grid_size=grid_size,
            test_episodes=test_episodes,
            map_types=map_types,
            epsilon=epsilon,
            verbose=True,
            save_results=save_path
        )
        
        all_results[grid_size] = results
    
    # Create comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    
    # Get training size
    training_size = all_results[grid_sizes[0]]['training_grid_size']
    baseline_coverage = None
    
    print(f"\n{'Grid Size':>12s} {'Change':>8s} {'Coverage':>12s} {'Degradation':>12s} {'Status':>10s}")
    print("-" * 80)
    
    for size in grid_sizes:
        results = all_results[size]
        coverage = results['overall']['mean_coverage']
        std = results['overall']['std_coverage']
        size_change = results['size_change_pct']
        
        if size == training_size:
            baseline_coverage = coverage
            degradation = 0.0
            status = "BASELINE"
            print(f"{size:>3d}×{size:<3d} {size_change:>+6.1f}% {coverage*100:>6.1f}% ± {std*100:>3.1f}% {degradation:>10.1f}% {status:>10s}")
        else:
            degradation = (coverage - baseline_coverage) / baseline_coverage * 100 if baseline_coverage else 0
            
            # Status based on degradation
            if abs(degradation) < 10:
                status = "✓ GOOD"
            elif abs(degradation) < 20:
                status = "⚠ OK"
            else:
                status = "✗ POOR"
            
            print(f"{size:>3d}×{size:<3d} {size_change:>+6.1f}% {coverage*100:>6.1f}% ± {std*100:>3.1f}% {degradation:>+10.1f}% {status:>10s}")
    
    print("-" * 80)
    print()
    
    # Interpretation
    print("INTERPRETATION:")
    print("-" * 80)
    
    # Find max degradation
    max_degradation = 0
    for size in grid_sizes:
        if size != training_size:
            results = all_results[size]
            coverage = results['overall']['mean_coverage']
            degradation = abs((coverage - baseline_coverage) / baseline_coverage * 100) if baseline_coverage else 0
            max_degradation = max(max_degradation, degradation)
    
    if max_degradation < 10:
        verdict = "✓ STRONG INVARIANCE"
        interpretation = "Spatial softmax provides excellent grid-size invariance."
        recommendation = "Claim validated! Proceed with publication emphasizing this."
    elif max_degradation < 20:
        verdict = "⚠ MODERATE INVARIANCE"
        interpretation = "Some invariance, but degrades with larger size changes."
        recommendation = "Consider multi-scale training to improve invariance."
    else:
        verdict = "✗ WEAK INVARIANCE"
        interpretation = "Spatial softmax alone does not provide strong invariance."
        recommendation = "CRITICAL: Implement multi-scale training immediately."
    
    print(f"Verdict: {verdict}")
    print(f"Max degradation: {max_degradation:.1f}%")
    print()
    print(interpretation)
    print()
    print(f"Recommendation: {recommendation}")
    print("=" * 80)
    print()
    
    # Save comparison
    comparison = {
        'checkpoint': checkpoint_path,
        'training_grid_size': training_size,
        'test_grid_sizes': grid_sizes,
        'baseline_coverage': baseline_coverage,
        'max_degradation': max_degradation,
        'verdict': verdict,
        'results': all_results
    }
    
    comparison_path = os.path.join(output_dir, "comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"✓ Comparison saved to {comparison_path}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Test FCN checkpoint on different grid sizes')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (e.g., checkpoints/fcn_final.pt)')
    parser.add_argument('--grid-size', type=int, default=None,
                       help='Single grid size to test (e.g., 30)')
    parser.add_argument('--grid-sizes', type=int, nargs='+', default=None,
                       help='Multiple grid sizes to test (e.g., 20 25 30 35 40 50)')
    parser.add_argument('--test-episodes', type=int, default=20,
                       help='Number of test episodes per map type (default: 20)')
    parser.add_argument('--map-types', type=str, nargs='+', default=['empty', 'random', 'room'],
                       help='Map types to test (default: empty random room)')
    parser.add_argument('--epsilon', type=float, default=0.0,
                       help='Exploration rate (0.0=greedy, default: 0.0)')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Path to save JSON results (default: results/results_size_X.json)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple sizes and generate comparison table')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.grid_size is None and args.grid_sizes is None:
        parser.error("Must specify either --grid-size or --grid-sizes")
    
    if args.grid_size is not None and args.grid_sizes is not None:
        parser.error("Cannot specify both --grid-size and --grid-sizes")
    
    # Single size test
    if args.grid_size is not None:
        save_path = args.save_results
        if save_path is None:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"results_size_{args.grid_size}.json")
        
        results = test_checkpoint_on_grid_size(
            checkpoint_path=args.checkpoint,
            grid_size=args.grid_size,
            test_episodes=args.test_episodes,
            map_types=args.map_types,
            epsilon=args.epsilon,
            verbose=True,
            save_results=save_path
        )
    
    # Multiple size comparison
    elif args.grid_sizes is not None:
        comparison = compare_multiple_sizes(
            checkpoint_path=args.checkpoint,
            grid_sizes=args.grid_sizes,
            test_episodes=args.test_episodes,
            map_types=args.map_types,
            epsilon=args.epsilon,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
