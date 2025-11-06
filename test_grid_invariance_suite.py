"""
Grid-Size Invariance Test Suite (Python Wrapper)

Cross-platform script to test trained FCN checkpoint on multiple grid sizes.

Usage:
    python test_grid_invariance_suite.py --checkpoint checkpoints/fcn_final.pt
    python test_grid_invariance_suite.py --checkpoint checkpoints/fcn_final.pt --episodes 50
"""

import argparse
import subprocess
import os
import sys
from datetime import datetime


def run_test_suite(checkpoint_path: str, test_episodes: int = 20, grid_sizes: list = None):
    """
    Run complete grid-size invariance test suite.
    
    Args:
        checkpoint_path: Path to checkpoint file
        test_episodes: Number of test episodes per size per map type
        grid_sizes: List of grid sizes to test (default: [20, 25, 30, 35, 40, 50])
    """
    
    if grid_sizes is None:
        grid_sizes = [20, 25, 30, 35, 40, 50]
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_grid_invariance_{timestamp}"
    
    print("=" * 80)
    print("GRID-SIZE INVARIANCE TEST SUITE")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test episodes per size: {test_episodes}")
    print(f"Grid sizes: {grid_sizes}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    print()
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = os.path.dirname(checkpoint_path) or "checkpoints"
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pt') or f.endswith('.pth'):
                    print(f"  - {os.path.join(checkpoint_dir, f)}")
        else:
            print(f"  Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 1: Test individual sizes
    print("Phase 1: Testing individual grid sizes")
    print("-" * 80)
    print()
    
    for size in grid_sizes:
        print(f"Testing {size}×{size} grid...")
        
        cmd = [
            sys.executable, "quick_test_fcn.py",
            "--checkpoint", checkpoint_path,
            "--grid-size", str(size),
            "--test-episodes", str(test_episodes),
            "--output-dir", output_dir,
            "--map-types", "empty", "random", "room"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Test failed for size {size}")
            print(f"Return code: {e.returncode}")
            sys.exit(1)
        
        print()
    
    # Phase 2: Generate comparison
    print("=" * 80)
    print("Phase 2: Generating comparison analysis")
    print("=" * 80)
    print()
    
    cmd = [
        sys.executable, "quick_test_fcn.py",
        "--checkpoint", checkpoint_path,
        "--grid-sizes"] + [str(s) for s in grid_sizes] + [
        "--test-episodes", str(test_episodes),
        "--output-dir", output_dir,
        "--compare"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Comparison failed")
        print(f"Return code: {e.returncode}")
        sys.exit(1)
    
    print()
    print("=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print()
    print("Key files:")
    print(f"  - {output_dir}/comparison.json       (Overall comparison)")
    print(f"  - {output_dir}/results_size_*.json   (Individual results)")
    print()
    print("Next steps:")
    print("  1. Review comparison.json for verdict")
    print("  2. Check degradation percentages")
    print("  3. Decide on next actions based on results:")
    print()
    print("     If degradation < 10%:  ✓ Strong invariance → Publish!")
    print("     If degradation 10-20%: ⚠ Moderate invariance → Consider multi-scale training")
    print("     If degradation > 20%:  ✗ Weak invariance → Implement multi-scale training")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Grid-Size Invariance Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default settings (20 episodes per size)
  python test_grid_invariance_suite.py --checkpoint checkpoints/fcn_final.pt
  
  # Test with more episodes for statistical confidence
  python test_grid_invariance_suite.py --checkpoint checkpoints/fcn_final.pt --episodes 50
  
  # Test specific checkpoint with custom grid sizes
  python test_grid_invariance_suite.py --checkpoint checkpoints/fcn_checkpoint_ep800.pt --episodes 30 --sizes 20 30 40 50
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (e.g., checkpoints/fcn_final.pt)')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of test episodes per size per map type (default: 20)')
    parser.add_argument('--sizes', type=int, nargs='+', default=None,
                       help='Grid sizes to test (default: 20 25 30 35 40 50)')
    
    args = parser.parse_args()
    
    run_test_suite(
        checkpoint_path=args.checkpoint,
        test_episodes=args.episodes,
        grid_sizes=args.sizes
    )


if __name__ == "__main__":
    main()
