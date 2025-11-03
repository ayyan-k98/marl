"""
Quick script to check multi-agent training progress.
Run periodically to monitor without interrupting training.
"""

import os
import re
from pathlib import Path

def parse_latest_metrics():
    """Parse metrics from latest log or checkpoint."""
    metrics_dir = Path('multi_agent_results/metrics')
    
    if not metrics_dir.exists():
        print("Metrics directory not found. Training may not have started.")
        return
    
    # Look for latest metrics file
    metric_files = sorted(metrics_dir.glob('*.txt'), key=os.path.getmtime)
    
    if not metric_files:
        print("No metrics files found yet.")
        return
    
    latest = metric_files[-1]
    print(f"\n{'='*70}")
    print(f"Latest metrics: {latest.name}")
    print(f"{'='*70}\n")
    
    with open(latest, 'r') as f:
        content = f.read()
        # Show last 500 characters
        print(content[-500:] if len(content) > 500 else content)

def check_checkpoints():
    """Check what checkpoints exist."""
    checkpoint_dir = Path('multi_agent_results/checkpoints')
    
    if not checkpoint_dir.exists():
        print("\nNo checkpoints yet.")
        return
    
    checkpoints = sorted(checkpoint_dir.glob('*.pt'))
    
    if checkpoints:
        print(f"\n{'='*70}")
        print(f"Checkpoints saved: {len(checkpoints)}")
        print(f"{'='*70}")
        for ckpt in checkpoints[-5:]:  # Show last 5
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"  {ckpt.name} ({size_mb:.1f} MB)")
    else:
        print("\nNo checkpoints saved yet.")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-AGENT TRAINING PROGRESS CHECK")
    print("="*70)
    
    parse_latest_metrics()
    check_checkpoints()
    
    print("\n" + "="*70)
    print("Tip: Run this script periodically to monitor training")
    print("="*70 + "\n")
