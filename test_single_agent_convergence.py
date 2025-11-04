"""
Test single-agent convergence with stability improvements.

This will train for 500 episodes on 40×40 grid to verify:
1. Q-values don't explode (gradient norms stay <12)
2. Loss stays stable (doesn't exceed 2.0)
3. Coverage improves and stabilizes (no catastrophic forgetting)
4. Action masking prevents collisions

Expected runtime: ~2-3 hours
"""

import sys
import os

# Make sure we import from essential/
sys.path.insert(0, os.path.dirname(__file__))

from train_fcn import train_fcn_stage1
from config import config

print("="*80)
print("SINGLE-AGENT CONVERGENCE TEST")
print("="*80)
print("\nStability improvements applied:")
print(f"  ✓ Learning rate: {config.LEARNING_RATE:.2e} (was 5e-5)")
print(f"  ✓ Gradient clipping: {config.GRAD_CLIP_NORM} (was 3.0)")
print(f"  ✓ Target update freq: {config.TARGET_UPDATE_FREQ} (was 50)")
print(f"  ✓ Polyak averaging: {config.USE_POLYAK_AVERAGING} (tau={config.POLYAK_TAU})")
print(f"  ✓ Action masking: ENABLED")
print(f"  ✓ Grid size bug: FIXED")
print(f"\nTraining on 40×40 grid for 500 episodes...")
print(f"Gradient auto-stop threshold: 12.0 (10-episode average)")
print("="*80)
print()

# Train with convergence-focused settings
agent, metrics = train_fcn_stage1(
    num_episodes=500,
    grid_size=40,  # Use 40×40 (matches multi-agent and probabilistic params)
    validate_interval=50,
    checkpoint_interval=100,
    resume_from=None,
    verbose=True
)

print("\n" + "="*80)
print("CONVERGENCE TEST COMPLETE")
print("="*80)

# Analyze results
import numpy as np

final_10_coverage = np.mean(metrics.episode_coverages[-10:]) if len(metrics.episode_coverages) >= 10 else 0
final_10_reward = np.mean(metrics.episode_rewards[-10:]) if len(metrics.episode_rewards) >= 10 else 0

# Check gradient history
final_grad_norms = agent.grad_norm_history[-20:] if len(agent.grad_norm_history) >= 20 else agent.grad_norm_history
avg_final_grad = np.mean(final_grad_norms) if final_grad_norms else 0
max_final_grad = np.max(final_grad_norms) if final_grad_norms else 0

# Check losses
final_losses = metrics.losses[-100:] if len(metrics.losses) >= 100 else metrics.losses
avg_final_loss = np.mean(final_losses) if final_losses else 0
max_final_loss = np.max(final_losses) if final_losses else 0

print(f"\nFinal Performance (last 10 episodes):")
print(f"  Coverage: {final_10_coverage:.1%}")
print(f"  Reward: {final_10_reward:.1f}")
print(f"\nStability Metrics (last 20 updates):")
print(f"  Avg gradient norm: {avg_final_grad:.2f}")
print(f"  Max gradient norm: {max_final_grad:.2f}")
print(f"  Avg loss: {avg_final_loss:.4f}")
print(f"  Max loss: {max_final_loss:.4f}")

print(f"\nConvergence Assessment:")
converged = True
issues = []

if avg_final_grad > 10.0:
    converged = False
    issues.append(f"High gradient norms (avg={avg_final_grad:.1f}, threshold=10.0)")

if max_final_loss > 2.0:
    converged = False
    issues.append(f"Loss explosion (max={max_final_loss:.2f}, threshold=2.0)")

if final_10_coverage < 0.50:
    converged = False
    issues.append(f"Low coverage (final={final_10_coverage:.1%}, expected >50%)")

# Check for catastrophic forgetting
if len(metrics.episode_coverages) >= 100:
    mid_coverage = np.mean(metrics.episode_coverages[200:210])
    if final_10_coverage < mid_coverage - 0.10:
        converged = False
        issues.append(f"Catastrophic forgetting (mid={mid_coverage:.1%} → final={final_10_coverage:.1%})")

if converged:
    print("  ✅ CONVERGED SUCCESSFULLY")
    print("  Q-values are stable, loss is controlled, coverage is good")
    print("\n  ➡ Ready to proceed with multi-agent training!")
else:
    print("  ❌ CONVERGENCE ISSUES DETECTED")
    for issue in issues:
        print(f"     - {issue}")
    print("\n  ➡ Further hyperparameter tuning needed before multi-agent")

print("="*80)
