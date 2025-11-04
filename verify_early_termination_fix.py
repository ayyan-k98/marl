"""
Quick test to verify early termination fix.

The bug was: _get_coverage_percentage() returns 0.0-1.0, but _check_done() was dividing by 100.
Result: 95.5% coverage became 0.00955, never >= 0.85 target.

After fix: Should terminate at 85%+ coverage.
"""

from config import config
from environment import CoverageEnvironment
from fcn_agent import FCNAgent
import numpy as np

print("=" * 80)
print("EARLY TERMINATION FIX VERIFICATION")
print("=" * 80)

# Check config
print(f"\n✓ Configuration:")
print(f"  ENABLE_EARLY_TERMINATION: {config.ENABLE_EARLY_TERMINATION}")
print(f"  EARLY_TERM_COVERAGE_TARGET: {config.EARLY_TERM_COVERAGE_TARGET:.1%}")
print(f"  EARLY_TERM_MIN_STEPS: {config.EARLY_TERM_MIN_STEPS}")
print(f"  MAX_EPISODE_STEPS: {config.MAX_EPISODE_STEPS}")

# Test environment
env = CoverageEnvironment(grid_size=20, map_type="empty")
agent = FCNAgent(grid_size=20, input_channels=6)

print(f"\n✓ Running episode on empty 20×20 grid...")
print(f"  Expected: Episode terminates at 85%+ coverage (after step 50)")

state = env.reset()
dummy_occupancy = np.zeros((20, 20), dtype=np.float32)

step_count = 0
terminated_early = False

for step in range(config.MAX_EPISODE_STEPS):
    # Encode and act
    grid_tensor = agent._encode_state(state, env.world_state, dummy_occupancy)
    valid_actions = env.get_valid_actions()
    action = agent.select_action_from_tensor(grid_tensor, valid_actions=valid_actions)
    
    # Step
    next_state, reward, done, info = env.step(action)
    step_count += 1
    
    # Log progress every 10 steps
    if step % 10 == 0:
        print(f"  Step {step_count}: Coverage = {info['coverage_pct']:.1%}")
    
    if done:
        if step_count < config.MAX_EPISODE_STEPS:
            terminated_early = True
            print(f"\n✅ EARLY TERMINATION at step {step_count}!")
            print(f"  Final coverage: {info['coverage_pct']:.1%}")
            print(f"  Early completion flag: {info['early_completion']}")
            print(f"  Completion bonus: +{info['completion_bonus']:.2f}")
            print(f"  Time bonus: +{info['time_bonus']:.2f}")
            print(f"  Steps saved: {config.MAX_EPISODE_STEPS - step_count}")
        else:
            print(f"\n⚠️  Terminated at MAX_STEPS ({step_count})")
            print(f"  Final coverage: {info['coverage_pct']:.1%}")
        break
    
    state = next_state

print("\n" + "=" * 80)
if terminated_early and info['coverage_pct'] >= config.EARLY_TERM_COVERAGE_TARGET:
    print("✅ TEST PASSED: Early termination works correctly!")
    print(f"  Episode ended at step {step_count} (before max {config.MAX_EPISODE_STEPS})")
    print(f"  Coverage: {info['coverage_pct']:.1%} >= {config.EARLY_TERM_COVERAGE_TARGET:.1%}")
elif not terminated_early:
    print("❌ TEST FAILED: Episode ran to MAX_STEPS without early termination")
    print(f"  This should only happen if coverage never reached {config.EARLY_TERM_COVERAGE_TARGET:.1%}")
else:
    print("⚠️  TEST INCONCLUSIVE: Episode terminated early but below target")
    print(f"  Coverage: {info['coverage_pct']:.1%} < {config.EARLY_TERM_COVERAGE_TARGET:.1%}")
print("=" * 80)
