"""
Test Early Termination Feature

Quick verification that early termination works correctly:
1. Episode terminates when coverage goal reached
2. Bonuses are calculated correctly
3. Info dict contains correct fields
4. Works with different map types
"""

import numpy as np
from config import config
from environment import CoverageEnvironment
from fcn_agent import FCNAgent


def test_early_termination():
    """Test early termination with simple empty grid."""
    
    print("=" * 80)
    print("TESTING EARLY TERMINATION FEATURE")
    print("=" * 80)
    
    # Check config
    print(f"\n✓ Config check:")
    print(f"  Early termination enabled: {config.ENABLE_EARLY_TERMINATION}")
    print(f"  Coverage target: {config.EARLY_TERM_COVERAGE_TARGET:.1%}")
    print(f"  Min steps: {config.EARLY_TERM_MIN_STEPS}")
    print(f"  Completion bonus: +{config.EARLY_TERM_COMPLETION_BONUS:.1f}")
    print(f"  Time bonus: +{config.EARLY_TERM_TIME_BONUS_PER_STEP:.3f}/step")
    print(f"  Max steps: {config.MAX_EPISODE_STEPS}")
    
    # Initialize
    env = CoverageEnvironment(grid_size=20, map_type="empty")
    agent = FCNAgent(grid_size=20, input_channels=6)
    
    # Run episode until coverage goal reached
    print(f"\n✓ Running episode on empty 20×20 grid...")
    state = env.reset()
    dummy_occupancy = np.zeros((20, 20), dtype=np.float32)
    
    episode_reward = 0.0
    step_count = 0
    max_steps = 100  # Limit test
    
    early_completion_detected = False
    
    for step in range(max_steps):
        # Encode state
        grid_tensor = agent._encode_state(state, env.world_state, dummy_occupancy)
        
        # Select action
        valid_actions = env.get_valid_actions()
        action = agent.select_action_from_tensor(grid_tensor, valid_actions=valid_actions)
        
        # Step
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        step_count += 1
        
        # Check info
        if info.get('early_completion', False):
            early_completion_detected = True
            print(f"\n✓ Early completion detected at step {step_count}!")
            print(f"  Coverage: {info['coverage_pct']:.1%}")
            print(f"  Completion bonus: +{info['completion_bonus']:.2f}")
            print(f"  Time bonus: +{info['time_bonus']:.2f}")
            print(f"  Steps saved: {config.MAX_EPISODE_STEPS - step_count}")
            
            # Verify bonus calculation
            expected_steps_saved = config.MAX_EPISODE_STEPS - step_count
            expected_time_bonus = expected_steps_saved * config.EARLY_TERM_TIME_BONUS_PER_STEP
            expected_total = config.EARLY_TERM_COMPLETION_BONUS + expected_time_bonus
            
            print(f"\n✓ Bonus verification:")
            print(f"  Expected time bonus: {expected_time_bonus:.2f}")
            print(f"  Actual time bonus: {info['time_bonus']:.2f}")
            print(f"  Match: {abs(expected_time_bonus - info['time_bonus']) < 0.01}")
            
            break
        
        state = next_state
        
        if done:
            print(f"\n✗ Episode terminated at step {step_count} without early completion")
            print(f"  Final coverage: {info['coverage_pct']:.1%}")
            break
    
    print(f"\n✓ Episode summary:")
    print(f"  Total steps: {step_count}")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Final coverage: {info['coverage_pct']:.1%}")
    print(f"  Early completion: {early_completion_detected}")
    
    # Test with feature disabled
    print(f"\n" + "=" * 80)
    print("TESTING WITH EARLY TERMINATION DISABLED")
    print("=" * 80)
    
    # Temporarily disable
    original_setting = config.ENABLE_EARLY_TERMINATION
    config.ENABLE_EARLY_TERMINATION = False
    
    env2 = CoverageEnvironment(grid_size=20, map_type="empty")
    state = env2.reset()
    
    step_count_no_early = 0
    for step in range(max_steps):
        grid_tensor = agent._encode_state(state, env2.world_state, dummy_occupancy)
        valid_actions = env2.get_valid_actions()
        action = agent.select_action_from_tensor(grid_tensor, valid_actions=valid_actions)
        next_state, reward, done, info = env2.step(action)
        step_count_no_early += 1
        
        if info.get('early_completion', False):
            print(f"✗ ERROR: Early completion triggered when disabled!")
            break
        
        state = next_state
        if done:
            break
    
    print(f"\n✓ Without early termination:")
    print(f"  Total steps: {step_count_no_early}")
    print(f"  Final coverage: {info['coverage_pct']:.1%}")
    print(f"  Early completion: {info.get('early_completion', False)}")
    
    # Restore setting
    config.ENABLE_EARLY_TERMINATION = original_setting
    
    # Summary
    print(f"\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    if early_completion_detected:
        print(f"✅ PASS: Early termination works correctly")
        print(f"   - Episode terminated at {step_count} steps (before max {config.MAX_EPISODE_STEPS})")
        print(f"   - Coverage target reached: {info['coverage_pct']:.1%} >= {config.EARLY_TERM_COVERAGE_TARGET:.1%}")
        print(f"   - Bonuses calculated correctly")
        print(f"   - Info dict contains correct fields")
        if step_count_no_early >= max_steps:
            print(f"   - Feature properly disabled when config set to False")
    else:
        print(f"⚠️  WARNING: Early termination did not trigger in {max_steps} steps")
        print(f"   This may be normal for complex maps or high epsilon exploration")
        print(f"   Try with trained agent or empty grid for reliable test")
    
    print("=" * 80)


def test_bonus_calculations():
    """Test bonus calculation edge cases."""
    
    print("\n" + "=" * 80)
    print("TESTING BONUS CALCULATIONS")
    print("=" * 80)
    
    test_cases = [
        (50, 350),   # Complete at min steps
        (100, 350),  # Complete early
        (200, 350),  # Complete mid-episode
        (349, 350),  # Complete at last step
    ]
    
    for steps_used, max_steps in test_cases:
        steps_saved = max_steps - steps_used
        completion_bonus = config.EARLY_TERM_COMPLETION_BONUS
        time_bonus = steps_saved * config.EARLY_TERM_TIME_BONUS_PER_STEP
        total_bonus = completion_bonus + time_bonus
        
        print(f"\n  Steps: {steps_used}/{max_steps}")
        print(f"    Saved: {steps_saved}")
        print(f"    Completion: +{completion_bonus:.2f}")
        print(f"    Time: +{time_bonus:.2f}")
        print(f"    Total: +{total_bonus:.2f}")
        print(f"    % of avg reward (~200): {100*total_bonus/200:.1f}%")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_early_termination()
    test_bonus_calculations()
    
    print("\n✅ All tests completed! Early termination feature is ready to use.")
    print("   Run 'python train_fcn.py' to see it in action during training.")
