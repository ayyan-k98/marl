# Gradient Explosion Fix - POMDP Partial Obstacle Maps

**Date:** November 4, 2025  
**Issue:** Gradient explosion at episode 100 (norm: 15.5, threshold: 12.0)  
**Root Cause:** Partial obstacle map implementation increased state complexity

---

## 🔴 Problem Analysis

### Training Crash Details
```
Ep  100/1500 | Cov: 54.5% | R: 405.5 | ε: 0.133 | Loss: 0.3597
⚠ HIGH GRADIENT NORM: 15.5 (10-ep avg: 16.6, threshold: 12.0)

🛑 TRAINING STOPPED: Gradient explosion detected
```

### Why It Happened
The **partial obstacle map (POMDP)** implementation added significant complexity:

1. **3-value Channel 4** (0.0/0.5/1.0) vs previous binary (0.0/1.0)
   - More nuanced obstacle representation
   - Larger Q-value ranges needed to distinguish states

2. **Sparse Rewards** with COVERAGE_THRESHOLD=0.85
   - Cells need 85% coverage to count
   - Reduced reward frequency → larger Q-value accumulation

3. **Partial Observability**
   - Agents don't see full map initially
   - More exploration needed → longer episodes → larger cumulative rewards

4. **Communication Overhead**
   - 6th channel (agent occupancy) + obstacle sharing
   - More complex coordination patterns to learn

---

## ✅ Applied Fixes

### Fix 1: Learning Rate Reduction
**Before:**
```python
LEARNING_RATE: float = 1e-5
LEARNING_RATE_MIN: float = 5e-6
```

**After:**
```python
LEARNING_RATE: float = 5e-6  # 50% reduction
LEARNING_RATE_MIN: float = 2e-6
```

**Impact:** Smaller Q-value updates → slower gradient growth

---

### Fix 2: Batch Size Reduction
**Before:**
```python
BATCH_SIZE: int = 256  # Large batch for stable gradients
```

**After:**
```python
BATCH_SIZE: int = 128  # Smaller batches for POMDP
```

**Impact:** 
- Faster convergence with partial observability
- Less GPU memory usage
- More frequent updates with noisier but responsive gradients

---

### Fix 3: Target Network Update Frequency
**Before:**
```python
TARGET_UPDATE_FREQ: int = 200  # Every 200 steps
POLYAK_TAU: float = 0.005
```

**After:**
```python
TARGET_UPDATE_FREQ: int = 100  # Every 100 steps (2× more frequent)
POLYAK_TAU: float = 0.01       # Faster target tracking
```

**Impact:**
- Target network tracks policy network more closely
- Reduces Q-value divergence
- Polyak averaging ensures smooth updates

---

### Fix 4: Gradient Clipping
**Before:**
```python
GRAD_CLIP_NORM: float = 2.0
```

**After:**
```python
GRAD_CLIP_NORM: float = 1.0  # Very aggressive clipping
```

**Impact:**
- Hard limit on maximum gradient magnitude
- Prevents explosive updates
- May slow learning slightly, but ensures stability

---

### Fix 5: Reward Scaling (CRITICAL)
**Before:**
```python
COVERAGE_REWARD: float = 1.2
COVERAGE_THRESHOLD: float = 0.85  # Very high!
EXPLORATION_REWARD: float = 0.07
FRONTIER_BONUS: float = 0.012
FRONTIER_CAP: float = 0.25
```

**After:**
```python
COVERAGE_REWARD: float = 0.6    # 50% reduction
COVERAGE_THRESHOLD: float = 0.70  # 15% reduction (less sparse)
EXPLORATION_REWARD: float = 0.04  # 50% reduction
FRONTIER_BONUS: float = 0.006     # 50% reduction
FRONTIER_CAP: float = 0.12        # 50% reduction
```

**Impact:**
- **Episode reward reduced from ~400 to ~200**
- Less reward accumulation per episode
- More frequent rewards (threshold 0.70 vs 0.85)
- Smaller Q-values → more stable gradients

---

## 📊 Expected Performance After Fix

### Training Stability
**Before Fix:**
- Gradient norm: 15.5 at episode 100
- Training crashed
- Q-values diverging

**After Fix (Expected):**
- Gradient norm: <8.0 throughout training
- Smooth learning curve
- Stable Q-values

### Coverage Performance
**Episode 50:**
- Expected: 55-60% coverage (vs 61% before crash)
- Lower due to reduced exploration reward

**Episode 100:**
- Expected: 60-65% coverage
- More stable, less overfitting

**Episode 500:**
- Expected: 70-75% coverage
- Gradual improvement with POMDP

**Episode 1500:**
- Expected: 75-80% validation coverage
- Slightly lower than 80-85% with full knowledge (POMDP trade-off)

---

## 🚀 Retry Training

Run the corrected training:

```powershell
python train_fcn.py
```

### Monitor These Metrics

**✅ Good Signs:**
- Gradient norm < 8.0 consistently
- Loss decreases smoothly (0.6 → 0.2 over 200 episodes)
- Coverage improves gradually (55% → 75% over 1500 episodes)
- Epsilon decays to 0.05-0.10

**🚨 Warning Signs:**
- Gradient norm > 10.0
- Loss plateaus or increases
- Coverage stuck below 50% after 300 episodes
- Epsilon doesn't decay (agent not learning)

---

## 🔧 If Still Unstable

### Additional Fixes (Apply If Needed)

**Fix 6: Even Lower Learning Rate**
```python
LEARNING_RATE: float = 2e-6  # Ultra-conservative
```

**Fix 7: Increase Polyak Tau**
```python
POLYAK_TAU: float = 0.02  # Even faster target tracking
```

**Fix 8: Reduce Gamma**
```python
GAMMA: float = 0.97  # Discount future rewards more (was 0.99)
```

**Fix 9: Smaller Replay Buffer**
```python
REPLAY_BUFFER_SIZE: int = 25000  # Half size (was 50000)
```

---

## 📝 Technical Explanation

### Why Gradient Explosion Happens

**Q-Learning Update:**
```
Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
```

**With POMDP:**
1. **Larger rewards** (r) due to cumulative coverage over longer episodes
2. **Higher Q-targets** (γ max Q(s', a')) because episodes last longer
3. **Sparse rewards** (threshold 0.85) create occasional huge updates
4. **Complex state space** (3-value obstacles + partial observability) requires larger Q-ranges

**Result:**
```
Gradient = ∂Loss/∂θ = ∂(Q_pred - Q_target)²/∂θ

If |Q_target| >> |Q_pred|:
  → Large squared error
  → Large gradient
  → Explosive parameter updates
  → Divergence
```

**Our Fix:**
- Reduce rewards → Smaller Q-targets
- More frequent target updates → Q_pred tracks Q_target better
- Lower learning rate → Smaller parameter updates
- Aggressive clipping → Hard limit on gradient magnitude

---

## ✅ Validation

After training completes, verify:

```powershell
# Check gradient norms in training log
python check_training_progress.py

# Look for:
# - Max gradient norm < 8.0
# - No divergence warnings
# - Smooth loss curve
```

---

## 🎯 Summary

| Parameter | Before | After | Change |
|-----------|--------|-------|--------|
| **Learning Rate** | 1e-5 | 5e-6 | -50% |
| **Batch Size** | 256 | 128 | -50% |
| **Target Update Freq** | 200 | 100 | 2× faster |
| **Polyak Tau** | 0.005 | 0.01 | 2× faster |
| **Grad Clip** | 2.0 | 1.0 | -50% |
| **Coverage Reward** | 1.2 | 0.6 | -50% |
| **Coverage Threshold** | 0.85 | 0.70 | -15% |
| **Exploration Reward** | 0.07 | 0.04 | -50% |
| **Frontier Bonus** | 0.012 | 0.006 | -50% |

**Net Effect:** Episode reward reduced from ~400 to ~200, gradient norms should stay below 8.0

---

**Status:** ✅ READY TO RETRY - Configuration optimized for POMDP partial obstacle maps
