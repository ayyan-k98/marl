# Root Cause Analysis: Grid-Size Invariance Failure

## 🔴 CRITICAL FINDING

The spatial softmax **IS NOT providing true grid-size invariance**. Diagnostic test shows:

```
Best actions across grid sizes: [7, 7, 5, 5, 5]
✗ Network chooses DIFFERENT actions at different grid sizes!

Q-value correlation degradation:
  20×20 vs 25×25: 0.985 ✓
  20×20 vs 30×30: 0.946 ⚠️
  20×20 vs 35×35: 0.883 ❌
  20×20 vs 40×40: 0.833 ❌
```

## 🔍 ROOT CAUSE

The problem is in **`_compute_global_features()`** (fcn_spatial_network.py, lines 276-341):

### The Offending Code:

```python
# Mean distance to uncovered cells
dist_map = torch.sqrt(
    (y_grid - agent_y.view(-1, 1, 1)) ** 2 +
    (x_grid - agent_x.view(-1, 1, 1)) ** 2
)
uncovered_mask = (coverage_channel < 0.5).float()
uncovered_count = uncovered_mask.sum(dim=[1, 2]) + 1e-8
mean_dist_to_uncovered = (dist_map * uncovered_mask).sum(dim=[1, 2]) / uncovered_count
```

### Why This Breaks Invariance:

1. **Coordinates are normalized** (0-1 range) ✓
2. **But perceptual distance changes**:
   - 20×20: 1 cell = 5% of grid → normalized distance = 0.05
   - 40×40: 1 cell = 2.5% of grid → normalized distance = 0.025
   
3. **Same physical distance has different normalized values**:
   - Agent 5 cells from target on 20×20 → distance = 0.25
   - Agent 5 cells from target on 40×40 → distance = 0.125
   
4. **Network learns to associate specific distance values with actions**:
   - At 20×20: "distance 0.25 means move SW"
   - At 40×40: "distance 0.125 is a different context" → chooses NW instead

## 📊 Impact on Performance

| Grid Size | Q-Correlation | Best Action | Coverage | Degradation |
|-----------|---------------|-------------|----------|-------------|
| 20×20 | 1.000 (baseline) | 7 (SW) | 48.3% | 0% |
| 25×25 | 0.985 | 7 (SW) | 39.3% | -18.7% |
| 30×30 | 0.946 | **5 (NW)** | 34.9% | -27.8% |
| 35×35 | 0.883 | **5 (NW)** | 24.2% | -50.0% |
| 40×40 | 0.833 | **5 (NW)** | 21.8% | -55.0% |

The action flip at 30×30 correlates with the sharp performance drop!

## 🛠️ SOLUTIONS

### Option 1: Remove Distance-Based Features (SIMPLEST)

Remove `mean_dist_to_uncovered` from global features:

```python
# In fcn_spatial_network.py, _compute_global_features()
# COMMENT OUT lines 327-336:
# mean_dist_to_uncovered = (dist_map * uncovered_mask).sum(dim=[1, 2]) / uncovered_count
# global_features.append(mean_dist_to_uncovered)

# Reduce num_global_stats from 8 to 7 (line 262)
num_global_stats = 7
```

**Pros**: Simple, guaranteed invariance
**Cons**: Loses useful spatial information

### Option 2: Scale-Normalize Distances (BETTER)

Multiply normalized distances by grid size to get **cell counts**:

```python
# Scale normalized coordinates back to grid cells
dist_map_cells = dist_map * grid_size
mean_dist_cells = (dist_map_cells * uncovered_mask).sum(dim=[1, 2]) / uncovered_count

# Now: 5 cells away = 5.0 regardless of grid size
global_features.append(mean_dist_cells / grid_size)  # Re-normalize for stability
```

**Pros**: Preserves spatial information, grid-size invariant
**Cons**: Requires retraining

### Option 3: Multi-Scale Training (MOST ROBUST)

Train on multiple grid sizes simultaneously:

```python
# During training, randomly sample grid size per episode:
grid_sizes = [15, 20, 25, 30]
grid_size = random.choice(grid_sizes)
env = CoverageEnvironment(grid_size=grid_size, ...)
```

**Pros**: Network learns true invariance, robust to any size
**Cons**: Requires full retraining, longer training time

## 🎯 RECOMMENDED ACTION

**Immediate (No Retraining)**:
1. Remove `mean_dist_to_uncovered` from global features
2. Update `num_global_stats = 7`
3. Re-test grid-size invariance

**Short-Term (Requires Retraining)**:
1. Implement scale-normalized distances (Option 2)
2. Retrain checkpoint for 1000 episodes
3. Verify invariance improves

**Long-Term (Best Solution)**:
1. Implement multi-scale training (Option 3)
2. Train on sizes [15, 20, 25, 30]
3. Test on [20, 30, 40, 50]
4. Publish results showing strong invariance

## 📝 VERIFICATION

After implementing fixes, run:

```bash
py test_network_invariance.py
```

**Expected results**:
- Best action should be **consistent** across all grid sizes
- Q-value correlation should stay **> 0.95** for all sizes
- Coverage degradation should be **< 15%** at 40×40

## 💭 LESSON LEARNED

**Spatial softmax provides invariance at the feature level**, but **global statistics can break it** if they encode scale-dependent information.

**Key insight**: Any feature that has different **semantic meaning** at different scales will break invariance, even if coordinates are normalized.

Distance is inherently scale-dependent:
- 0.1 normalized distance on 20×20 = 2 cells (nearby)
- 0.1 normalized distance on 40×40 = 4 cells (farther)

The network can't distinguish these contexts, leading to different behaviors.

## 🔬 RELATED WORK

DeepMind's original spatial softmax paper (Finn et al., 2016) used:
- Pure spatial features (x, y coordinates only)
- No global distance metrics
- Multi-scale training when needed

Our implementation added distance-based features for better performance on single scale, but this broke invariance.

**Trade-off**: Performance on training size vs. invariance to other sizes.
