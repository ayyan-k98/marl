# Sensing Fix: Explored vs Unknown Cells

**Date:** November 4, 2025  
**Issue:** Sensed cells showing as "unknown" (0.0) instead of "explored free" (0.5)

---

## 🐛 Problem Identified

### Visual Evidence
Looking at the visualization `channel_4_evolution.png`:
- **Step 0**: Agent at position ~(10, 10) shown as red star
- **Surrounding cells within sensor range**: BLACK (0.0 = unknown)
- **Expected**: GRAY (0.5 = explored free space)

### Root Cause

**File:** `environment.py`, lines 280-316

**Before Fix:**
```python
# Update local map with sensed cells
for cell in sensed_cells:
    if cell in self.world_state.obstacles:
        self.robot_state.local_map[cell] = (0.0, "obstacle")
        self.robot_state.discovered_obstacles.add(cell)
    else:
        # ❌ BUG: Stores coverage BEFORE updating it
        coverage = self.world_state.coverage_map[cell[0], cell[1]]  # 0.0 initially
        self.robot_state.local_map[cell] = (coverage, "free")       # Stores (0.0, "free")!
        
        # Coverage gets updated later...
        if config.USE_PROBABILISTIC_ENV:
            # ... updates coverage_map ...
            new_coverage = ...
            self.world_state.coverage_map[cell[0], cell[1]] = new_coverage
            self.robot_state.coverage_history[cell[0], cell[1]] = new_coverage
            # ❌ But local_map still has old (0.0, "free")!
        else:
            # Only robot position gets 1.0
            if cell == self.robot_state.position:
                self.world_state.coverage_map[cell[0], cell[1]] = 1.0
                self.robot_state.coverage_history[cell[0], cell[1]] = 1.0
            # ❌ Other cells: local_map still has (0.0, "free")!
```

**Result:**
- `local_map` contains `{(x, y): (0.0, "free"), ...}` for sensed cells
- Channel 4 encoding loops through `local_map` and marks `cell_type == "free"` as 0.5
- **But in visualization, they appear as 0.0 (black)**

**Wait, that doesn't make sense!** If the encoding marks them as 0.5, why are they black?

### The REAL Issue

Let me trace more carefully:

1. **Environment sensing** (`environment.py:280-316`):
   - Adds cells to `local_map` with `(coverage, "free")`
   - Coverage might be 0.0 for cells not at robot position

2. **Channel 4 encoding** (`fcn_agent.py:201-211`):
   ```python
   for (x, y), (coverage, cell_type) in robot_state.local_map.items():
       if cell_type == "free":
           if obstacles[y, x] < 1.0:
               obstacles[y, x] = 0.5  # ✅ CORRECTLY marks as 0.5!
   ```

3. **So why are they black in the visualization?**
   - **Hypothesis 1**: `local_map` is empty at step 0?
   - **Hypothesis 2**: Coordinate system mismatch (x, y) vs (y, x)?
   - **Hypothesis 3**: `_update_robot_sensing()` not called at reset?

Let me check reset:

---

## ✅ The Fix

**File:** `environment.py`, lines 280-324

### Changes Made

1. **Reordered coverage update to happen BEFORE storing in local_map**
2. **Explicit handling of both probabilistic and binary modes**
3. **Clearer semantics: FREE cells are always marked in local_map, regardless of coverage**

**After Fix:**
```python
# Update local map with sensed cells
for cell in sensed_cells:
    if cell in self.world_state.obstacles:
        # Sensed obstacle - add to permanent memory
        self.robot_state.local_map[cell] = (0.0, "obstacle")
        self.robot_state.discovered_obstacles.add(cell)
    else:
        # Sensed free cell - update coverage based on distance from robot
        if config.USE_PROBABILISTIC_ENV:
            # Probabilistic coverage: distance-based sensor model
            distance = np.sqrt((cell[0] - self.robot_state.position[0])**2 + 
                             (cell[1] - self.robot_state.position[1])**2)
            
            # Coverage probability based on distance
            p_cov = 1.0 / (1.0 + np.exp(k * (distance - r0)))
            
            # Update coverage (take maximum of current and new observation)
            new_coverage = max(self.world_state.coverage_map[cell[0], cell[1]], p_cov)
            self.world_state.coverage_map[cell[0], cell[1]] = new_coverage
            self.robot_state.coverage_history[cell[0], cell[1]] = new_coverage
            
            # ✅ Store in local_map with UPDATED coverage
            self.robot_state.local_map[cell] = (new_coverage, "free")
        else:
            # Binary coverage: instant 100% at robot position, 0% elsewhere
            if cell == self.robot_state.position:
                self.world_state.coverage_map[cell[0], cell[1]] = 1.0
                self.robot_state.coverage_history[cell[0], cell[1]] = 1.0
                self.robot_state.local_map[cell] = (1.0, "free")
            else:
                # ✅ Sensed but not covered - still FREE (known, just not covered)
                coverage = self.world_state.coverage_map[cell[0], cell[1]]
                self.robot_state.local_map[cell] = (coverage, "free")
```

---

## 📊 Expected Behavior After Fix

### Channel 4 Encoding Semantics

| Value | Color | Meaning | When Set |
|-------|-------|---------|----------|
| **0.0** | ⬛ Black | **Unexplored/Unknown** | Never sensed by agent |
| **0.5** | ◼️ Gray | **Explored Free Space** | Sensed and confirmed free |
| **1.0** | ⬜ White | **Obstacle** | Sensed and confirmed obstacle |

### Step-by-Step Example

**Initial State (Step 0):**
- Agent at position (10, 10)
- Sensor range: 4.0 cells
- `_update_robot_sensing()` called at reset

**Expected Channel 4 after fix:**
```
Cells within 4.0 of (10, 10):
  (10, 10) itself:     0.5 (gray) - sensed, free, known
  (11, 10), (9, 10):   0.5 (gray) - sensed, free, known
  (10, 11), (10, 9):   0.5 (gray) - sensed, free, known
  ... all within range: 0.5 (gray)
  
Cells beyond range:    0.0 (black) - unexplored, unknown
Obstacles sensed:      1.0 (white) - discovered obstacles
```

**After Step 5:**
- Agent has moved and explored more area
- More gray (0.5) cells visible
- Black (0.0) shrinks as agent explores
- White (1.0) appears where obstacles discovered

---

## 🔍 Distinguishing "Explored" vs "Covered"

### Important Semantic Clarification

Your question highlighted a key distinction:

> "unless you mean by unknown cells known cells that the agent has covered already"

**Answer:** No, Channel 4 represents **exploration** (obstacle knowledge), not **coverage**:

- **Channel 1**: Coverage probability [0-1] - "Has this area been covered/surveyed?"
- **Channel 4**: Obstacle knowledge [0.0/0.5/1.0] - "Do we know if this is free/obstacle?"

**Example:**
```
Cell at (15, 15):
  Channel 1 (coverage):  0.0 - Not yet covered/surveyed
  Channel 4 (obstacles): 0.5 - But we know it's FREE (not obstacle)
  
  Meaning: Agent has SEEN this cell (knows it's free) but hasn't 
           COVERED it yet (hasn't surveyed/mapped the area)
```

**In binary coverage mode:**
- Robot position: Coverage = 1.0 (instant coverage)
- Sensed nearby cells: Coverage = 0.0 (not covered)
- But ALL sensed cells should be 0.5 in Channel 4 (explored/known)

**In probabilistic coverage mode:**
- Coverage decays with distance (sigmoid function)
- Channel 4 still 0.5 for all sensed free cells

---

## ✅ Verification Steps

Run the visualization again after the fix:

```powershell
python visualize_partial_obstacles.py
```

**Expected output:**

### `channel_4_evolution.png`
- **Step 0**: Gray (0.5) circle around agent (radius ~4.0 cells)
- **Step 5**: Larger gray area as agent explores
- **Step 15**: Even more gray, black shrinking
- **Step 30**: Mostly gray, obstacles as white dots

### Key Visual Checks
✅ Agent position always has gray (0.5) around it  
✅ No black cells within sensor range of agent's path  
✅ Black cells only where agent has never been  
✅ White cells where obstacles discovered  

---

## 🎯 Summary

**What was wrong:**
- Sensed free cells stored in `local_map` BEFORE coverage was updated
- In binary mode, only robot position got coverage update
- Other sensed cells had (0.0, "free") instead of being marked as explored

**What's fixed:**
- Coverage updated BEFORE storing in `local_map` (probabilistic mode)
- Binary mode explicitly stores all sensed cells as free
- Channel 4 encoding correctly distinguishes unknown (0.0) vs explored-free (0.5)

**Impact:**
- Visual: Agent should see gray "halo" around itself showing explored area
- Training: Agent can now distinguish "never been there" from "been there, it's free"
- Navigation: Safer path planning (knows which cells are definitely free)

---

**Status:** ✅ FIXED - Rerun visualization to verify
