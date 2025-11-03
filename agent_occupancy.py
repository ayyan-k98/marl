"""
Agent occupancy probability channel computation.

Computes P(other agent at cell (x,y)) based on:
1. Last communicated positions
2. Time decay (uncertainty grows with time)
3. Probabilistic union (multiple agents)

This provides a soft representation of where other agents are likely to be,
enabling proactive coordination and collision avoidance.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import torch


class AgentOccupancyComputer:
    """Compute agent occupancy probability channel for multi-agent coordination.
    
    The occupancy channel represents P(another agent occupies cell (x,y)) based on:
    - Communicated positions from other agents
    - Time since last communication (uncertainty grows)
    - Movement model (agents can move with max_velocity)
    
    The resulting probability map is used as the 6th input channel to the FCN,
    allowing agents to anticipate and avoid collisions proactively.
    """
    
    def __init__(self, 
                 grid_size: int,
                 base_sigma: float = 0.5,
                 max_velocity: float = 1.0,
                 time_decay_rate: float = 0.1,
                 prediction_horizon: int = 5,
                 use_predictive: bool = True):
        """
        Initialize agent occupancy computer.
        
        Args:
            grid_size: Size of the grid (assumes square grid)
            base_sigma: Base standard deviation for Gaussian (in grid cells)
            max_velocity: Maximum agent velocity (cells per timestep)
            time_decay_rate: Rate at which uncertainty grows with time
            prediction_horizon: Number of future timesteps to predict (for trajectory-based occupancy)
            use_predictive: If True, use predictive trajectory model; if False, use current position only
        """
        self.grid_size = grid_size
        self.base_sigma = base_sigma
        self.max_velocity = max_velocity
        self.time_decay_rate = time_decay_rate
        self.prediction_horizon = prediction_horizon
        self.use_predictive = use_predictive
        
        # Cache for Gaussian computation (optimization)
        self._gaussian_cache = {}
        
        # Pre-compute coordinate grids for vectorized operations
        self.y_grid, self.x_grid = np.meshgrid(
            np.arange(grid_size), 
            np.arange(grid_size),
            indexing='ij'
        )
    
    def compute(self,
                agent_id: int,
                messages: List[Dict],
                current_time: int) -> np.ndarray:
        """
        Compute occupancy probability map for OTHER agents.
        
        Uses predictive trajectory model if enabled: projects where agents
        will likely be in the next prediction_horizon timesteps based on
        position + velocity, creating Gaussian probability distributions
        along predicted trajectories.
        
        Args:
            agent_id: ID of agent computing occupancy (exclude self)
            messages: List of recent messages from other agents
                     Each message should have:
                     - 'sender_id': int
                     - 'position': (x, y) tuple or [x, y] array
                     - 'velocity': (vx, vy) tuple or [vx, vy] array (optional, for predictive mode)
                     - 'timestamp': int (optional, defaults to current_time)
            current_time: Current timestep
            
        Returns:
            occupancy: [H, W] array of probabilities in [0, 1]
                      Higher values = more likely another agent is/will be there
        """
        if not messages:
            # No messages = no other agents known
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        occupancy = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        for msg in messages:
            # Skip messages from self
            if msg.get('sender_id') == agent_id:
                continue
            
            # Extract position
            position = msg.get('position')
            if position is None:
                continue
            
            # Handle both tuple and array formats
            if isinstance(position, (list, np.ndarray, tuple)):
                x, y = float(position[0]), float(position[1])
            else:
                continue  # Invalid position format
            
            # Extract velocity (for predictive model)
            velocity = msg.get('velocity', (0.0, 0.0))
            if isinstance(velocity, (list, np.ndarray, tuple)):
                vx, vy = float(velocity[0]), float(velocity[1])
            else:
                vx, vy = 0.0, 0.0
            
            # Get timestamp (default to current if not provided)
            timestamp = msg.get('timestamp', current_time)
            delta_t = current_time - timestamp
            
            if self.use_predictive and (abs(vx) > 0.01 or abs(vy) > 0.01):
                # PREDICTIVE MODE: Project trajectory into future
                # Create Gaussian probability blobs at multiple future positions
                
                for t in range(1, self.prediction_horizon + 1):
                    # Predict future position based on velocity
                    future_x = x + vx * t
                    future_y = y + vy * t
                    
                    # Clip to grid bounds
                    if not (0 <= future_x < self.grid_size and 0 <= future_y < self.grid_size):
                        continue  # Out of bounds
                    
                    # Uncertainty grows with prediction horizon
                    # sigma = base + message_age + prediction_time
                    sigma = (self.base_sigma + 
                            self.max_velocity * delta_t +  # Uncertainty from message age
                            self.time_decay_rate * delta_t**2 +
                            0.5 * t)  # Additional uncertainty from prediction
                    
                    # Clamp sigma to reasonable range
                    sigma = max(0.3, min(sigma, self.grid_size / 3))
                    
                    # Compute Gaussian at predicted future position
                    prob_map = self._gaussian_at((future_x, future_y), sigma)
                    
                    # Weight by recency (more recent predictions = higher weight)
                    # Predictions further in future = lower contribution
                    time_weight = np.exp(-0.2 * t)  # Decay with prediction horizon
                    
                    # Add to occupancy (weighted)
                    occupancy = 1.0 - (1.0 - occupancy) * (1.0 - prob_map * time_weight)
            
            else:
                # NON-PREDICTIVE MODE: Use current/last known position
                # (Fallback for stationary agents or when velocity not available)
                
                # Compute uncertainty that grows with time since message
                sigma = (self.base_sigma + 
                        self.max_velocity * delta_t + 
                        self.time_decay_rate * delta_t**2)
                
                # Clamp sigma to reasonable range
                sigma = max(0.1, min(sigma, self.grid_size / 2))
                
                # Compute Gaussian probability distribution centered at position
                prob_map = self._gaussian_at((x, y), sigma)
                
                # Probabilistic union: P(A or B) = 1 - (1-P(A))(1-P(B))
                occupancy = 1.0 - (1.0 - occupancy) * (1.0 - prob_map)
        
        return occupancy.astype(np.float32)
    
    def compute_batch(self,
                     agent_ids: List[int],
                     messages: List[Dict],
                     current_time: int) -> List[np.ndarray]:
        """
        Compute occupancy maps for multiple agents (batch version).
        
        Args:
            agent_ids: List of agent IDs
            messages: List of messages from all agents
            current_time: Current timestep
            
        Returns:
            occupancies: List of [H, W] occupancy maps, one per agent
        """
        return [
            self.compute(agent_id, messages, current_time)
            for agent_id in agent_ids
        ]
    
    def _gaussian_at(self, position: Tuple[float, float], 
                     sigma: float) -> np.ndarray:
        """
        Compute 2D Gaussian probability distribution centered at position.
        
        Uses vectorized operations for efficiency.
        
        Args:
            position: (x, y) center of Gaussian
            sigma: Standard deviation
            
        Returns:
            gaussian: [H, W] probability map
        """
        x, y = position
        
        # Vectorized distance computation
        dist_sq = (self.x_grid - x)**2 + (self.y_grid - y)**2
        
        # Gaussian formula: exp(-d^2 / (2*sigma^2))
        gaussian = np.exp(-dist_sq / (2 * sigma**2))
        
        # Normalize so max probability is 1.0 at center
        max_val = np.max(gaussian)
        if max_val > 0:
            gaussian = gaussian / max_val
        
        return gaussian
    
    def compute_from_positions(self,
                              agent_id: int,
                              positions: List[Tuple[int, Tuple[float, float]]],
                              current_time: int) -> np.ndarray:
        """
        Compute occupancy from simple position list (convenience method).
        
        Args:
            agent_id: ID of agent computing occupancy
            positions: List of (other_agent_id, (x, y)) tuples
            current_time: Current timestep
            
        Returns:
            occupancy: [H, W] probability map
        """
        # Convert positions to message format
        messages = [
            {
                'sender_id': other_id,
                'position': pos,
                'timestamp': current_time
            }
            for other_id, pos in positions
        ]
        
        return self.compute(agent_id, messages, current_time)
    
    def visualize(self, occupancy: np.ndarray) -> str:
        """
        Create ASCII visualization of occupancy map (for debugging).
        
        Args:
            occupancy: [H, W] occupancy map
            
        Returns:
            visualization: ASCII string representation
        """
        # Map probabilities to characters
        chars = ' .:-=+*#@'
        
        lines = []
        for row in occupancy:
            line = ''
            for prob in row:
                idx = min(int(prob * len(chars)), len(chars) - 1)
                line += chars[idx]
            lines.append(line)
        
        return '\n'.join(lines)
    
    def compare_predictive_vs_static(self,
                                     agent_id: int,
                                     messages: List[Dict],
                                     current_time: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compare predictive vs static occupancy maps (for debugging/analysis).
        
        Args:
            agent_id: ID of agent computing occupancy
            messages: List of messages
            current_time: Current timestep
            
        Returns:
            (static_occupancy, predictive_occupancy): Tuple of [H, W] maps
        """
        # Temporarily disable predictive mode
        original_mode = self.use_predictive
        
        # Compute static (current position only)
        self.use_predictive = False
        static_occ = self.compute(agent_id, messages, current_time)
        
        # Compute predictive (trajectory-based)
        self.use_predictive = True
        predictive_occ = self.compute(agent_id, messages, current_time)
        
        # Restore original mode
        self.use_predictive = original_mode
        
        return static_occ, predictive_occ


def create_dummy_occupancy(grid_size: int) -> np.ndarray:
    """
    Create dummy occupancy map (all zeros) for single-agent training.
    
    This allows using 6-channel networks in single-agent mode by providing
    a dummy 6th channel that is always zero.
    
    Args:
        grid_size: Size of grid
        
    Returns:
        dummy: [H, W] array of zeros
    """
    return np.zeros((grid_size, grid_size), dtype=np.float32)


if __name__ == "__main__":
    # Test the agent occupancy computer
    print("=" * 70)
    print("TESTING AgentOccupancyComputer")
    print("=" * 70)
    
    # Test case 1: Static mode (no velocity)
    print("\n" + "=" * 70)
    print("TEST 1: STATIC MODE (no velocity)")
    print("=" * 70)
    
    computer = AgentOccupancyComputer(grid_size=20, use_predictive=False)
    
    messages = [
        {'sender_id': 1, 'position': (10.0, 10.0), 'velocity': (0.0, 0.0), 'timestamp': 0}
    ]
    occupancy = computer.compute(agent_id=0, messages=messages, current_time=0)
    
    print("Single agent at (10, 10), no velocity")
    print(f"Max probability: {occupancy.max():.3f}")
    print(f"Probability at (10,10): {occupancy[10, 10]:.3f}")
    print(f"Probability at (12,10): {occupancy[12, 10]:.3f}")
    print("\nVisualization (rows 8-12, cols 8-12):")
    print(computer.visualize(occupancy[8:13, 8:13]))
    
    # Test case 2: Predictive mode with moving agent
    print("\n" + "=" * 70)
    print("TEST 2: PREDICTIVE MODE (agent moving right)")
    print("=" * 70)
    
    computer_pred = AgentOccupancyComputer(
        grid_size=20, 
        use_predictive=True,
        prediction_horizon=5
    )
    
    messages = [
        {
            'sender_id': 1, 
            'position': (10.0, 10.0),
            'velocity': (1.0, 0.0),  # Moving right at 1 cell/step
            'timestamp': 0
        }
    ]
    occupancy_pred = computer_pred.compute(agent_id=0, messages=messages, current_time=0)
    
    print("Single agent at (10, 10), velocity=(1.0, 0.0) [moving RIGHT]")
    print(f"Max probability: {occupancy_pred.max():.3f}")
    print(f"Probability at (10,10): {occupancy_pred[10, 10]:.3f} (current)")
    print(f"Probability at (11,10): {occupancy_pred[11, 10]:.3f} (t+1)")
    print(f"Probability at (12,10): {occupancy_pred[12, 10]:.3f} (t+2)")
    print(f"Probability at (13,10): {occupancy_pred[13, 10]:.3f} (t+3)")
    print(f"Probability at (15,10): {occupancy_pred[15, 10]:.3f} (t+5)")
    print("\nVisualization (rows 8-12, cols 8-17) - should see trajectory to RIGHT:")
    print(computer_pred.visualize(occupancy_pred[8:13, 8:18]))
    
    # Test case 3: Diagonal movement
    print("\n" + "=" * 70)
    print("TEST 3: PREDICTIVE MODE (agent moving diagonally)")
    print("=" * 70)
    
    messages = [
        {
            'sender_id': 1,
            'position': (10.0, 10.0),
            'velocity': (0.8, 0.6),  # Moving diagonally (right-down)
            'timestamp': 0
        }
    ]
    occupancy_diag = computer_pred.compute(agent_id=0, messages=messages, current_time=0)
    
    print("Single agent at (10, 10), velocity=(0.8, 0.6) [moving DIAGONALLY]")
    print(f"Probability at (10,10): {occupancy_diag[10, 10]:.3f}")
    print(f"Probability at (11,11): {occupancy_diag[11, 11]:.3f} (t+1 approx)")
    print(f"Probability at (12,11): {occupancy_diag[12, 11]:.3f} (t+2 approx)")
    print(f"Probability at (13,13): {occupancy_diag[13, 13]:.3f} (t+4 approx)")
    print("\nVisualization (rows 8-16, cols 8-16) - should see diagonal trajectory:")
    print(computer_pred.visualize(occupancy_diag[8:17, 8:17]))
    
    # Test case 4: Multiple agents with different trajectories
    print("\n" + "=" * 70)
    print("TEST 4: MULTIPLE AGENTS WITH DIFFERENT TRAJECTORIES")
    print("=" * 70)
    
    messages = [
        {
            'sender_id': 1,
            'position': (5.0, 10.0),
            'velocity': (1.0, 0.0),  # Moving RIGHT
            'timestamp': 0
        },
        {
            'sender_id': 2,
            'position': (10.0, 5.0),
            'velocity': (0.0, 1.0),  # Moving DOWN
            'timestamp': 0
        }
    ]
    occupancy_multi = computer_pred.compute(agent_id=0, messages=messages, current_time=0)
    
    print("Agent 1 at (5, 10), moving RIGHT")
    print("Agent 2 at (10, 5), moving DOWN")
    print(f"Max probability: {occupancy_multi.max():.3f}")
    print("\nVisualization (full grid, center region):")
    print(computer_pred.visualize(occupancy_multi[3:15, 3:15]))
    
    # Test case 5: Comparison - static vs predictive
    print("\n" + "=" * 70)
    print("TEST 5: COMPARISON - STATIC vs PREDICTIVE")
    print("=" * 70)
    
    messages = [
        {
            'sender_id': 1,
            'position': (10.0, 10.0),
            'velocity': (1.0, 0.0),
            'timestamp': 0
        }
    ]
    
    static_occ, pred_occ = computer_pred.compare_predictive_vs_static(
        agent_id=0, 
        messages=messages, 
        current_time=0
    )
    
    print("\nSTATIC MODE (position only):")
    print(computer_pred.visualize(static_occ[8:13, 8:18]))
    
    print("\nPREDICTIVE MODE (position + velocity trajectory):")
    print(computer_pred.visualize(pred_occ[8:13, 8:18]))
    
    print("\nKEY INSIGHT:")
    print("  Static:     Agent occupancy concentrated at CURRENT position")
    print("  Predictive: Agent occupancy spreads along FUTURE trajectory")
    print("              → Enables proactive avoidance of future collisions!")
    print("              → Agents can coordinate by staying out of each other's paths!")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED!")
    print("=" * 70)
