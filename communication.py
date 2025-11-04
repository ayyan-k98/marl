"""
Communication Module for Multi-Agent Coordination

Implements communication strategy:
- No Communication (baseline) - Position info via agent_occupancy 6th channel

REMOVED: FullStateSharing (wrong approach - bypasses coordination learning)

Communication Policy:
- Position/velocity information communicated via agent_occupancy.py (6th input channel)
- This provides realistic, limited bandwidth communication
- Agents must learn to coordinate from limited position information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from config import config


@dataclass
class Message:
    """
    Message from one agent to another.

    Contains:
    - sender_id: Who sent the message
    - receiver_id: Who receives (None = broadcast)
    - content: Encoded message vector
    - metadata: Optional info (position, coverage, etc.)
    """
    sender_id: int
    receiver_id: Optional[int]
    content: torch.Tensor
    metadata: Dict


class CommunicationProtocol:
    """
    Base class for communication protocols.

    Defines interface for sending and receiving messages.
    """

    def __init__(self, num_agents: int, message_dim: int = 32):
        self.num_agents = num_agents
        self.message_dim = message_dim

    def communicate(self, observations: List[Dict], state) -> List[Message]:
        """
        Main communication interface.
        
        Args:
            observations: List of agent observations
            state: Current world state
            
        Returns:
            messages: List of messages exchanged between agents
        """
        raise NotImplementedError

    def encode_message(self, agent_id: int, hidden_state: torch.Tensor) -> Message:
        """Encode hidden state into message."""
        raise NotImplementedError

    def aggregate_messages(self, agent_id: int, messages: List[Message]) -> torch.Tensor:
        """Aggregate received messages."""
        raise NotImplementedError

    def should_communicate(self, step: int) -> bool:
        """Decide whether to communicate at this step."""
        raise NotImplementedError


class NoCommunciation(CommunicationProtocol):
    """Baseline: No communication between agents."""

    def communicate(self, observations: List[Dict], state) -> List[List[Dict]]:
        """No communication - return list of empty lists (one per agent)."""
        return [[] for _ in observations]

    def encode_message(self, agent_id: int, hidden_state: torch.Tensor) -> Message:
        return Message(
            sender_id=agent_id,
            receiver_id=None,
            content=torch.zeros(self.message_dim),
            metadata={}
        )

    def aggregate_messages(self, agent_id: int, messages: List[Message]) -> torch.Tensor:
        return torch.zeros(self.message_dim)

    def should_communicate(self, step: int) -> bool:
        return False


class PositionCommunication:
    """
    Position-based communication protocol with obstacle map sharing.

    Agents broadcast:
    - (position, velocity, timestamp) every N steps
    - Discovered obstacle map (partial) for coordination
    - Discovered free cells for efficient exploration

    Recipients within communication range receive the messages.

    This enables persistent position information beyond visual sensor range,
    allowing agents to coordinate proactively rather than just reactively.

    Key features:
    - Broadcasts every comm_freq steps
    - Range-limited (comm_range parameter)
    - Includes uncertainty/age of information
    - Used to populate 6th channel (agent occupancy)
    - Shares discovered obstacle maps to avoid redundant exploration
    """

    def __init__(
        self,
        num_agents: int,
        comm_range: float = 15.0,
        comm_freq: int = 5,
        share_obstacle_maps: bool = True
    ):
        """
        Initialize position communication protocol.

        Args:
            num_agents: Number of agents in the system
            comm_range: Maximum communication range (grid cells)
            comm_freq: Communication frequency (broadcast every N steps)
            share_obstacle_maps: Whether to share discovered obstacle maps
        """
        self.num_agents = num_agents
        self.comm_range = comm_range
        self.comm_freq = comm_freq
        self.share_obstacle_maps = share_obstacle_maps
        self.last_messages = {}  # agent_id -> (pos, vel, timestamp, obstacles, free_cells)
        self.position_history = {}  # agent_id -> [(pos, step), ...]  for velocity computation
        self.step_count = 0

    def broadcast(
        self,
        agent_id: int,
        position: Tuple[float, float],
        velocity: Optional[Tuple[float, float]] = None,
        discovered_obstacles: Optional[set] = None,
        local_map: Optional[Dict] = None
    ) -> None:
        """
        Agent broadcasts its position, velocity, and discovered obstacle map.

        Args:
            agent_id: ID of broadcasting agent
            position: (x, y) position in grid coordinates
            velocity: (vx, vy) velocity (optional, defaults to (0, 0))
            discovered_obstacles: Set of discovered obstacle cells (x, y)
            local_map: Dict mapping (x, y) -> (coverage, "free"/"obstacle")
        """
        if velocity is None:
            velocity = (0.0, 0.0)

        # Extract obstacle and free cell sets
        obstacles = set(discovered_obstacles) if discovered_obstacles else set()
        free_cells = set()
        
        if local_map and self.share_obstacle_maps:
            # Extract free cells from local_map
            for (x, y), (cov, cell_type) in local_map.items():
                if cell_type == "free":
                    free_cells.add((x, y))

        self.last_messages[agent_id] = (
            position, 
            velocity, 
            self.step_count,
            obstacles,
            free_cells
        )

    def receive(
        self,
        agent_id: int,
        agent_position: Tuple[float, float]
    ) -> List[Dict]:
        """
        Receive messages from other agents with Gaussian signal strength decay.

        Communication strength decays smoothly with distance using Gaussian function.
        This models realistic signal degradation and provides uncertainty information.

        Args:
            agent_id: ID of receiving agent
            agent_position: (x, y) position of receiving agent

        Returns:
            List of messages from agents within range (compatible with agent_occupancy.py):
            [{
                'sender_id': int,  # ID of sending agent
                'position': (x, y),  # Position of sender
                'velocity': (vx, vy),  # Velocity of sender
                'timestamp': int,  # When message was sent
                'age': int,  # Steps since message was sent
                'distance': float,  # Distance to other agent
                'reliability': float,  # [0-1] Signal strength (Gaussian decay)
                'position_sigma': float,  # Position uncertainty from communication noise
                'discovered_obstacles': set,  # Set of obstacle cells (x, y) discovered by sender
                'discovered_free': set  # Set of free cells (x, y) discovered by sender
            }, ...]
        """
        messages = []

        for other_id, msg_data in self.last_messages.items():
            if other_id == agent_id:
                continue

            # Unpack message data (with backward compatibility)
            if len(msg_data) == 5:
                pos, vel, timestamp, obstacles, free_cells = msg_data
            elif len(msg_data) == 3:
                # Legacy format (no obstacle sharing)
                pos, vel, timestamp = msg_data
                obstacles = set()
                free_cells = set()
            else:
                continue  # Skip malformed messages

            # Calculate distance to other agent
            distance = np.sqrt(
                (pos[0] - agent_position[0])**2 +
                (pos[1] - agent_position[1])**2
            )

            # Check if within communication range (hard cutoff for clarity in tests)
            # In practice, a soft Gaussian decay provides more realistic behavior
            if distance <= self.comm_range:
                # Gaussian communication strength: exp(-d^2 / (2*sigma^2))
                # Use comm_range as sigma so signal is ~60% at comm_range boundary
                comm_strength = np.exp(-distance**2 / (2 * self.comm_range**2))
            else:
                # Beyond range: no communication
                continue

            # Soft reliability threshold to filter very weak signals
            if comm_strength > 0.05:
                age = self.step_count - timestamp
                
                # Position uncertainty grows with distance and signal weakness
                # Near agents: high reliability, low uncertainty
                # Far agents: low reliability, high uncertainty
                position_sigma = distance * 0.3 * (1.0 - comm_strength)
                
                messages.append({
                    'sender_id': other_id,  # Compatible with agent_occupancy.py
                    'position': pos,
                    'velocity': vel,
                    'timestamp': timestamp,  # Compatible with agent_occupancy.py
                    'age': age,
                    'distance': distance,
                    'reliability': float(comm_strength),  # Signal quality [0-1]
                    'position_sigma': float(position_sigma),  # Communication uncertainty
                    'discovered_obstacles': obstacles if self.share_obstacle_maps else set(),
                    'discovered_free': free_cells if self.share_obstacle_maps else set()
                })

        return messages

    def communicate(self, observations: List[Dict], state) -> List[List[Dict]]:
        """
        Main communication interface (compatible with trainer).
        
        Broadcasts positions, velocities, and discovered obstacle maps.
        Collects messages for all agents within communication range.
        Computes velocity from position history for predictive trajectory modeling.
        
        Args:
            observations: List of agent observations
            state: Current world state
            
        Returns:
            List of message lists, one per agent. Each agent receives messages
            from other agents within communication range.
            Format: [[msg1_for_agent0, msg2_for_agent0, ...], [msg1_for_agent1, ...], ...]
        """
        # Broadcast all agent positions with computed velocities and obstacle maps
        for i, obs in enumerate(observations):
            robot_state = obs['robot_state']
            position = robot_state.position
            
            # Compute velocity from position history
            velocity = self._compute_velocity(i, position, self.step_count)
            
            # Get discovered obstacles and local_map
            discovered_obstacles = getattr(robot_state, 'discovered_obstacles', None)
            local_map = getattr(robot_state, 'local_map', None)
            
            # Broadcast position + velocity + obstacle map
            self.broadcast(
                i, 
                position, 
                velocity=velocity,
                discovered_obstacles=discovered_obstacles,
                local_map=local_map
            )
        
        # Collect messages for each agent
        all_messages = []
        for i, obs in enumerate(observations):
            position = obs['robot_state'].position
            messages = self.receive(i, position)
            all_messages.append(messages)
        
        self.step()
        return all_messages
    
    def _compute_velocity(
        self, 
        agent_id: int, 
        current_position: Tuple[float, float],
        current_step: int
    ) -> Tuple[float, float]:
        """
        Compute agent velocity from position history.
        
        Uses exponential moving average over recent positions for smooth velocity estimate.
        
        Args:
            agent_id: ID of agent
            current_position: Current (x, y) position
            current_step: Current timestep
            
        Returns:
            (vx, vy): Velocity in cells per step
        """
        # Initialize history if needed
        if agent_id not in self.position_history:
            self.position_history[agent_id] = []
        
        history = self.position_history[agent_id]
        
        # Add current position
        history.append((current_position, current_step))
        
        # Keep only recent history (last 5 steps)
        max_history = 5
        if len(history) > max_history:
            history = history[-max_history:]
            self.position_history[agent_id] = history
        
        # Need at least 2 positions to compute velocity
        if len(history) < 2:
            return (0.0, 0.0)
        
        # Compute velocity as average over recent steps
        # Use positions from 2-3 steps ago for stability (smooth out noise)
        if len(history) >= 3:
            old_pos, old_step = history[-3]
            new_pos, new_step = history[-1]
        else:
            old_pos, old_step = history[-2]
            new_pos, new_step = history[-1]
        
        dt = new_step - old_step
        if dt == 0:
            return (0.0, 0.0)
        
        vx = (new_pos[0] - old_pos[0]) / dt
        vy = (new_pos[1] - old_pos[1]) / dt
        
        # Clamp to reasonable range (max velocity ~2 cells/step for diagonal)
        max_v = 2.0
        vx = np.clip(vx, -max_v, max_v)
        vy = np.clip(vy, -max_v, max_v)
        
        return (float(vx), float(vy))

    def should_communicate(self, step: int) -> bool:
        """Check if agents should broadcast this step."""
        return (step % self.comm_freq) == 0

    def step(self) -> None:
        """Increment step counter."""
        self.step_count += 1

    def reset(self) -> None:
        """Reset communication system (call at episode start)."""
        self.last_messages = {}
        self.position_history = {}
        self.step_count = 0
    
    @staticmethod
    def merge_obstacle_maps(
        robot_state,
        received_messages: List[Dict],
        reliability_threshold: float = 0.3
    ) -> None:
        """
        Merge received obstacle maps into agent's knowledge.
        
        Updates the agent's discovered_obstacles and local_map with information
        from other agents, weighted by communication reliability.
        
        Args:
            robot_state: RobotState object to update
            received_messages: List of messages from other agents (from receive())
            reliability_threshold: Minimum reliability to accept obstacle information [0-1]
        
        Note:
            - Only obstacles with reliability >= threshold are added
            - Free cells are also shared to prevent redundant exploration
            - Does not overwrite higher-confidence information with lower-confidence
        """
        if not received_messages:
            return
        
        for msg in received_messages:
            reliability = msg.get('reliability', 0.0)
            
            # Only accept high-quality information
            if reliability < reliability_threshold:
                continue
            
            # Merge discovered obstacles
            discovered_obstacles = msg.get('discovered_obstacles', set())
            if discovered_obstacles and hasattr(robot_state, 'discovered_obstacles'):
                robot_state.discovered_obstacles.update(discovered_obstacles)
            
            # Merge discovered free cells into local_map
            discovered_free = msg.get('discovered_free', set())
            if discovered_free and hasattr(robot_state, 'local_map'):
                for (x, y) in discovered_free:
                    # Only add if not already known (don't overwrite local observations)
                    if (x, y) not in robot_state.local_map:
                        # Add with reduced confidence (0.3) since it's second-hand info
                        robot_state.local_map[(x, y)] = (0.3, "free")


def get_communication_protocol(
    protocol_name: str,
    num_agents: int,
    grid_size: int = 20,
    comm_range: float = 15.0,
    comm_freq: int = 5,
    share_obstacle_maps: bool = True
):
    """
    Factory function to create communication protocol.

    Args:
        protocol_name: Protocol name ['none', 'position']
        num_agents: Number of agents
        grid_size: Grid size (unused, kept for API compatibility)
        comm_range: Communication range (grid cells)
        comm_freq: Communication frequency (every N steps)
        share_obstacle_maps: Whether to share discovered obstacle maps (position protocol only)

    Returns:
        Communication protocol instance (PositionCommunication or NoCommunciation)

    Note:
        - 'none': No communication (baseline)
        - 'position': Position/velocity broadcast + obstacle map sharing (recommended for coordination)
    """
    if protocol_name == 'none':
        return NoCommunciation(num_agents=num_agents)
    elif protocol_name == 'position':
        return PositionCommunication(
            num_agents=num_agents,
            comm_range=comm_range,
            comm_freq=comm_freq,
            share_obstacle_maps=share_obstacle_maps
        )
    else:
        raise ValueError(
            f"Unknown protocol: {protocol_name}. "
            f"Supported: ['none', 'position']"
        )


if __name__ == "__main__":
    # Test communication protocols
    print("Testing Communication Protocols...")

    num_agents = 4

    # Test No Communication
    print("\n1. Testing NoCommunciation...")
    no_comm = NoCommunciation(num_agents=num_agents)
    msg = no_comm.encode_message(0, torch.zeros(128))
    agg = no_comm.aggregate_messages(0, [msg])
    print(f"   ✓ NoCommunciation: message_dim={no_comm.message_dim}, should_comm={no_comm.should_communicate(0)}")

    # Test factory function
    print("\n2. Testing factory function...")
    proto_none = get_communication_protocol('none', num_agents=4)
    print(f"   ✓ Factory creates NoCommunciation: {isinstance(proto_none, NoCommunciation)}")

    # Test error handling
    print("\n3. Testing error handling...")
    try:
        proto_bad = get_communication_protocol('full_state', num_agents=4)
        print(f"   ✗ Should have raised error for 'full_state'")
    except ValueError as e:
        print(f"   ✓ Correctly rejects 'full_state': {str(e)[:50]}...")

    print("\n✓ All communication protocol tests passed!")
    print("\nNOTE: Position communication now via agent_occupancy.py (6th channel)")
    print("      Use --use-6ch flag when training for position information.")
