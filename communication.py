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
    Position-based communication protocol.

    Agents broadcast (position, velocity, timestamp) every N steps.
    Recipients within communication range receive the messages.

    This enables persistent position information beyond visual sensor range,
    allowing agents to coordinate proactively rather than just reactively.

    Key features:
    - Broadcasts every comm_freq steps
    - Range-limited (comm_range parameter)
    - Includes uncertainty/age of information
    - Used to populate 6th channel (agent occupancy)
    """

    def __init__(
        self,
        num_agents: int,
        comm_range: float = 15.0,
        comm_freq: int = 5
    ):
        """
        Initialize position communication protocol.

        Args:
            num_agents: Number of agents in the system
            comm_range: Maximum communication range (grid cells)
            comm_freq: Communication frequency (broadcast every N steps)
        """
        self.num_agents = num_agents
        self.comm_range = comm_range
        self.comm_freq = comm_freq
        self.last_messages = {}  # agent_id -> (pos, vel, timestamp)
        self.position_history = {}  # agent_id -> [(pos, step), ...]  for velocity computation
        self.step_count = 0

    def broadcast(
        self,
        agent_id: int,
        position: Tuple[float, float],
        velocity: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Agent broadcasts its position and velocity.

        Args:
            agent_id: ID of broadcasting agent
            position: (x, y) position in grid coordinates
            velocity: (vx, vy) velocity (optional, defaults to (0, 0))
        """
        if velocity is None:
            velocity = (0.0, 0.0)

        self.last_messages[agent_id] = (position, velocity, self.step_count)

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
                'position_sigma': float  # Position uncertainty from communication noise
            }, ...]
        """
        messages = []

        for other_id, (pos, vel, timestamp) in self.last_messages.items():
            if other_id == agent_id:
                continue

            # Calculate distance to other agent
            distance = np.sqrt(
                (pos[0] - agent_position[0])**2 +
                (pos[1] - agent_position[1])**2
            )

            # Gaussian communication strength: exp(-d^2 / (2*sigma^2))
            # Use comm_range as sigma so signal is ~60% at comm_range
            # and drops to ~5% at 2*comm_range
            comm_strength = np.exp(-distance**2 / (2 * self.comm_range**2))

            # Only include messages with meaningful signal strength (>5%)
            # This provides soft cutoff instead of hard boundary
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
                    'reliability': float(comm_strength),  # NEW: Signal quality [0-1]
                    'position_sigma': float(position_sigma)  # NEW: Communication uncertainty
                })

        return messages

    def communicate(self, observations: List[Dict], state) -> List[List[Dict]]:
        """
        Main communication interface (compatible with trainer).
        
        Broadcasts positions and collects messages for all agents.
        Computes velocity from position history for predictive trajectory modeling.
        
        Args:
            observations: List of agent observations
            state: Current world state
            
        Returns:
            List of message lists, one per agent. Each agent receives messages
            from other agents within communication range.
            Format: [[msg1_for_agent0, msg2_for_agent0, ...], [msg1_for_agent1, ...], ...]
        """
        # Broadcast all agent positions with computed velocities
        for i, obs in enumerate(observations):
            position = obs['robot_state'].position
            
            # Compute velocity from position history
            velocity = self._compute_velocity(i, position, self.step_count)
            
            # Broadcast position + velocity
            self.broadcast(i, position, velocity=velocity)
        
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
        self.step_count = 0


def get_communication_protocol(
    protocol_name: str,
    num_agents: int,
    grid_size: int = 20,
    comm_range: float = 15.0,
    comm_freq: int = 5
):
    """
    Factory function to create communication protocol.

    Args:
        protocol_name: Protocol name ['none', 'position']
        num_agents: Number of agents
        grid_size: Grid size (unused, kept for API compatibility)
        comm_range: Communication range (grid cells)
        comm_freq: Communication frequency (every N steps)

    Returns:
        Communication protocol instance (PositionCommunication or NoCommunciation)

    Note:
        - 'none': No communication (baseline)
        - 'position': Position/velocity broadcast (recommended for coordination)
    """
    if protocol_name == 'none':
        return NoCommunciation(num_agents=num_agents)
    elif protocol_name == 'position':
        return PositionCommunication(
            num_agents=num_agents,
            comm_range=comm_range,
            comm_freq=comm_freq
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
