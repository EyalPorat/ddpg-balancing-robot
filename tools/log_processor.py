import sys
import os
import socket
import struct
import numpy as np
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Generator, Tuple
import pandas as pd
from pathlib import Path
import logging
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from python.src.balancing_robot.models import SimNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoggedState:
    """Represents a single logged state from the robot."""

    timestamp: int  # milliseconds since boot
    dt: float  # control loop period in seconds

    # State variables
    theta: float  # body angle (rad)
    theta_dot: float  # angular velocity (rad/s)

    # Control outputs
    model_output: float  # raw model output
    motor_pwm: int  # applied PWM value

    # System status
    standing: bool  # whether robot is in standing mode
    model_active: bool  # whether DDPG model is active
    battery_voltage: float  # battery voltage

    # Additional metrics
    acc_x: float  # X acceleration (g)
    acc_z: float  # Z acceleration (g)
    gyro_x: float  # X gyro rate (deg/s)


@dataclass
class Episode:
    """Collection of logged states forming an episode."""

    states: List[LoggedState]
    metadata: Dict

    @property
    def duration(self) -> float:
        """Episode duration in seconds."""
        return sum(state.dt for state in self.states)

    @property
    def state_array(self) -> np.ndarray:
        """Convert states to numpy array format for training."""
        return np.array(
            [
                [
                    s.theta,  # Angle
                    s.theta_dot,  # Angular velocity
                    s.motor_pwm / 127.0,  # Previous motor command (normalized to [-1, 1])
                ]
                for s in self.states
            ]
        )

    @property
    def action_array(self) -> np.ndarray:
        """Convert actions to numpy array format for training."""
        return np.array([[s.motor_pwm / 127.0] for s in self.states])


class LogProcessor:
    """Processes telemetry data from the balancing robot."""

    def __init__(self, port: int = 44444, config_path: Optional[str] = None):
        """Initialize processor with network settings and optional config."""
        # Create UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)  # Allow interrupt between packets
        self.sock.bind(("0.0.0.0", port))

        # Data format matching C++ struct (updated for simplified state)
        self.format = "<I f f f f b ?? f f f f"
        self.episodes: List[Episode] = []

        # Load config if provided
        self.config = None
        if config_path:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

        # Statistics tracking
        self.packet_count = 0
        self.last_print_time = time.time()

    def process_packet(self, data: bytes) -> LoggedState:
        """Unpack incoming bytes into LoggedState."""
        unpacked = struct.unpack(self.format, data)
        return LoggedState(
            timestamp=unpacked[0],
            dt=unpacked[1],
            theta=unpacked[2],
            theta_dot=unpacked[3],
            model_output=unpacked[4],
            motor_pwm=unpacked[5],
            standing=unpacked[6],
            model_active=unpacked[7],
            battery_voltage=unpacked[8],
            acc_x=unpacked[9],
            acc_z=unpacked[10],
            gyro_x=unpacked[11],
        )

    def collect_episodes(
        self, duration_seconds: int = 1500, min_episode_length: int = 10
    ) -> Generator[Episode, None, None]:
        """Collect episodes for specified duration."""
        logger.info(f"Collecting data for {duration_seconds} seconds...")
        end_time = time.time() + duration_seconds

        current_episode: List[LoggedState] = []

        try:
            while time.time() < end_time:
                try:
                    data, _ = self.sock.recvfrom(1024)
                except socket.timeout:
                    continue

                self.packet_count += 1

                if len(data) < struct.calcsize(self.format):
                    continue

                state = self.process_packet(data)

                # New episode starts when standing transitions from False to True
                if not current_episode and state.standing:
                    current_episode = []

                if current_episode is not None:
                    current_episode.append(state)

                    # Episode ends when standing becomes False
                    if not state.standing:
                        if len(current_episode) > min_episode_length:
                            episode = Episode(
                                states=current_episode, metadata=self._compute_episode_metadata(current_episode)
                            )
                            self.episodes.append(episode)
                            yield episode

                        current_episode = None

                # Print packet rate every second
                now = time.time()
                if now - self.last_print_time >= 1.0:
                    rate = self.packet_count / (now - self.last_print_time)
                    logger.info(f"Packet rate: {rate:.1f} packets/sec")
                    self.packet_count = 0
                    self.last_print_time = now

        except KeyboardInterrupt:
            logger.info("\nData collection stopped by user")

        logger.info(f"Collected {len(self.episodes)} episodes")

    def _compute_episode_metadata(self, states: List[LoggedState]) -> Dict:
        """Compute metadata for episode."""
        return {
            "duration": sum(s.dt for s in states),
            "length": len(states),
            "max_angle": max(abs(s.theta) for s in states),
            "avg_angle": np.mean([abs(s.theta) for s in states]),
            "battery_voltage": np.mean([s.battery_voltage for s in states]),
            "model_active": all(s.model_active for s in states),
        }

    def save_episodes(self, filename: str):
        """Save collected episodes to file."""
        data = {
            "episodes": [
                {"states": [vars(state) for state in episode.states], "metadata": episode.metadata}
                for episode in self.episodes
            ]
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.episodes)} episodes to {filename}")

    def prepare_training_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Prepare collected data for SimNet training."""
        states = []
        actions = []
        next_states = []

        for episode in self.episodes:
            episode_states = episode.state_array[:-1]  # All but last state
            episode_actions = episode.action_array[:-1]
            episode_next_states = episode.state_array[1:]  # All but first state

            states.append(episode_states)
            actions.append(episode_actions)
            next_states.append(episode_next_states)

        # Combine all episodes
        states = np.concatenate(states)
        actions = np.concatenate(actions)
        next_states = np.concatenate(next_states)

        # Split into train/validation
        split = int(0.9 * len(states))
        train_data = {"states": states[:split], "actions": actions[:split], "next_states": next_states[:split]}
        val_data = {"states": states[split:], "actions": actions[split:], "next_states": next_states[split:]}

        return train_data, val_data


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process robot telemetry data")
    parser.add_argument("--port", type=int, default=44444, help="UDP port for receiving data")
    parser.add_argument("--duration", type=int, default=1500, help="Duration to collect data (seconds)")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output", type=str, default=None, help="Output file for collected data")

    args = parser.parse_args()

    processor = LogProcessor(args.port, args.config)

    try:
        for episode in processor.collect_episodes(args.duration):
            logger.info(
                f"Collected episode: length={episode.metadata['length']}, "
                f"max_angle={episode.metadata['max_angle']:.2f}, "
                f"battery={episode.metadata['battery_voltage']:.2f}V"
            )
    except KeyboardInterrupt:
        logger.info("\nData collection interrupted")

    if args.output or processor.episodes:
        output_file = args.output or f"robot_logs_{time.strftime('%Y%m%d_%H%M%S')}.json"
        processor.save_episodes(output_file)


if __name__ == "__main__":
    main()
