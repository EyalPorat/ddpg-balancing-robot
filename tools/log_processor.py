import socket
import struct
import numpy as np
import json
import time
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd

@dataclass
class Episode:
    states: np.ndarray  # [theta, theta_dot, x, x_dot, phi, phi_dot]
    actions: np.ndarray  # [motor_pwm]
    timestamps: np.ndarray
    metadata: Dict

class LogProcessor:
    def __init__(self, port=44444):
        # Create UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Set a timeout (so we can catch Ctrl+C in between packets)
        self.sock.settimeout(1.0)
        self.sock.bind(('0.0.0.0', port))
        
        # Match your C++ struct (55 bytes, packed):
        #  1) uint32_t timestamp
        #  2) float dt
        #  3) float theta
        #  4) float theta_dot
        #  5) float x
        #  6) float x_dot
        #  7) float phi
        #  8) float phi_dot
        #  9) float model_output
        # 10) int8_t motor_pwm
        # 11) bool standing
        # 12) bool model_active
        # 13) float battery_voltage
        # 14) float acc_x
        # 15) float acc_z
        # 16) float gyro_x
        self.format = '<I f f f f f f f f b ?? f f f f'
        
        self.current_episode = []
        self.episodes = []
        
        # Variables for tracking packet rate:
        self.last_print_time = time.time()
        self.packet_count = 0
        
    def process_packet(self, data: bytes) -> Dict:
        """
        Unpack incoming bytes into a dictionary consistent with our C++ struct.
        """
        unpacked = struct.unpack(self.format, data)
        return {
            'timestamp':       unpacked[0],
            'dt':              unpacked[1],
            'theta':           unpacked[2],
            'theta_dot':       unpacked[3],
            'x':               unpacked[4],
            'x_dot':           unpacked[5],
            'phi':             unpacked[6],
            'phi_dot':         unpacked[7],
            'model_output':    unpacked[8],
            'motor_pwm':       unpacked[9],
            'standing':        unpacked[10],
            'model_active':    unpacked[11],
            'battery_voltage': unpacked[12],
            'acc_x':           unpacked[13],
            'acc_z':           unpacked[14],
            'gyro_x':          unpacked[15],
        }
    
    def collect_episodes(self, duration_seconds: int = 1500):
        """
        Collect episodes for a specified duration or until Ctrl+C is pressed.
        An 'episode' starts when 'standing' transitions from False to True,
        and ends when 'standing' becomes False again.
        """
        print(f"Collecting data for {duration_seconds} seconds...")
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            try:
                data, addr = self.sock.recvfrom(1024)
            except socket.timeout:
                # No data arrived within timeout; check for Ctrl+C again
                pass
            except KeyboardInterrupt:
                # Allow user to stop data collection with Ctrl+C
                print("\nCtrl+C detected. Stopping data collection...")
                break
            else:
                # If we successfully received data, increment packet_count
                self.packet_count += 1
                
                # If the packet is too short or doesn't match our struct, ignore it
                if len(data) < struct.calcsize(self.format):
                    continue
                
                log_entry = self.process_packet(data)
                
                # New episode starts when 'standing' transitions from False to True
                if not self.current_episode and log_entry['standing']:
                    self.current_episode = []
                
                # If we are currently collecting an episode
                if self.current_episode is not None:
                    self.current_episode.append(log_entry)
                    
                    # Episode ends when 'standing' becomes False
                    if not log_entry['standing']:
                        # Filter out episodes of length 1
                        if len(self.current_episode) > 1:
                            episode = self._create_episode(self.current_episode)
                            self.episodes.append(episode)
                            print(f"New episode collected: length={len(episode.states)}, "
                                    f"max_angle={episode.metadata['max_angle']:.2f}, "
                                    f"avg_battery_voltage={episode.metadata['battery_voltage']:.2f}, "
                                    f"model_active={episode.metadata['model_active']}")
                            print(f"Received episode length is {len(self.current_episode)}")
                        self.current_episode = None

            # Check if >1 second has passed; if so, print packet rate
            now = time.time()
            if now - self.last_print_time >= 1.0:
                # Compute packets per second over the last interval
                elapsed = now - self.last_print_time
                rate = self.packet_count / elapsed
                print(f"Packet rate: {rate:.1f} packets/sec")
                
                # Reset counters
                self.packet_count = 0
                self.last_print_time = now
        
        print(f"Data collection complete. Collected {len(self.episodes)} episodes.")
    
    def _create_episode(self, log_entries: List[Dict]) -> Episode:
        """
        Convert a list of log dictionaries into an Episode object for easier post-processing.
        """
        df = pd.DataFrame(log_entries)
        
        # We'll store these states: [theta, theta_dot, x, x_dot, phi, phi_dot]
        states = df[['theta', 'theta_dot', 'x', 'x_dot', 'phi', 'phi_dot']].values
        
        # We store a single action = [motor_pwm], normalized to [-1, 1]
        actions = df[['motor_pwm']].values / 127.0
        timestamps = df['timestamp'].values
        
        # Cast .all() to a Python bool to avoid NumPy bool_ JSON errors
        metadata = {
            'battery_voltage': df['battery_voltage'].mean(),
            'duration': len(df),
            'max_angle': df['theta'].abs().max(),
            'model_active': bool(df['model_active'].all())
        }
        
        return Episode(states, actions, timestamps, metadata)
    
    def save_episodes(self, filename: str):
        """
        Save the collected episodes as JSON for offline analysis.
        """
        data = {
            'episodes': [
                {
                    'states': episode.states.tolist(),
                    'actions': episode.actions.tolist(),
                    'timestamps': episode.timestamps.tolist(),
                    'metadata': episode.metadata
                }
                for episode in self.episodes
            ]
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Saved {len(self.episodes)} episodes to {filename}")

def calculate_rewards(episode: Episode) -> np.ndarray:
    """
    Example reward function that penalizes large angles, velocity, and motor usage.
    Adjust as desired for your RL problem.
    """
    states = episode.states
    actions = episode.actions
    
    # For instance:
    # Reward for staying near zero angle
    angle_reward = np.exp(-5.0 * np.square(states[:, 0]))  # theta
    
    # Small penalty for angular velocity
    stability_reward = -0.1 * np.square(states[:, 1])      # theta_dot
    
    # Reward for staying near x=0
    position_reward = np.exp(-2.0 * np.square(states[:, 2])) # x
    
    # Penalty for large motor commands
    power_penalty = -0.05 * np.square(actions[:, 0])
    
    return angle_reward + stability_reward + position_reward + power_penalty

if __name__ == "__main__":
    processor = LogProcessor()
    try:
        print("Starting data collection. Press Ctrl+C to stop and save...")
        processor.collect_episodes(duration_seconds=1500)  # 25 minutes
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Saving collected episodes...")
    finally:
        if processor.episodes:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"robot_logs_{timestamp}.json"
            processor.save_episodes(filename)
            print(f"Saved {len(processor.episodes)} episodes to {filename}")
        else:
            print("No episodes collected")
