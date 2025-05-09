import sys
import os
import socket
import struct
import torch
import numpy as np
import argparse
from pathlib import Path
import yaml
import time
import logging
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from python.src.balancing_robot.models import Actor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDeployer:
    """Handles deployment of trained models to the robot."""

    def __init__(self, udp_ip: str = "192.168.1.255", udp_port: int = 44445):
        """Initialize deployer with network settings."""
        self.udp_ip = udp_ip
        self.udp_port = udp_port

        # Create UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def export_network_weights(self, network: torch.nn.Module, filename: str):
        """Export network weights to binary format for embedded system."""
        with open(filename, "wb") as f:
            # Export first hidden layer (L1)
            first_layer = network.network[0]
            weights = first_layer.weight.data.cpu().numpy()
            bias = first_layer.bias.data.cpu().numpy()

            # Write L1 shape
            f.write(struct.pack("II", weights.shape[0], weights.shape[1]))  # 10x9 for the first layer (enhanced state)
            f.write(weights.astype("float32").tobytes())
            f.write(bias.astype("float32").tobytes())

            # Write L1 LayerNorm parameters
            layer_norm1 = network.network[1]
            f.write(layer_norm1.weight.data.cpu().numpy().astype("float32").tobytes())  # gamma
            f.write(layer_norm1.bias.data.cpu().numpy().astype("float32").tobytes())  # beta

            # Export second hidden layer (L2)
            second_layer = network.network[3]  # After ReLU
            weights = second_layer.weight.data.cpu().numpy()
            bias = second_layer.bias.data.cpu().numpy()

            # Write L2 shape
            f.write(struct.pack("II", weights.shape[0], weights.shape[1]))  # 10x10 for the second layer
            f.write(weights.astype("float32").tobytes())
            f.write(bias.astype("float32").tobytes())

            # Write L2 LayerNorm parameters
            layer_norm2 = network.network[4]
            f.write(layer_norm2.weight.data.cpu().numpy().astype("float32").tobytes())  # gamma
            f.write(layer_norm2.bias.data.cpu().numpy().astype("float32").tobytes())  # beta

            # Export output layer (L3)
            weights = network.output_layer.weight.data.cpu().numpy()
            bias = network.output_layer.bias.data.cpu().numpy()

            # Write output layer shape
            f.write(struct.pack("II", weights.shape[0], weights.shape[1]))  # 1x10 for output layer
            f.write(weights.astype("float32").tobytes())
            f.write(bias.astype("float32").tobytes())

            logger.info(f"Weight file created with structure: L1(10x9) -> L2(10x10) -> L3(1x10)")

    def verify_weights_file(self, filename: str) -> bool:
        """Verify exported weights file structure."""
        try:
            with open(filename, "rb") as f:
                # Verify L1
                rows, cols = struct.unpack("II", f.read(8))
                logger.info(f"L1 shape: {rows}x{cols}")

                # We expect 10x9 for the first layer
                if cols != 9:
                    raise ValueError(f"Invalid L1 shape: {rows}x{cols}, expected 10x9 (enhanced state)")
                if rows != 10:
                    raise ValueError(f"Invalid L1 shape: {rows}x{cols}, expected 10x9")

                # Skip weights and biases
                f.seek(rows * cols * 4 + rows * 4, 1)  # float32 = 4 bytes
                # Skip LayerNorm
                f.seek(rows * 4 * 2, 1)

                # Verify L2
                rows, cols = struct.unpack("II", f.read(8))
                logger.info(f"L2 shape: {rows}x{cols}")

                # Second layer should be 10x10
                if rows != 10 or cols != 10:
                    raise ValueError(f"Invalid L2 shape: {rows}x{cols}, expected 10x10")

                # Skip weights and biases
                f.seek(rows * cols * 4 + rows * 4, 1)  # float32 = 4 bytes
                # Skip LayerNorm
                f.seek(rows * 4 * 2, 1)

                # Verify L3 (output layer)
                rows, cols = struct.unpack("II", f.read(8))
                logger.info(f"L3 shape: {rows}x{cols}")

                # Output layer should be 1x10
                if rows != 1:
                    raise ValueError(f"Invalid L3 shape: {rows}x{cols}, expected 1x10")
                if cols != 10:
                    raise ValueError(f"Invalid L3 shape: {rows}x{cols}, expected 1x10")

                return True

        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            return False

    def send_weights(self, filename: str, chunk_size: int = 1024):
        """Send weights file over UDP in chunks."""
        try:
            with open(filename, "rb") as f:
                data = f.read()

            # Split into chunks
            chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

            # Send number of chunks first
            self.sock.sendto(struct.pack("I", len(chunks)), (self.udp_ip, self.udp_port))

            # Send each chunk with index
            for i, chunk in enumerate(chunks):
                packet = struct.pack("I", i) + chunk
                self.sock.sendto(packet, (self.udp_ip, self.udp_port))

                if i % 10 == 0:  # Progress update
                    logger.info(f"Sent chunk {i+1}/{len(chunks)}")
                time.sleep(0.001)  # Small delay to prevent overwhelming receiver

            logger.info("Weight transmission complete")
            return True

        except Exception as e:
            logger.error(f"Transmission failed: {str(e)}")
            return False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Deploy model to robot")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to model config")
    parser.add_argument("--ip", type=str, default="192.168.1.255", help="Robot IP address")
    parser.add_argument("--port", type=int, default=44445, help="Robot UDP port")

    args = parser.parse_args()

    # Load config to get state dimensions
    config = load_config(args.config)

    # Determine state dimension from config
    state_dim = 9  # Default enhanced state size

    # Check if we have observation parameters
    if "observation" in config:
        # Basic state (3) + moving averages (2) + action history
        action_history_size = config["observation"].get("action_history_size", 4)
        state_dim = 3 + 2 + action_history_size
        logger.info(f"Using state dimension from config: {state_dim}")
    else:
        logger.info(f"Using default enhanced state dimension: {state_dim}")

    # Create model with enhanced state input
    # We always use max_action=1.0 for the actor, as actions are normalized to [-1, 1]
    # The scaling to actual PWM values happens in the embedded controller
    logger.info(f"Creating model with state_dim={state_dim}, action_dim=1, max_action=1.0")
    actor = Actor(state_dim=state_dim, action_dim=1, max_action=1.0, hidden_dims=(10, 10))

    # Load saved weights
    checkpoint = torch.load(args.model, map_location=torch.device("cpu"))
    actor.load_state_dict(checkpoint["state_dict"])

    # Export and deploy
    deployer = ModelDeployer(args.ip, args.port)

    temp_file = "temp_weights.bin"
    deployer.export_network_weights(actor, temp_file)

    if deployer.verify_weights_file(temp_file):
        if deployer.send_weights(temp_file):
            logger.info("Model deployed successfully")
        else:
            logger.error("Failed to send weights")
    else:
        logger.error("Weight file verification failed")

    # Cleanup
    Path(temp_file).unlink()


if __name__ == "__main__":
    main()
