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

from src.balancing_robot.models import Actor

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
            # Export layer 1
            weights = network.l1.weight.data.cpu().numpy()
            bias = network.l1.bias.data.cpu().numpy()

            # Write L1 shape
            f.write(struct.pack("II", 8, 6))  # Fixed shape for the robot
            f.write(weights.astype("float32").tobytes())
            f.write(bias.astype("float32").tobytes())

            # Write L1 LayerNorm parameters
            f.write(network.ln1.weight.data.cpu().numpy().astype("float32").tobytes())
            f.write(network.ln1.bias.data.cpu().numpy().astype("float32").tobytes())

            # Export layer 2
            weights = network.l2.weight.data.cpu().numpy()
            bias = network.l2.bias.data.cpu().numpy()

            # Write L2 shape
            f.write(struct.pack("II", 8, 8))
            f.write(weights.astype("float32").tobytes())
            f.write(bias.astype("float32").tobytes())
            f.write(network.ln2.weight.data.cpu().numpy().astype("float32").tobytes())
            f.write(network.ln2.bias.data.cpu().numpy().astype("float32").tobytes())

            # Export layer 3
            weights = network.l3.weight.data.cpu().numpy()
            bias = network.l3.bias.data.cpu().numpy()

            # Write L3 shape
            f.write(struct.pack("II", 1, 8))
            f.write(weights.astype("float32").tobytes())
            f.write(bias.astype("float32").tobytes())

    def verify_weights_file(self, filename: str) -> bool:
        """Verify exported weights file structure."""
        try:
            with open(filename, "rb") as f:
                # Verify L1
                rows, cols = struct.unpack("II", f.read(8))
                logger.info(f"L1 shape: {rows}x{cols}")
                if rows != 8 or cols != 6:
                    raise ValueError(f"Invalid L1 shape: {rows}x{cols}")

                # Skip weights and biases
                f.seek(rows * cols * 4 + rows * 4, 1)  # float32 = 4 bytes
                # Skip LayerNorm
                f.seek(rows * 4 * 2, 1)

                # Verify L2
                rows, cols = struct.unpack("II", f.read(8))
                logger.info(f"L2 shape: {rows}x{cols}")
                if rows != 8 or cols != 8:
                    raise ValueError(f"Invalid L2 shape: {rows}x{cols}")

                # Skip L2 data
                f.seek(rows * cols * 4 + rows * 4, 1)
                f.seek(rows * 4 * 2, 1)  # LayerNorm

                # Verify L3
                rows, cols = struct.unpack("II", f.read(8))
                logger.info(f"L3 shape: {rows}x{cols}")
                if rows != 1 or cols != 8:
                    raise ValueError(f"Invalid L3 shape: {rows}x{cols}")

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


def main():
    parser = argparse.ArgumentParser(description="Deploy model to robot")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to model config")
    parser.add_argument("--ip", type=str, default="192.168.1.255", help="Robot IP address")
    parser.add_argument("--port", type=int, default=44445, help="Robot UDP port")

    args = parser.parse_args()

    # Load model and config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    actor = Actor(state_dim=6, action_dim=1, max_action=config["physics"]["max_torque"])  # Fixed for our robot

    checkpoint = torch.load(args.model)
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
