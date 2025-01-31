# Balancing Robot with DDPG Reinforcement Learning

A self-balancing robotics project utilizing Deep Deterministic Policy Gradient (DDPG) reinforcement learning. The robot learns to maintain balance through physics simulation training and deploys the trained model on real hardware, fine tuning to improve performance.

## Technologies

- Python + PyTorch for DDPG implementation and training
- Gymnasium for simulation environment creation
- ESP32 (M5StickC Plus) for robot hardware
- Two-way UDP socket for model deployment and data logging

## Project Components

- Reinforcement learning training in simulation
- Real-time model deployment via UDP
- Telemetry logging and visualization
- Real-world performance analysis

## Quick Start Guide

### Training

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Open `balancing_robot_ddpg/balancing_robot_ddpg.ipynb` and run all cells up to model deployment and fine-tuning
3. Model checkpoints saved to `checkpoints/` directory

### Deploying to Robot

1. Flash firmware to M5StickC-Plus (ESP32) using PlatformIO
2. Connect to robot's WiFi network
3. Run notebook's model deployment cell to send weights via UDP
4. Robot automatically loads weights and switches to DDPG mode

### Fine Tuning

1. Run log processor:
   ```bash
   python tools/log_processor.py
   ```
2. Play several training "episodes" (from starting to an end contition)
3. Run the fine-tuning cells in notebook
4. Run model deployment cells in notebook
5. Repeat until reaching ideal results

## Notes

- Robot requires stable WiFi connection for weight transfer
- Battery voltage affects motor response - monitor in telemetry
- Initial balancing requires flat surface
- Fine-tuning improves real-world performance
- For video generation in notebook, ffmpeg needs to be installed
