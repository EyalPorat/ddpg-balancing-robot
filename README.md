# Balancing Robot Project

A complete self-balancing robot implementation combining reinforcement learning, physics simulation, and real-world deployment. This project includes both the embedded control system and the Python training framework.

## Project Overview

This project implements a two-wheeled self-balancing robot using:
- Deep Deterministic Policy Gradient (DDPG) for control
- Neural network-based dynamics simulation (SimNet)
- ESP32 (M5StickC Plus) for real-time control
- Web-based monitoring and configuration interface

## Repository Structure

```
balancing_robot/
├── docs/                      # Project documentation
├── embedded/                  # ESP32 firmware
│   ├── include/              # C++ headers
│   │   ├── ddpg_controller.h # DDPG inference
│   │   ├── ddpg_network.h    # Neural network implementation
│   │   └── model_receiver.h  # Model loading via UDP
│   └── src/                  # Implementation files
├── python/                   # Python training framework
│   └── README.md            # Python package documentation
└── tools/                   # Utility scripts
```

## Getting Started

### Hardware Requirements

1. M5StickC Plus (ESP32-based controller)
2. Custom robot chassis
   - 2x DC motors with encoders
   - Motor driver
   - Battery pack
   - Wheels and frame

### Software Requirements

1. PlatformIO IDE for embedded development
2. Python 3.8+ for training framework
3. Modern web browser for monitoring interface

### Initial Setup

1. Set up the embedded system:
```bash
cd embedded
pio init
pio lib install
```

2. Set up the Python environment:
```bash
cd python
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
pip install -e .
```

## Development Workflow

1. Train the model:
   - Use Jupyter notebooks in `python/notebooks/` (`train_ddpg.ipynb` and `train_simnet.ipynb`)
   - Models are saved to `python/notebooks/logs/`

2. Deploy to robot:
   - Configure network settings in `embedded/include/config.h`
   - Flash firmware to ESP32
   - Use `tools/model_deployer.py` to send trained model

3. Monitor performance:
   - Connect to robot's WiFi access point
   - Open web interface
   - View real-time telemetry

## Modules

### Embedded System
- Real-time control loop
- DDPG model inference
- Sensor fusion and state estimation
- WiFi connectivity and web server
- UDP-based model loading and telemetry

### Python Framework
- DDPG implementation
- SimNet for dynamics simulation
- Training environments and utilities
- Visualization and analysis tools
- Model deployment utilities

## License

MIT License - see [LICENSE](LICENSE) for details
