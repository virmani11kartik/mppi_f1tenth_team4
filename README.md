# MPPI Example Code for ROS2 with Jax

## Features

- 🔁 Sampling-based MPPI algorithm
- 🚗 Various dynamics models (`dynamics_models/`)
- 🗺️ Waypoint tools for trajectory tracking
- 📈 Visualization tools for control and trajectories
- 🧩 ROS2 node integration

For installing Jax on Jetson with CUDA, please refer to [here](https://github.com/zzangupenn/jax_ros_docker_jetson) and [here](https://github.com/zzangupenn/jax_for_jetson_orin).

## Installation & Usage

### Quick Start

1. Clone this repository into your ROS2 workspace's `src` directory:
   ```bash
   cd ~/your_workspace/src
   git clone https://github.com/your_username/mppi_f1tenth_team4.git
   ```

2. Build your workspace:
   ```bash
   cd ~/your_workspace
   colcon build
   ```

3. Source the setup script:
   ```bash
   source install/setup.bash
   ```

4. Run the MPPI controller node:
   ```bash
   ros2 run mppi mppi_node.py
   ```

### Notes

- This package includes predefined waypoints in the `waypoints` directory
- The MPPI controller subscribes to odometry data and publishes control commands to the `/drive` topic
- Configuration parameters can be modified in the `config/config.yaml` file
- The controller works best with JAX GPU acceleration (CUDA) for real-time performance
