#!/usr/bin/env python3
import numpy as np
import os
import time
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

class DataRecorder:
    def __init__(self, save_dir=None, save_interval=10.0):
        """
        Initialize the data recorder
        
        Args:
            save_dir: Directory to save data (default: create timestamped directory)
            save_interval: How often to save data to disk (seconds)
        """
        # Create data directory if not specified
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"mppi_data_{timestamp}"
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Timing
        self.save_interval = save_interval
        self.last_save_time = time.time()
        
        # Data storage
        self.vehicle_data = []
        self.control_data = []
        self.trajectory_data = []
        self.mppi_metrics = []
        self.obstacle_data = []
        self.timing_data = []
        
        # File paths
        self.vehicle_file = self.save_dir / "vehicle_data.csv"
        self.control_file = self.save_dir / "control_data.csv"
        self.trajectory_file = self.save_dir / "trajectory_metrics.csv"
        self.mppi_file = self.save_dir / "mppi_metrics.csv"
        self.obstacle_file = self.save_dir / "obstacle_data.csv"
        self.timing_file = self.save_dir / "timing_data.csv"
        
        # Create CSV headers
        self._create_csv_files()
        
        print(f"Data recorder initialized. Saving to {self.save_dir}")
    
    def _create_csv_files(self):
        """Create CSV files with headers"""
        # Vehicle data: timestamp, x, y, speed, theta, omega, beta
        with open(self.vehicle_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'x', 'y', 'speed', 'theta', 'omega', 'beta'])
            
        # Control data: timestamp, steering, target_speed, actual_speed
        with open(self.control_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'steering', 'target_speed', 'actual_speed'])
            
        # Trajectory metrics: timestamp, cross_track_error, tracking_cost, obstacle_cost, total_cost
        with open(self.trajectory_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'cross_track_error', 'tracking_cost', 'obstacle_cost', 'total_cost'])
            
        # MPPI metrics: timestamp, max_weight, min_weight, mean_weight, weight_entropy
        with open(self.mppi_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'max_weight', 'min_weight', 'mean_weight', 'weight_entropy'])
            
        # Obstacle data: timestamp, free_points, unknown_points, obstacle_points, outside_points
        with open(self.obstacle_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'free_points', 'unknown_points', 'obstacle_points', 'outside_points'])
            
        # Timing data: timestamp, computation_time
        with open(self.timing_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'computation_time'])
    
    def record_vehicle_state(self, timestamp, state):
        """Record vehicle state data"""
        # state = [x, y, steering, speed, theta, omega, beta]
        self.vehicle_data.append([timestamp, state[0], state[1], state[3], state[4], state[5], state[6]])
    
    def record_control(self, timestamp, steering, target_speed, actual_speed):
        """Record control commands"""
        self.control_data.append([timestamp, steering, target_speed, actual_speed])
    
    def record_trajectory_metrics(self, timestamp, cross_track_error, tracking_cost, obstacle_cost, total_cost=None):
        """Record trajectory metrics"""
        if total_cost is None:
            total_cost = tracking_cost + obstacle_cost
        self.trajectory_data.append([timestamp, cross_track_error, tracking_cost, obstacle_cost, total_cost])
    
    def record_mppi_metrics(self, timestamp, weights):
        """Record MPPI algorithm metrics"""
        max_weight = np.max(weights)
        min_weight = np.min(weights)
        mean_weight = np.mean(weights)
        
        # Calculate weight entropy as a measure of diversity
        weights_normalized = weights / np.sum(weights)
        entropy = -np.sum(weights_normalized * np.log(weights_normalized + 1e-10))
        
        self.mppi_metrics.append([timestamp, max_weight, min_weight, mean_weight, entropy])
    
    def record_obstacle_data(self, timestamp, free_points, unknown_points, obstacle_points, outside_points):
        """Record obstacle detection data"""
        self.obstacle_data.append([timestamp, free_points, unknown_points, obstacle_points, outside_points])
    
    def record_computation_time(self, timestamp, computation_time):
        """Record computational performance"""
        self.timing_data.append([timestamp, computation_time])
    
    def check_and_save(self):
        """Check if it's time to save data and save if needed"""
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            self.save_data()
            self.last_save_time = current_time
            return True
        return False
    
    def save_data(self):
        """Save all data to disk"""
        if len(self.vehicle_data) > 0:
            with open(self.vehicle_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(self.vehicle_data)
            self.vehicle_data = []
            
        if len(self.control_data) > 0:
            with open(self.control_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(self.control_data)
            self.control_data = []
            
        if len(self.trajectory_data) > 0:
            with open(self.trajectory_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(self.trajectory_data)
            self.trajectory_data = []
            
        if len(self.mppi_metrics) > 0:
            with open(self.mppi_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(self.mppi_metrics)
            self.mppi_metrics = []
            
        if len(self.obstacle_data) > 0:
            with open(self.obstacle_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(self.obstacle_data)
            self.obstacle_data = []
            
        if len(self.timing_data) > 0:
            with open(self.timing_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(self.timing_data)
            self.timing_data = []
            
        print(f"Data saved to {self.save_dir}")
        
    def generate_plots(self):
        """Generate basic plots from the saved data"""
        # This could be expanded to create more sophisticated visualizations
        try:
            # Load saved data
            vehicle_df = np.genfromtxt(self.vehicle_file, delimiter=',', skip_header=1, 
                                      names=['timestamp', 'x', 'y', 'speed', 'theta', 'omega', 'beta'])
            traj_df = np.genfromtxt(self.trajectory_file, delimiter=',', skip_header=1, 
                                   names=['timestamp', 'cross_track_error', 'tracking_cost', 'obstacle_cost', 'total_cost'])
            
            # Create plots directory
            plots_dir = self.save_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Position plot
            plt.figure(figsize=(10, 8))
            plt.plot(vehicle_df['x'], vehicle_df['y'])
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.title('Vehicle Path')
            plt.grid(True)
            plt.savefig(plots_dir / "vehicle_path.png")
            plt.close()
            
            # Speed plot
            plt.figure(figsize=(12, 6))
            plt.plot(vehicle_df['timestamp'] - vehicle_df['timestamp'][0], vehicle_df['speed'])
            plt.xlabel('Time (s)')
            plt.ylabel('Speed (m/s)')
            plt.title('Vehicle Speed')
            plt.grid(True)
            plt.savefig(plots_dir / "speed_profile.png")
            plt.close()
            
            # Costs plot
            plt.figure(figsize=(12, 6))
            plt.plot(traj_df['timestamp'] - traj_df['timestamp'][0], traj_df['tracking_cost'], label='Tracking Cost')
            plt.plot(traj_df['timestamp'] - traj_df['timestamp'][0], traj_df['obstacle_cost'], label='Obstacle Cost')
            plt.plot(traj_df['timestamp'] - traj_df['timestamp'][0], traj_df['total_cost'], label='Total Cost')
            plt.xlabel('Time (s)')
            plt.ylabel('Cost')
            plt.title('Trajectory Costs')
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / "costs.png")
            plt.close()
            
            print(f"Plots generated and saved to {plots_dir}")
            
        except Exception as e:
            print(f"Error generating plots: {e}")