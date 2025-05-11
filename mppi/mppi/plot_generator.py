#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
import glob
import os
from datetime import datetime

class PlotGenerator:
    def __init__(self, data_dir, output_dir=None):
        """
        Initialize plot generator with data directory
        
        Args:
            data_dir: Directory containing CSV files from DataRecorder
            output_dir: Directory to save generated plots (defaults to data_dir/plots)
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Set output directory
        if output_dir is None:
            self.output_dir = self.data_dir / "plots"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all data files
        self.vehicle_data = self._load_csv(self.data_dir / "vehicle_data.csv")
        self.control_data = self._load_csv(self.data_dir / "control_data.csv")
        self.trajectory_metrics = self._load_csv(self.data_dir / "trajectory_metrics.csv")
        self.mppi_metrics = self._load_csv(self.data_dir / "mppi_metrics.csv")
        self.obstacle_data = self._load_csv(self.data_dir / "obstacle_data.csv")
        self.timing_data = self._load_csv(self.data_dir / "timing_data.csv")
        
        # Set a consistent color palette
        self.colors = {
            'path': '#1f77b4',  # Blue
            'reference': '#ff7f0e',  # Orange
            'speed': '#2ca02c',  # Green
            'steering': '#d62728',  # Red
            'tracking_cost': '#9467bd',  # Purple
            'obstacle_cost': '#8c564b',  # Brown
            'total_cost': '#e377c2',  # Pink
            'free': '#7f7f7f',  # Gray
            'unknown': '#bcbd22',  # Olive
            'obstacle': '#17becf',  # Cyan
            'computation': '#1f77b4'  # Blue
        }
        
        print(f"Plot generator initialized with data from {self.data_dir}")
        
    def _load_csv(self, filepath):
        """Load CSV file into pandas DataFrame"""
        try:
            return pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Warning: File not found - {filepath}")
            return None
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def generate_all_plots(self):
        """Generate all available plots"""
        print("Generating all plots...")
        
        # Generate individual plots
        self.plot_vehicle_path()
        self.plot_speed_profile()
        self.plot_steering_profile()
        self.plot_costs_over_time()
        self.plot_cross_track_error()
        self.plot_weight_distribution()
        self.plot_obstacle_metrics()
        self.plot_computation_time()
        
        # Generate dashboard plots
        self.generate_performance_dashboard()
        self.generate_trajectory_dashboard()
        self.generate_mppi_dashboard()
        
        print(f"All plots generated and saved to {self.output_dir}")
        
    def plot_vehicle_path(self, show=False):
        """Plot the vehicle path"""
        if self.vehicle_data is None:
            print("Vehicle data not available")
            return
        
        plt.figure(figsize=(12, 10))
        plt.plot(self.vehicle_data['x'], self.vehicle_data['y'], 
                 color=self.colors['path'], linewidth=2)
        
        # Add arrows to show direction
        skip = max(1, len(self.vehicle_data) // 50)  # Show ~50 arrows
        for i in range(0, len(self.vehicle_data), skip):
            x, y = self.vehicle_data['x'].iloc[i], self.vehicle_data['y'].iloc[i]
            theta = self.vehicle_data['theta'].iloc[i]
            dx, dy = 0.3 * np.cos(theta), 0.3 * np.sin(theta)
            plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.2, fc='r', ec='r')
        
        plt.grid(True)
        plt.axis('equal')
        plt.title("Vehicle Path", fontsize=16)
        plt.xlabel("X Position (m)", fontsize=12)
        plt.ylabel("Y Position (m)", fontsize=12)
        
        # Annotate start and end points
        plt.scatter(self.vehicle_data['x'].iloc[0], self.vehicle_data['y'].iloc[0], 
                    color='green', s=100, label='Start')
        plt.scatter(self.vehicle_data['x'].iloc[-1], self.vehicle_data['y'].iloc[-1], 
                    color='red', s=100, label='End')
        plt.legend()
        
        plt.savefig(self.output_dir / "vehicle_path.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf()
    
    def plot_speed_profile(self, show=False):
        """Plot the speed profile over time"""
        if self.control_data is None:
            print("Control data not available")
            return
        
        # Calculate relative timestamps in seconds
        timestamps = self.control_data['timestamp'] - self.control_data['timestamp'].iloc[0]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, self.control_data['actual_speed'], 
                 color=self.colors['speed'], linewidth=2, label='Actual Speed')
        plt.plot(timestamps, self.control_data['target_speed'], 
                 '--', color=self.colors['reference'], linewidth=1, label='Target Speed')
        
        plt.grid(True)
        plt.title("Vehicle Speed Profile", fontsize=16)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Speed (m/s)", fontsize=12)
        plt.legend()
        
        # Add statistics
        avg_speed = self.control_data['actual_speed'].mean()
        max_speed = self.control_data['actual_speed'].max()
        plt.annotate(f"Avg: {avg_speed:.2f} m/s\nMax: {max_speed:.2f} m/s",
                     xy=(0.02, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.savefig(self.output_dir / "speed_profile.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf()
    
    def plot_steering_profile(self, show=False):
        """Plot the steering profile over time"""
        if self.control_data is None:
            print("Control data not available")
            return
        
        # Calculate relative timestamps in seconds
        timestamps = self.control_data['timestamp'] - self.control_data['timestamp'].iloc[0]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, self.control_data['steering'], 
                 color=self.colors['steering'], linewidth=2)
        
        # Add zero line for reference
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.grid(True)
        plt.title("Steering Angle Profile", fontsize=16)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Steering Angle (rad)", fontsize=12)
        
        # Add statistics
        mean_steering = self.control_data['steering'].mean()
        std_steering = self.control_data['steering'].std()
        max_abs_steering = abs(self.control_data['steering']).max()
        
        plt.annotate(f"Mean: {mean_steering:.3f} rad\nStd: {std_steering:.3f} rad\nMax: {max_abs_steering:.3f} rad",
                     xy=(0.02, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.savefig(self.output_dir / "steering_profile.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf()
    
    def plot_costs_over_time(self, show=False):
        """Plot cost functions over time"""
        if self.trajectory_metrics is None:
            print("Trajectory metrics not available")
            return
        
        # Calculate relative timestamps in seconds
        timestamps = self.trajectory_metrics['timestamp'] - self.trajectory_metrics['timestamp'].iloc[0]
        
        plt.figure(figsize=(12, 8))
        plt.plot(timestamps, self.trajectory_metrics['tracking_cost'], 
                 color=self.colors['tracking_cost'], linewidth=2, label='Tracking Cost')
        plt.plot(timestamps, self.trajectory_metrics['obstacle_cost'], 
                 color=self.colors['obstacle_cost'], linewidth=2, label='Obstacle Cost')
        plt.plot(timestamps, self.trajectory_metrics['total_cost'], 
                 color=self.colors['total_cost'], linewidth=2, label='Total Cost')
        
        plt.grid(True)
        plt.title("Cost Functions Over Time", fontsize=16)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Cost Value", fontsize=12)
        plt.legend()
        
        # Add statistics
        mean_total = self.trajectory_metrics['total_cost'].mean()
        mean_tracking = self.trajectory_metrics['tracking_cost'].mean()
        mean_obstacle = self.trajectory_metrics['obstacle_cost'].mean()
        
        plt.annotate(f"Mean Total Cost: {mean_total:.2f}\nMean Tracking: {mean_tracking:.2f}\nMean Obstacle: {mean_obstacle:.2f}",
                     xy=(0.02, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.savefig(self.output_dir / "costs_over_time.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf()
    
    def plot_cross_track_error(self, show=False):
        """Plot cross-track error over time"""
        if self.trajectory_metrics is None:
            print("Trajectory metrics not available")
            return
        
        # Calculate relative timestamps in seconds
        timestamps = self.trajectory_metrics['timestamp'] - self.trajectory_metrics['timestamp'].iloc[0]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, self.trajectory_metrics['cross_track_error'], 
                 color=self.colors['path'], linewidth=2)
        
        plt.grid(True)
        plt.title("Cross-Track Error Over Time", fontsize=16)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Cross-Track Error (m)", fontsize=12)
        
        # Add statistics
        mean_error = self.trajectory_metrics['cross_track_error'].mean()
        max_error = self.trajectory_metrics['cross_track_error'].max()
        std_error = self.trajectory_metrics['cross_track_error'].std()
        
        plt.annotate(f"Mean: {mean_error:.3f} m\nMax: {max_error:.3f} m\nStd: {std_error:.3f} m",
                     xy=(0.02, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.savefig(self.output_dir / "cross_track_error.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf()
    
    def plot_weight_distribution(self, show=False):
        """Plot MPPI weight distribution statistics"""
        if self.mppi_metrics is None:
            print("MPPI metrics not available")
            return
        
        # Calculate relative timestamps in seconds
        timestamps = self.mppi_metrics['timestamp'] - self.mppi_metrics['timestamp'].iloc[0]
        
        plt.figure(figsize=(12, 10))
        
        # Create a subplot grid
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        
        # Top plot: Weight values
        ax1 = plt.subplot(gs[0])
        ax1.plot(timestamps, self.mppi_metrics['max_weight'], 
                 color='red', linewidth=2, label='Max Weight')
        ax1.plot(timestamps, self.mppi_metrics['mean_weight'], 
                 color='green', linewidth=2, label='Mean Weight')
        ax1.plot(timestamps, self.mppi_metrics['min_weight'], 
                 color='blue', linewidth=2, label='Min Weight')
        
        ax1.set_title("MPPI Weight Distribution", fontsize=16)
        ax1.set_ylabel("Weight Value", fontsize=12)
        ax1.grid(True)
        ax1.legend()
        
        # Bottom plot: Weight entropy (diversity measure)
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(timestamps, self.mppi_metrics['weight_entropy'], 
                 color='purple', linewidth=2)
        
        ax2.set_xlabel("Time (s)", fontsize=12)
        ax2.set_ylabel("Weight Entropy", fontsize=12)
        ax2.grid(True)
        
        # Add statistics
        mean_entropy = self.mppi_metrics['weight_entropy'].mean()
        ax2.annotate(f"Mean Entropy: {mean_entropy:.3f}",
                     xy=(0.02, 0.85), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "weight_distribution.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf()
    
    def plot_obstacle_metrics(self, show=False):
        """Plot obstacle detection metrics"""
        if self.obstacle_data is None:
            print("Obstacle data not available")
            return
        
        # Calculate relative timestamps in seconds
        timestamps = self.obstacle_data['timestamp'] - self.obstacle_data['timestamp'].iloc[0]
        
        plt.figure(figsize=(12, 8))
        
        # Create stacked area chart of point types
        plt.stackplot(timestamps, 
                     self.obstacle_data['free_points'],
                     self.obstacle_data['unknown_points'], 
                     self.obstacle_data['obstacle_points'],
                     self.obstacle_data['outside_points'],
                     labels=['Free', 'Unknown', 'Obstacle', 'Outside'],
                     colors=[self.colors['free'], self.colors['unknown'], 
                             self.colors['obstacle'], 'darkgray'])
        
        plt.grid(True)
        plt.title("Obstacle Detection Metrics", fontsize=16)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Point Count", fontsize=12)
        plt.legend(loc='upper right')
        
        # Add statistics
        avg_free = self.obstacle_data['free_points'].mean()
        avg_obstacle = self.obstacle_data['obstacle_points'].mean()
        
        plt.annotate(f"Avg Free: {avg_free:.1f}\nAvg Obstacles: {avg_obstacle:.1f}",
                     xy=(0.02, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.savefig(self.output_dir / "obstacle_metrics.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf()
    
    def plot_computation_time(self, show=False):
        """Plot computation time performance"""
        if self.timing_data is None:
            print("Timing data not available")
            return
        
        # Calculate relative timestamps in seconds
        timestamps = self.timing_data['timestamp'] - self.timing_data['timestamp'].iloc[0]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, self.timing_data['computation_time'] * 1000, 
                 color=self.colors['computation'], linewidth=1)
        
        # Add a rolling average
        window = min(20, len(self.timing_data) // 5)
        if window > 0:
            rolling_avg = self.timing_data['computation_time'].rolling(window=window).mean() * 1000
            plt.plot(timestamps, rolling_avg, 'r-', linewidth=2, label=f'{window}-point Moving Avg')
        
        plt.grid(True)
        plt.title("Computation Time Performance", fontsize=16)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Computation Time (ms)", fontsize=12)
        if window > 0:
            plt.legend()
        
        # Add statistics
        mean_time = self.timing_data['computation_time'].mean() * 1000
        max_time = self.timing_data['computation_time'].max() * 1000
        min_time = self.timing_data['computation_time'].min() * 1000
        std_time = self.timing_data['computation_time'].std() * 1000
        
        plt.annotate(f"Mean: {mean_time:.2f} ms\nMax: {max_time:.2f} ms\nMin: {min_time:.2f} ms\nStd: {std_time:.2f} ms",
                     xy=(0.02, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.savefig(self.output_dir / "computation_time.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf()
    
    def generate_performance_dashboard(self, show=False):
        """Generate a performance dashboard with multiple metrics"""
        if self.vehicle_data is None or self.control_data is None:
            print("Vehicle or control data not available")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 2)
        
        # Vehicle path
        ax1 = fig.add_subplot(gs[0, 0])
        if self.vehicle_data is not None:
            ax1.plot(self.vehicle_data['x'], self.vehicle_data['y'], 
                    color=self.colors['path'], linewidth=2)
            ax1.scatter(self.vehicle_data['x'].iloc[0], self.vehicle_data['y'].iloc[0], 
                        color='green', s=100, label='Start')
            ax1.scatter(self.vehicle_data['x'].iloc[-1], self.vehicle_data['y'].iloc[-1], 
                        color='red', s=100, label='End')
            ax1.set_title("Vehicle Path", fontsize=14)
            ax1.set_xlabel("X Position (m)", fontsize=10)
            ax1.set_ylabel("Y Position (m)", fontsize=10)
            ax1.grid(True)
            ax1.axis('equal')
            ax1.legend()
        
        # Speed profile
        ax2 = fig.add_subplot(gs[0, 1])
        if self.control_data is not None:
            timestamps = self.control_data['timestamp'] - self.control_data['timestamp'].iloc[0]
            ax2.plot(timestamps, self.control_data['actual_speed'], 
                    color=self.colors['speed'], linewidth=2, label='Actual Speed')
            ax2.plot(timestamps, self.control_data['target_speed'], 
                    '--', color=self.colors['reference'], linewidth=1, label='Target Speed')
            ax2.set_title("Speed Profile", fontsize=14)
            ax2.set_xlabel("Time (s)", fontsize=10)
            ax2.set_ylabel("Speed (m/s)", fontsize=10)
            ax2.grid(True)
            ax2.legend()
        
        # Steering profile
        ax3 = fig.add_subplot(gs[1, 0])
        if self.control_data is not None:
            timestamps = self.control_data['timestamp'] - self.control_data['timestamp'].iloc[0]
            ax3.plot(timestamps, self.control_data['steering'], 
                    color=self.colors['steering'], linewidth=2)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax3.set_title("Steering Angle", fontsize=14)
            ax3.set_xlabel("Time (s)", fontsize=10)
            ax3.set_ylabel("Steering Angle (rad)", fontsize=10)
            ax3.grid(True)
        
        # Cross-track error
        ax4 = fig.add_subplot(gs[1, 1])
        if self.trajectory_metrics is not None:
            timestamps = self.trajectory_metrics['timestamp'] - self.trajectory_metrics['timestamp'].iloc[0]
            ax4.plot(timestamps, self.trajectory_metrics['cross_track_error'], 
                    color=self.colors['path'], linewidth=2)
            ax4.set_title("Cross-Track Error", fontsize=14)
            ax4.set_xlabel("Time (s)", fontsize=10)
            ax4.set_ylabel("Error (m)", fontsize=10)
            ax4.grid(True)
            
            # Add statistics
            mean_error = self.trajectory_metrics['cross_track_error'].mean()
            max_error = self.trajectory_metrics['cross_track_error'].max()
            ax4.annotate(f"Mean: {mean_error:.3f} m\nMax: {max_error:.3f} m",
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Cost functions
        ax5 = fig.add_subplot(gs[2, 0])
        if self.trajectory_metrics is not None:
            timestamps = self.trajectory_metrics['timestamp'] - self.trajectory_metrics['timestamp'].iloc[0]
            ax5.plot(timestamps, self.trajectory_metrics['tracking_cost'], 
                    color=self.colors['tracking_cost'], linewidth=2, label='Tracking Cost')
            ax5.plot(timestamps, self.trajectory_metrics['obstacle_cost'], 
                    color=self.colors['obstacle_cost'], linewidth=2, label='Obstacle Cost')
            ax5.plot(timestamps, self.trajectory_metrics['total_cost'], 
                    color=self.colors['total_cost'], linewidth=2, label='Total Cost')
            ax5.set_title("Cost Functions", fontsize=14)
            ax5.set_xlabel("Time (s)", fontsize=10)
            ax5.set_ylabel("Cost Value", fontsize=10)
            ax5.grid(True)
            ax5.legend()
        
        # Computation time
        ax6 = fig.add_subplot(gs[2, 1])
        if self.timing_data is not None:
            timestamps = self.timing_data['timestamp'] - self.timing_data['timestamp'].iloc[0]
            ax6.plot(timestamps, self.timing_data['computation_time'] * 1000, 
                    color=self.colors['computation'], linewidth=2)
            ax6.set_title("Computation Time", fontsize=14)
            ax6.set_xlabel("Time (s)", fontsize=10)
            ax6.set_ylabel("Time (ms)", fontsize=10)
            ax6.grid(True)
            
            # Add statistics
            mean_time = self.timing_data['computation_time'].mean() * 1000
            max_time = self.timing_data['computation_time'].max() * 1000
            ax6.annotate(f"Mean: {mean_time:.2f} ms\nMax: {max_time:.2f} ms",
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_dashboard.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def generate_trajectory_dashboard(self, show=False):
        """Generate a trajectory-focused dashboard"""
        # Similar implementation to performance dashboard but focused on trajectory metrics
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(2, 2)
        
        # Vehicle path with color-coded speed
        ax1 = fig.add_subplot(gs[0, 0])
        if self.vehicle_data is not None:
            points = np.array([self.vehicle_data['x'], self.vehicle_data['y']]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Create a line collection with speed-based coloring
            norm = plt.Normalize(self.vehicle_data['speed'].min(), self.vehicle_data['speed'].max())
            lc = plt.matplotlib.collections.LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(self.vehicle_data['speed'])
            line = ax1.add_collection(lc)
            fig.colorbar(line, ax=ax1, label='Speed (m/s)')
            
            ax1.set_xlim(self.vehicle_data['x'].min(), self.vehicle_data['x'].max())
            ax1.set_ylim(self.vehicle_data['y'].min(), self.vehicle_data['y'].max())
            ax1.set_title("Vehicle Path with Speed", fontsize=14)
            ax1.set_xlabel("X Position (m)", fontsize=10)
            ax1.set_ylabel("Y Position (m)", fontsize=10)
            ax1.grid(True)
            ax1.axis('equal')
        
        # Cross-track error heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        if self.vehicle_data is not None and self.trajectory_metrics is not None:
            # Interpolate cross_track_error to match vehicle data
            from scipy.interpolate import interp1d
            
            # Ensure we have trajectory metrics
            if len(self.trajectory_metrics) > 1:
                # Create interpolation function
                traj_time = self.trajectory_metrics['timestamp'].values
                veh_time = self.vehicle_data['timestamp'].values
                cte = self.trajectory_metrics['cross_track_error'].values
                
                # Make sure times are in order
                sorted_indices = np.argsort(traj_time)
                traj_time = traj_time[sorted_indices]
                cte = cte[sorted_indices]
                
                # Create interpolator and get values at vehicle timestamps
                if len(traj_time) > 1:  # Need at least 2 points for interpolation
                    f = interp1d(traj_time, cte, bounds_error=False, fill_value='extrapolate')
                    interpolated_cte = f(veh_time)
                    
                    # Plot the path with CTE coloring
                    points = np.array([self.vehicle_data['x'], self.vehicle_data['y']]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    
                    norm = plt.Normalize(0, max(0.5, np.percentile(interpolated_cte, 95)))
                    lc = plt.matplotlib.collections.LineCollection(segments, cmap='coolwarm', norm=norm)
                    lc.set_array(interpolated_cte)
                    line = ax2.add_collection(lc)
                    fig.colorbar(line, ax=ax2, label='Cross-Track Error (m)')
                    
                    ax2.set_xlim(self.vehicle_data['x'].min(), self.vehicle_data['x'].max())
                    ax2.set_ylim(self.vehicle_data['y'].min(), self.vehicle_data['y'].max())
            
            ax2.set_title("Cross-Track Error on Path", fontsize=14)
            ax2.set_xlabel("X Position (m)", fontsize=10)
            ax2.set_ylabel("Y Position (m)", fontsize=10)
            ax2.grid(True)
            ax2.axis('equal')
        
        # Obstacle avoidance visualization
        ax3 = fig.add_subplot(gs[1, 0])
        if self.obstacle_data is not None:
            timestamps = self.obstacle_data['timestamp'] - self.obstacle_data['timestamp'].iloc[0]
            
            # Stack the different point types
            ax3.stackplot(timestamps, 
                         self.obstacle_data['free_points'],
                         self.obstacle_data['unknown_points'], 
                         self.obstacle_data['obstacle_points'],
                         self.obstacle_data['outside_points'],
                         labels=['Free', 'Unknown', 'Obstacle', 'Outside'],
                         colors=[self.colors['free'], self.colors['unknown'], 
                                 self.colors['obstacle'], 'darkgray'],
                         alpha=0.7)
            
            ax3.set_title("Obstacle Detection", fontsize=14)
            ax3.set_xlabel("Time (s)", fontsize=10)
            ax3.set_ylabel("Point Count", fontsize=10)
            ax3.grid(True)
            ax3.legend(loc='upper right')
        
        # Cost function breakdown
        ax4 = fig.add_subplot(gs[1, 1])
        if self.trajectory_metrics is not None:
            timestamps = self.trajectory_metrics['timestamp'] - self.trajectory_metrics['timestamp'].iloc[0]
            
            # Create percentage stacked area
            tracking = self.trajectory_metrics['tracking_cost'].values
            obstacle = self.trajectory_metrics['obstacle_cost'].values
            
            # Convert to absolute values for visualization
            abs_tracking = np.abs(tracking)
            abs_obstacle = np.abs(obstacle)
            total = abs_tracking + abs_obstacle
            
            # Handle division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_tracking = np.where(total > 0, abs_tracking / total * 100, 0)
                pct_obstacle = np.where(total > 0, abs_obstacle / total * 100, 0)
            
            ax4.stackplot(timestamps, pct_tracking, pct_obstacle,
                         labels=['Tracking Cost', 'Obstacle Cost'],
                         colors=[self.colors['tracking_cost'], self.colors['obstacle_cost']],
                         alpha=0.7)
            
            ax4.set_title("Cost Function Distribution", fontsize=14)
            ax4.set_xlabel("Time (s)", fontsize=10)
            ax4.set_ylabel("Percentage of Total Cost", fontsize=10)
            ax4.set_ylim(0, 100)
            ax4.grid(True)
            ax4.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "trajectory_dashboard.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def generate_mppi_dashboard(self, show=False):
        """Generate an MPPI algorithm performance dashboard"""
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(2, 2)
        
        # Weight distribution over time
        ax1 = fig.add_subplot(gs[0, 0])
        if self.mppi_metrics is not None:
            timestamps = self.mppi_metrics['timestamp'] - self.mppi_metrics['timestamp'].iloc[0]
            
            ax1.plot(timestamps, self.mppi_metrics['max_weight'], 
                    color='red', linewidth=2, label='Max Weight')
            ax1.plot(timestamps, self.mppi_metrics['mean_weight'], 
                    color='green', linewidth=2, label='Mean Weight')
            ax1.plot(timestamps, self.mppi_metrics['min_weight'], 
                    color='blue', linewidth=2, label='Min Weight')
            
            ax1.set_title("MPPI Weight Distribution", fontsize=14)
            ax1.set_xlabel("Time (s)", fontsize=10)
            ax1.set_ylabel("Weight Value", fontsize=10)
            ax1.grid(True)
            ax1.legend()
        
        # Weight entropy (diversity measure)
        ax2 = fig.add_subplot(gs[0, 1])
        if self.mppi_metrics is not None:
            timestamps = self.mppi_metrics['timestamp'] - self.mppi_metrics['timestamp'].iloc[0]
            
            ax2.plot(timestamps, self.mppi_metrics['weight_entropy'], 
                    color='purple', linewidth=2)
            
            ax2.set_title("Weight Entropy (Diversity)", fontsize=14)
            ax2.set_xlabel("Time (s)", fontsize=10)
            ax2.set_ylabel("Entropy", fontsize=10)
            ax2.grid(True)
            
            # Add statistics
            mean_entropy = self.mppi_metrics['weight_entropy'].mean()
            ax2.annotate(f"Mean Entropy: {mean_entropy:.3f}",
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Computation time histogram
        ax3 = fig.add_subplot(gs[1, 0])
        if self.timing_data is not None:
            # Convert to milliseconds for better readability
            comp_times_ms = self.timing_data['computation_time'] * 1000
            
            # Create histogram
            ax3.hist(comp_times_ms, bins=30, color=self.colors['computation'], alpha=0.7)
            ax3.axvline(x=comp_times_ms.mean(), color='r', linestyle='--', 
                       label=f'Mean: {comp_times_ms.mean():.2f} ms')
            ax3.axvline(x=comp_times_ms.median(), color='g', linestyle='-.', 
                       label=f'Median: {comp_times_ms.median():.2f} ms')
            
            ax3.set_title("Computation Time Distribution", fontsize=14)
            ax3.set_xlabel("Computation Time (ms)", fontsize=10)
            ax3.set_ylabel("Frequency", fontsize=10)
            ax3.grid(True)
            ax3.legend()
        
        # Cost vs entropy correlation
        ax4 = fig.add_subplot(gs[1, 1])
        if self.mppi_metrics is not None and self.trajectory_metrics is not None:
            # Merge dataframes on timestamp
            from scipy.interpolate import interp1d
            
            # Get data
            mppi_time = self.mppi_metrics['timestamp'].values
            traj_time = self.trajectory_metrics['timestamp'].values
            
            # Interpolate total cost at MPPI timestamps
            if len(traj_time) > 1 and len(mppi_time) > 1:
                # Sort arrays by time
                mppi_sorted_idx = np.argsort(mppi_time)
                traj_sorted_idx = np.argsort(traj_time)
                
                mppi_time_sorted = mppi_time[mppi_sorted_idx]
                entropy_sorted = self.mppi_metrics['weight_entropy'].values[mppi_sorted_idx]
                
                traj_time_sorted = traj_time[traj_sorted_idx]
                total_cost_sorted = self.trajectory_metrics['total_cost'].values[traj_sorted_idx]
                
                # Create interpolation function
                f = interp1d(traj_time_sorted, total_cost_sorted, bounds_error=False, fill_value='extrapolate')
                interpolated_cost = f(mppi_time_sorted)
                
                # Create scatter plot
                scatter = ax4.scatter(entropy_sorted, interpolated_cost, 
                                     c=mppi_time_sorted, cmap='viridis', alpha=0.6)
                
                # Add colorbar for time
                cbar = fig.colorbar(scatter, ax=ax4, label='Time (s)')
                cbar.set_label('Time (s)')
                
                # Add trend line
                from scipy import stats
                
                # Calculate correlation coefficient
                if len(entropy_sorted) > 1:
                    r, p = stats.pearsonr(entropy_sorted, interpolated_cost)
                    
                    # Add trend line if there's a significant correlation
                    if not np.isnan(r):
                        slope, intercept = np.polyfit(entropy_sorted, interpolated_cost, 1)
                        x_range = np.linspace(min(entropy_sorted), max(entropy_sorted), 100)
                        ax4.plot(x_range, intercept + slope * x_range, 'r--', 
                                linewidth=2, label=f'r = {r:.3f}, p = {p:.3e}')
                        
                        ax4.set_title(f"Cost vs Weight Entropy (r = {r:.3f})", fontsize=14)
                    else:
                        ax4.set_title("Cost vs Weight Entropy", fontsize=14)
                    
                    ax4.set_xlabel("Weight Entropy", fontsize=10)
                    ax4.set_ylabel("Total Cost", fontsize=10)
                    ax4.grid(True)
                    
                    if not np.isnan(r):
                        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "mppi_dashboard.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def generate_report(self, output_path=None):
        """Generate a comprehensive PDF report"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
            from reportlab.lib.styles import getSampleStyleSheet
            
            if output_path is None:
                output_path = self.output_dir / "mppi_report.pdf"
            
            # Generate all plots first
            self.generate_all_plots()
            
            # Create the document
            doc = SimpleDocTemplate(str(output_path), pagesize=letter)
            elements = []
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = styles['Title']
            heading_style = styles['Heading1']
            subheading_style = styles['Heading2']
            normal_style = styles['Normal']
            
            # Add title
            elements.append(Paragraph("MPPI Controller Performance Report", title_style))
            elements.append(Spacer(1, 20))
            
            # Add date and time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elements.append(Paragraph(f"Generated: {current_time}", normal_style))
            elements.append(Spacer(1, 20))
            
            # Summary statistics
            elements.append(Paragraph("Summary Statistics", heading_style))
            
            # Create summary table
            data = [["Metric", "Value"]]
            
            if self.vehicle_data is not None:
                data.append(["Distance Traveled (m)", f"{self.calculate_distance_traveled():.2f}"])
                data.append(["Average Speed (m/s)", f"{self.vehicle_data['speed'].mean():.2f}"])
                data.append(["Max Speed (m/s)", f"{self.vehicle_data['speed'].max():.2f}"])
            
            if self.trajectory_metrics is not None:
                data.append(["Average Cross-Track Error (m)", f"{self.trajectory_metrics['cross_track_error'].mean():.3f}"])
                data.append(["Max Cross-Track Error (m)", f"{self.trajectory_metrics['cross_track_error'].max():.3f}"])
            
            if self.timing_data is not None:
                data.append(["Average Computation Time (ms)", f"{self.timing_data['computation_time'].mean() * 1000:.2f}"])
                data.append(["Max Computation Time (ms)", f"{self.timing_data['computation_time'].max() * 1000:.2f}"])
            
            if self.mppi_metrics is not None:
                data.append(["Average Weight Entropy", f"{self.mppi_metrics['weight_entropy'].mean():.3f}"])
            
            # Create the table
            table = Table(data, colWidths=[300, 150])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (1, 0), 12),
                ('BACKGROUND', (0, 1), (1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 20))
            
            # Performance Visualizations section
            elements.append(Paragraph("Performance Visualizations", heading_style))
            elements.append(Spacer(1, 10))
            
            # Add dashboard plots
            for plot_name in ["performance_dashboard.png", "trajectory_dashboard.png", "mppi_dashboard.png"]:
                plot_path = self.output_dir / plot_name
                if plot_path.exists():
                    img = Image(str(plot_path), width=500, height=400)
                    elements.append(img)
                    elements.append(Spacer(1, 20))
            
            # Detailed Metrics section
            elements.append(Paragraph("Detailed Metrics", heading_style))
            elements.append(Spacer(1, 10))
            
            # Add individual metrics plots
            for plot_name, description in [
                ("vehicle_path.png", "Vehicle path showing the trajectory followed by the vehicle."),
                ("speed_profile.png", "Speed profile showing the actual and target speeds over time."),
                ("steering_profile.png", "Steering angle profile showing the control inputs over time."),
                ("cross_track_error.png", "Cross-track error showing the deviation from the reference path."),
                ("costs_over_time.png", "Cost functions showing the tracking and obstacle costs over time."),
                ("weight_distribution.png", "MPPI weight distribution showing the diversity of sampled trajectories."),
                ("obstacle_metrics.png", "Obstacle metrics showing the free, unknown, and obstacle points detected."),
                ("computation_time.png", "Computation time performance showing the processing time per control cycle.")
            ]:
                plot_path = self.output_dir / plot_name
                if plot_path.exists():
                    elements.append(Paragraph(plot_name.replace(".png", "").replace("_", " ").title(), subheading_style))
                    elements.append(Paragraph(description, normal_style))
                    img = Image(str(plot_path), width=450, height=300)
                    elements.append(img)
                    elements.append(Spacer(1, 20))
            
            # Build the PDF
            doc.build(elements)
            print(f"Report generated successfully: {output_path}")
            return output_path
        
        except ImportError:
            print("ReportLab package is required to generate PDF reports.")
            print("Install it using: pip install reportlab")
            return None
        except Exception as e:
            print(f"Error generating report: {e}")
            return None
    
    def calculate_distance_traveled(self):
        """Calculate the total distance traveled by the vehicle"""
        if self.vehicle_data is None or len(self.vehicle_data) < 2:
            return 0.0
        
        # Calculate Euclidean distance between consecutive points and sum
        x = self.vehicle_data['x'].values
        y = self.vehicle_data['y'].values
        
        # Calculate differences between consecutive points
        dx = np.diff(x)
        dy = np.diff(y)
        
        # Calculate Euclidean distances and sum
        distances = np.sqrt(dx**2 + dy**2)
        total_distance = np.sum(distances)
        
        return total_distance

def main():
    """Main function to demonstrate usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate plots from MPPI data')
    parser.add_argument('data_dir', type=str, help='Directory containing CSV data files')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save plots (default: data_dir/plots)')
    parser.add_argument('--report', action='store_true', help='Generate a PDF report')
    parser.add_argument('--show', action='store_true', help='Show plots instead of just saving them')
    
    args = parser.parse_args()
    
    try:
        # Create PlotGenerator instance
        plot_gen = PlotGenerator(args.data_dir, args.output_dir)
        
        if args.report:
            # Generate report
            plot_gen.generate_report()
        else:
            # Generate all plots
            plot_gen.generate_all_plots()
        
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
            
        print(f"All plots generated successfully and saved to {plot_gen.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

# example usage:
# python plot_generator.py /path/to/data --output_dir /path/to/output --report

# /bin/python3 /home/navjot/Desktop/work/sim_ws/src/mppi_f1tenth_team4/mppi/mppi/plot_generator.py /home/navjot/mppi_data --output_dir /home/navjot/mppi_data/plots --report