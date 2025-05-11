#!/usr/bin/env python3
import numpy as np
import os, sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import MultiArrayDimension, Float32MultiArray
from mppi.utils.ros_np_multiarray import to_multiarray_f32, to_numpy_f32
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from mppi.utils.Track import Track
from mppi.utils.utils import ConfigYAML
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry


class Visualizer_Node(Node):
    def __init__(self):
        super().__init__('visualizer_node')
        # fixed display position for speed HUD
        self.speed_x, self.speed_y, self.speed_z = 0.0, 0.0, 1.0
        # publisher for execution speed
        self.exec_speed_pub = self.create_publisher(MarkerArray, '/vis/exec_speed', 1)
        # subscribe to drive commands
        self.drive_sub = self.create_subscription(AckermannDriveStamped, '/drive', self.drive_callback, 1)
        # publisher for odom speed
        self.odom_speed_pub = self.create_publisher(MarkerArray, '/vis/odom_speed', 1)
        # subscribe odometry
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 1)
        # publisher for opt first point speed
        self.opt_speed_pub = self.create_publisher(MarkerArray, '/vis/opt_speed', 1)
        self.obs_max_num = 108
        self.reference_max_num = 100
        self.opt_traj_max_num = 10
        self.opt_traj_f = None

        # publishers
        self.reference_pub = self.create_publisher(MarkerArray, "/vis/reference", 1)
        self.opt_traj_pub = self.create_publisher(MarkerArray, "/vis/opt_traj", 1)
        self.opt_traj_speed_pub = self.create_publisher(MarkerArray, "/vis/opt_traj_speed", 1)
        self.obstacle_pub = self.create_publisher(MarkerArray, "/vis/obstacle", 1)
        self.track_pub = self.create_publisher(MarkerArray, '/vis/track', 1)
        self.sampled_vis_pub = self.create_publisher(MarkerArray, '/vis/sampled', 1)
        self.heavy_weights_pub = self.create_publisher(MarkerArray, '/vis/heavy_weights', 1)
        self.weight_labels_pub = self.create_publisher(MarkerArray, '/vis/weight_labels', 1)
        self.speed_pub = self.create_publisher(MarkerArray, '/vis/ref_speed', 1)
        # -----------------------------------------------------------------------------------------------------------
        self.nominal_pub = self.create_publisher(MarkerArray, '/vis/nominal', 1)
        # -----------------------------------------------------------------------------------------------------------

        
        self.frenet_pose_sub = self.create_subscription(Float32MultiArray, "/frenet_pose", self.frenet_pose_callback, 1)
        self.reference_sub = self.create_subscription(Float32MultiArray, "/reference_arr", self.reference_callback, 1)
        self.opt_traj_sub = self.create_subscription(Float32MultiArray, "/opt_traj_arr", self.opt_traj_callback, 1)
        self.obstacle_sub = self.create_subscription(Float32MultiArray, "/obstacle_arr_xy", self.obstacle_callback, 1)
        self.reward_sub = self.create_subscription(Float32MultiArray, "/reward_arr", self.reward_callback, 1)
        self.sampled_sub = self.create_subscription(Float32MultiArray, '/sampled_arr', self.sampled_callback, 1)
        self.weights_sub = self.create_subscription(Float32MultiArray, '/traj_weights', self.weights_callback, 1)
        # ----------------------------------------------------------------------------------------------------------
        self.nominal_sub = self.create_subscription(Float32MultiArray, '/nominal_arr',self.nominal_callback, 1)
        # ----------------------------------------------------------------------------------------------------------
        
        self.sampled_trajectories = None
        self.trajectory_weights = None

        # publish full track on startup
        config = ConfigYAML()
        cfg_path = Path(get_package_share_directory('mppi')) / 'config' / 'config.yaml'
        config.load_file(str(cfg_path))
        # load map_info
        waypoints_dir = Path(get_package_share_directory('mppi')) / 'waypoints'
        map_info = np.genfromtxt(str(waypoints_dir / 'map_info.txt'), delimiter='|', dtype='str')
        track, _ = Track.load_map(str(waypoints_dir)+'/', map_info, config.map_ind, config)
        # visualize full track as a thin line
        pts = track.waypoints[:, [1,2]].astype(np.float32)
        line_marker = Marker()
        line_marker.header.frame_id = 'map'
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = 'track_line'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.02  # line width
        line_marker.color.a = 1.0; line_marker.color.r = 1.0; line_marker.color.g = 1.0; line_marker.color.b = 0.0
        for x, y in pts:
            pt = Point(); pt.x = float(x); pt.y = float(y); pt.z = 0.0
            line_marker.points.append(pt)
        self.track_pub.publish(MarkerArray(markers=[line_marker]))

    def weights_callback(self, arr_msg):
        self.trajectory_weights = to_numpy_f32(arr_msg)
        if self.sampled_trajectories is not None and self.trajectory_weights is not None:
            self.visualize_heavy_weight_trajectories()

    def visualize_heavy_weight_trajectories(self):
        if self.heavy_weights_pub.get_subscription_count() == 0:
            return
            
        if len(self.trajectory_weights) != self.sampled_trajectories.shape[0]:
            self.get_logger().warn(f"权重数量({len(self.trajectory_weights)})与轨迹数量({self.sampled_trajectories.shape[0]})不匹配")
            return
        
        heavy_indices = np.where(self.trajectory_weights > 0.1)[0].astype(int)
        
        markers = MarkerArray()
        weight_labels = MarkerArray()
        
        for i, idx in enumerate(heavy_indices):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = f'heavy_traj_{int(idx)}'
            m.id = int(idx)
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.03
            m.color.a = 1.0
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 1.0
            
            traj = self.sampled_trajectories[idx]
            for j in range(traj.shape[0]):
                p = Point()
                p.x = float(traj[j, 0])
                p.y = float(traj[j, 1])
                p.z = 0.0
                m.points.append(p)
            
            markers.markers.append(m)
            
            txt = Marker()
            txt.header.frame_id = 'map'
            txt.header.stamp = self.get_clock().now().to_msg()
            txt.ns = f'weight_label_{int(idx)}'
            txt.id = int(idx)
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(traj[-1, 0])
            txt.pose.position.y = float(traj[-1, 1])
            txt.pose.position.z = 0.3
            txt.scale.z = 0.3
            txt.color.a = 1.0
            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0
            txt.text = f"{self.trajectory_weights[idx]:.3f}"
            
            weight_labels.markers.append(txt)
        
        if len(heavy_indices) < 100:
            for i in range(len(heavy_indices), 100):
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = f'heavy_traj_{i}'
                m.id = i
                m.action = Marker.DELETE
                markers.markers.append(m)
                
                txt = Marker()
                txt.header.frame_id = 'map'
                txt.header.stamp = self.get_clock().now().to_msg()
                txt.ns = f'weight_label_{i}'
                txt.id = i
                txt.action = Marker.DELETE
                weight_labels.markers.append(txt)
        
        self.heavy_weights_pub.publish(markers)
        self.weight_labels_pub.publish(weight_labels)

    def reward_callback(self, arr_msg):
        # QtPlotter calls removed: using RViz MarkerArray topics for visualization
        pass

    def reference_callback(self, arr_msg):
        reference_arr = to_numpy_f32(arr_msg)
        reference_msg = self.waypoints_to_markerArray(reference_arr, self.reference_max_num, 0, 1, r=0.0, g=0.0, b=1.0)
        self.reference_pub.publish(reference_msg)
        # visualize speed as text at each reference point
        speed_markers = MarkerArray()
        for i, v in enumerate(reference_arr[:self.reference_max_num]):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'ref_speed'
            m.id = i
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            # offset text further for clarity
            m.pose.position.x = float(v[0]) + 0.3
            m.pose.position.y = float(v[1]) + 0.3
            m.pose.position.z = 0.2
            # larger font size and black color
            m.scale.z = 0.3
            m.color.a = 0.9; m.color.r = 0.0; m.color.g = 0.0; m.color.b = 0.0
            # display speed (third column)
            m.text = f"{v[2]:.1f}"
            speed_markers.markers.append(m)
        self.speed_pub.publish(speed_markers)
        
    def frenet_pose_callback(self, arr_msg):
        reference_arr = to_numpy_f32(arr_msg)
        
    def obstacle_callback(self, arr_msg):
        obstacle_arr = to_numpy_f32(arr_msg)
        obstacle_msg = self.waypoints_to_markerArray(obstacle_arr, self.obs_max_num, 0, 1, r=1.0, g=0.0, b=0.0)
        self.obstacle_pub.publish(obstacle_msg)

    def opt_traj_callback(self, arr_msg):
        opt_traj_arr = to_numpy_f32(arr_msg)
        # self.qt_plotter.scatter(opt_traj_arr[:, 0], opt_traj_arr[:, 1], s=20, plot_num=0, live=1)
        opt_traj_msg = self.waypoints_to_markerArray(opt_traj_arr[:10], self.opt_traj_max_num, 0, 1, r=0.0, g=1.0, b=0.0)
        self.opt_traj_f = opt_traj_arr[10:]
        self.opt_traj_pub.publish(opt_traj_msg)
        # visualize speed as text at each opt_traj point
        opt_speed_markers = MarkerArray()
        for i, v in enumerate(opt_traj_arr[:self.opt_traj_max_num]):
            if i >= len(opt_traj_arr):
                break
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'opt_traj_speed'
            m.id = i
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            # position near the optimized trajectory point
            m.pose.position.x = float(v[0]) + 0.3
            m.pose.position.y = float(v[1]) - 0.3  # offset in different direction than reference
            m.pose.position.z = 0.2
            # green color for optimized speed text
            m.scale.z = 0.3
            m.color.a = 0.9; m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0
            # display speed (usually fourth column is speed in state vector)
            m.text = f"{v[3]:.1f}"
            opt_speed_markers.markers.append(m)
        self.opt_traj_speed_pub.publish(opt_speed_markers)

        # publish opt first point speed in blue text
        if self.opt_speed_pub.get_subscription_count() > 0 and opt_traj_arr.shape[0] > 0:
            v0 = opt_traj_arr[0, 3]
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'opt_speed'
            m.id = 0
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = self.speed_x
            m.pose.position.y = self.speed_y - 0.0
            m.pose.position.z = self.speed_z
            m.scale.z = 0.3
            m.color.a = 1.0; m.color.r = 0.0; m.color.g = 0.0; m.color.b = 1.0
            m.text = f"{v0:.1f}"
            self.opt_speed_pub.publish(MarkerArray(markers=[m]))

    def sampled_callback(self, arr_msg):
        sampled_arr = to_numpy_f32(arr_msg)  # [n_samples, n_steps, state_dim]
        self.sampled_trajectories = sampled_arr
        
        n_samples, n_steps, _ = sampled_arr.shape
        markers = MarkerArray()
        for i in range(n_samples):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'sampled_{}'.format(i)
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.01
            m.color.a = 1.0; m.color.r = 1.0; m.color.g = 0.0; m.color.b = 1.0
            for j in range(n_steps):
                p = Point(); p.x = float(sampled_arr[i,j,0]); p.y = float(sampled_arr[i,j,1]); p.z = 0.0
                m.points.append(p)
            markers.markers.append(m)
        self.sampled_vis_pub.publish(markers)
        
        if self.trajectory_weights is not None:
            self.visualize_heavy_weight_trajectories()

    def drive_callback(self, msg):
        # show current execution speed at car frame
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns = 'exec_speed'
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        # position relative to base_link
        # get current car position from odom subscription
        # use fixed position in map frame instead
        m.pose.position.x = self.speed_x
        m.pose.position.y = self.speed_y - 0.8
        m.pose.position.z = self.speed_z
        m.scale.z = 0.3
        m.color.a = 1.0; m.color.r = 0.0; m.color.g = 0.0; m.color.b = 0.0
        m.text = f"{msg.drive.speed:.1f}"
        self.exec_speed_pub.publish(MarkerArray(markers=[m]))
        # also publish execution speed HUD in green at same pos
        if self.exec_speed_pub.get_subscription_count() > 0:
            m2 = Marker()
            m2.header.frame_id = 'map'
            m2.header.stamp = self.get_clock().now().to_msg()
            m2.ns = 'exec_speed_hud'
            m2.id = 1
            m2.type = Marker.TEXT_VIEW_FACING
            m2.action = Marker.ADD
            m2.pose.position.x = self.speed_x
            m2.pose.position.y = self.speed_y - 0.4
            m2.pose.position.z = self.speed_z
            m2.scale.z = 0.3
            m2.color.a = 1.0; m2.color.r = 0.0; m2.color.g = 1.0; m2.color.b = 0.0
            m2.text = f"{msg.drive.speed:.1f}"
            self.exec_speed_pub.publish(MarkerArray(markers=[m2]))

    def odom_callback(self, msg):
        # publish odometry speed HUD in red
        if self.odom_speed_pub.get_subscription_count() > 0:
            v = msg.twist.twist.linear.x
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = msg.header.stamp
            m.ns = 'odom_speed'
            m.id = 0
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = self.speed_x
            m.pose.position.y = self.speed_y + 0.4
            m.pose.position.z = self.speed_z
            m.scale.z = 0.3
            m.color.a = 1.0; m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0
            m.text = f"{v:.1f}"
            self.odom_speed_pub.publish(MarkerArray(markers=[m]))


    # ──────────────────────────────────────────────────────────────
    def nominal_callback(self, arr_msg):
        traj = to_numpy_f32(arr_msg).reshape(-1, 2)   # [T,2]
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'nominal'
        m.id = 1000
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.04            # line width
        m.color.a = 1.0
        m.color.r = 0.0; m.color.g = 1.0; m.color.b = 1.0  # blue
        for x, y in traj:
            p = Point(); p.x = float(x); p.y = float(y); p.z = 0.0
            m.points.append(p)
        self.nominal_pub.publish(MarkerArray(markers=[m]))

    def waypoints_to_markerArray(self, waypoints, max_num, xind, yind, r=0.0, g=1.0, b=0.0):
        # Publish the reference trajectory
        array = MarkerArray()

        for i in range(max_num):
            message = Marker()
            message.header.frame_id = "map"
            message.header.stamp = self.get_clock().now().to_msg()
            message.type= Marker.SPHERE
            message.id=i
            message.pose.orientation.x=0.0
            message.pose.orientation.y=0.0
            message.pose.orientation.z=0.0
            message.pose.orientation.w=1.0
            message.scale.x=0.2
            message.scale.y=0.2
            message.scale.z=0.2
            message.color.a=1.0
            message.color.r=r
            message.color.b=g
            message.color.g=b
            if i < waypoints.shape[0]:
                message.pose.position.x=float(waypoints[i,xind])
                message.pose.position.y=float(waypoints[i,yind])
                message.action = Marker.ADD
            else:
                message.pose.position.x=0.
                message.pose.position.y=0.
                message.action = Marker.DELETE
            message.pose.position.z=0.0
            array.markers.append(message)

        return array

def main(args=None):
    rclpy.init(args=args)
    print("Track Visualizer Mode Initialized")
    lmppi_node = Visualizer_Node()
    rclpy.spin(lmppi_node)

    lmppi_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()