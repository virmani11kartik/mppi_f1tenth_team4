#!/usr/bin/env python3
import time, os, sys
import numpy as np
import jax
import jax.numpy as jnp
import rclpy
from rclpy.node import Node
import tf_transformations
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from mppi.utils.ros_np_multiarray import to_multiarray_f32, to_numpy_f32
from visualization_msgs.msg import Marker

from mppi.infer_env import InferEnv
from mppi.mppi_tracking import MPPI
import mppi.utils.utils as utils
from mppi.utils.jax_utils import numpify
import mppi.utils.jax_utils as jax_utils
from mppi.utils.Track import Track
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from visualization_msgs.msg import Marker, MarkerArray

jax.config.update("jax_compilation_cache_dir", str(Path.home() / "jax_cache"))

class MPPI_Node(Node):
    def __init__(self):
        super().__init__('lmppi_node')
        self.config = utils.ConfigYAML()
        config_path = Path(get_package_share_directory('mppi')) / 'config' / 'config.yaml'
        self.config.load_file(str(config_path))
        self.config.norm_params = np.array(self.config.norm_params).T
        if self.config.random_seed is None:
            self.config.random_seed = np.random.randint(0, 1e6)
        jrng = jax_utils.oneLineJaxRNG(self.config.random_seed)    
        
        waypoints_dir = Path(get_package_share_directory('mppi')) / 'waypoints'
        map_info_path = waypoints_dir / 'map_info.txt'
        map_info = np.genfromtxt(str(map_info_path), delimiter='|', dtype='str')
        map_dir_str = str(waypoints_dir)
        if not map_dir_str.endswith('/'):
            map_dir_str += '/'
        self.config.map_dir = map_dir_str
        track, self.config = Track.load_map(self.config.map_dir, map_info, self.config.map_ind, self.config)
        # track.waypoints[:, 3] += 0.5 * np.pi
        self.infer_env = InferEnv(track, self.config, DT=self.config.sim_time_step)
        self.mppi = MPPI(self.config, self.infer_env, jrng)

        # Do a dummy call on the MPPI to initialize the variables
        state_c_0 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.control = np.asarray([0.0, 0.0])
        reference_traj, waypoint_ind = self.infer_env.get_refernece_traj(state_c_0.copy(), self.config.ref_vel, self.config.n_steps)
        self.mppi.update(jnp.asarray(state_c_0), jnp.asarray(reference_traj))
        self.get_logger().info('MPPI initialized')
        
        
        qos = rclpy.qos.QoSProfile(history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
                                   depth=1,
                                   reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
                                   durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE)
        # create subscribers
        if self.config.is_sim:
            self.pose_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.pose_callback, qos)
        else:
            self.pose_sub = self.create_subscription(Odometry, "/pf/pose/odom", self.pose_callback, qos)
        # publishers
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, qos
        )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", qos)
        self.reference_pub = self.create_publisher(Float32MultiArray, "/reference_arr", qos)
        self.opt_traj_pub = self.create_publisher(Float32MultiArray, "/opt_traj_arr", qos)
        # publisher for sample trajectories
        self.sampled_pub = self.create_publisher(Float32MultiArray, "/sampled_arr", qos)


        self.scan_vis_pub = self.create_publisher(MarkerArray, "/vis/scan_points", qos)
        # Publisher for the single closest‐point marker
        self.closest_obs_pub = self.create_publisher(Marker, "/vis/closest_obs", qos)


    def pose_callback(self, pose_msg):
        """
        Callback function for subscribing to particle filter's inferred pose.
        This funcion saves the current pose of the car and obtain the goal
        waypoint from the pure pursuit module.

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        """
        pose = pose_msg.pose.pose
        twist = pose_msg.twist.twist

        # Beta calculated by the arctan of the lateral velocity and the longitudinal velocity
        beta = np.arctan2(twist.linear.y, twist.linear.x)

        # For demonstration, let's assume we have these quaternion values
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        # Convert quaternion to Euler angles
        euler = tf_transformations.euler_from_quaternion(quaternion)

        # Extract the Z-angle (yaw)
        theta = euler[2]  # Yaw is the third element

        state_c_0 = np.asarray([
            pose.position.x,
            pose.position.y,
            self.control[0],
            max(twist.linear.x, self.config.init_vel),
            theta,
            twist.angular.z,
            beta,
        ])
        
        # 使用None作为target_speed，让系统使用waypoints上的速度
        reference_traj, waypoint_ind = self.infer_env.get_refernece_traj(state_c_0.copy(), None, self.config.n_steps)

    
        if hasattr(self, 'latest_scan'):
            # state_c_0 = [x, y, steering, v, theta, yaw_rate, beta]
            self.infer_env.set_scan(self.latest_scan, state_c_0[:5])
            #                                     ^^^^^^^ give [x,y,steering,v,theta]

        # 1) compute the closest LiDAR point in world frame
        #################################################################################333
        pts = np.array(self.infer_env.scan_points)   # [N,2]
        if pts.shape[0] > 0:
            dists = np.hypot(pts[:,0] - state_c_0[0], pts[:,1] - state_c_0[1])
            idx_min = np.argmin(dists)
            x_closest, y_closest = pts[idx_min]
        else:
            x_closest, y_closest = float('nan'), float('nan')

        # 2) build and publish the big‐dot marker
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns              = "closest_obstacle"
        m.id              = 0
        m.type            = Marker.SPHERE
        m.action          = Marker.ADD

        # position
        m.pose.position.x = float(x_closest)
        m.pose.position.y = float(y_closest)
        m.pose.position.z = 0.1

        # appearance: big red semi-transparent
        m.scale.x = m.scale.y = m.scale.z = 0.3  
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 0.6

        self.closest_obs_pub.publish(m)
        ################################################################################3

        # ——— VISUALIZE scan_points in RViz ———
        # self.infer_env.scan_points is a jnp.array[[x,y],…]
        pts = np.array(self.infer_env.scan_points)  # convert to NumPy

        markers = MarkerArray()
        stamp   = self.get_clock().now().to_msg()
        for i, (x, y) in enumerate(pts):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp    = stamp
            m.ns              = "scan_points"
            m.id              = i
            m.type            = Marker.SPHERE
            m.action          = Marker.ADD

            # position & appearance
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.1              # slightly above ground
            m.scale.x = m.scale.y = m.scale.z = 0.2  # sphere diameter

            # red, semi-transparent
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.7

            markers.markers.append(m)

        # send to /vis/scan_points
        self.scan_vis_pub.publish(markers)
        ############################################################################################

        ## MPPI call
        self.mppi.update(jnp.asarray(state_c_0), jnp.asarray(reference_traj))
        mppi_control = numpify(self.mppi.a_opt[0]) * self.config.norm_params[0, :2]/2
        self.control[0] = float(mppi_control[0]) * self.config.sim_time_step + self.control[0]
        self.control[1] = float(mppi_control[1]) * self.config.sim_time_step + twist.linear.x
        
        if self.reference_pub.get_subscription_count() > 0:
            ref_traj_cpu = numpify(reference_traj)
            arr_msg = to_multiarray_f32(ref_traj_cpu.astype(np.float32))
            self.reference_pub.publish(arr_msg)

        if self.opt_traj_pub.get_subscription_count() > 0:
            opt_traj_cpu = numpify(self.mppi.traj_opt)
            arr_msg = to_multiarray_f32(opt_traj_cpu.astype(np.float32))
            self.opt_traj_pub.publish(arr_msg)

        # publish sampled trajectories
        if self.sampled_pub.get_subscription_count() > 0:
            sampled_cpu = numpify(self.mppi.states)  # [n_samples, n_steps, state_dim]
            arr_msg = to_multiarray_f32(sampled_cpu.astype(np.float32))
            self.sampled_pub.publish(arr_msg)

        if twist.linear.x < self.config.init_vel:
            self.control = [0.0, self.config.init_vel * 2]

        if np.isnan(self.control).any() or np.isinf(self.control).any():
            self.control = np.array([0.0, 0.0])
            self.mppi.a_opt = np.zeros_like(self.mppi.a_opt)

        # Publish the control command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = self.control[0]
        drive_msg.drive.speed = self.control[1]
        # self.get_logger().info(f"Steering Angle: {drive_msg.drive.steering_angle}, Speed: {drive_msg.drive.speed}")
        self.drive_pub.publish(drive_msg)

    def scan_callback(self, scan_msg):
        """
        Save the latest LaserScan so InferEnv can compute obstacle costs.
        """
        self.latest_scan = scan_msg
        

def main(args=None):
    rclpy.init(args=args)
    print("MPPI node initialized")
    mppi_node = MPPI_Node()
    rclpy.spin(mppi_node)

    mppi_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

