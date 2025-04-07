import time
import numpy as np
import os, sys
import jax
import jax.numpy as jnp
import math

np.set_printoptions(suppress=True, precision=5)
if 'ON_SERVER' not in os.environ:
    os.environ['ON_SERVER'] = 'False'
jax.config.update("jax_compilation_cache_dir", "/home/nvidia/jax_cache") 

import tf_transformations
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from utils.ros_np_multiarray import to_multiarray_f32, to_numpy_f32

# Append LMPPI_jax to the path
sys.path.append('../')
sys.path.append('./LMPPI_jax')
sys.path.append(os.path.join(os.path.dirname(__file__), 'LMPPI_jax'))

from infer_env import InferEnv
from mppi_tracking import MPPI
import utils.utils as utils
from utils.jax_utils import numpify
import utils.jax_utils as jax_utils
from utils.Track import Track

## This is a demosntration of how to use the MPPI planner with the F1Tenth Gym environment
## Zirui Zang 2024/05/30
class Config(utils.ConfigYAML):
    exp_name = 'mppi_tracking'
    segment_length = 1
    sim_time_step = 0.1
    render = 1
    kmonitor_enable = 0
    map_dir = './LMPPI_jax/f1tenth_racetracks/'
    save_dir = './results/' + exp_name + '/'
    max_lap = 300
    random_seed = None
    
    use_blank_map = True
    map_ext = '.png'
    map_scale = 1
    friction = 0.5
    ref_vel = 4.5
    init_vel = 1.0
    map_ind = 63
    n_steps = 10
    max_vel = 3.5
    n_samples = 1024
    n_iterations = 1
    control_dim = 2
    control_sample_noise = [1.0, 1.0]
    state_predictor = 'dynamic_ST' # 'dynamic_ST'
    cartesian_models = ['dynamic_ST', 'kinematic_ST', 'nn_dynamic_ST']
    half_width = 4
    
    adaptive_covariance = False
    init_noise = [5e-3, 5e-3, 5e-3] # control_vel, control_steering, state 
    # init_noise = [0, 0, 0] # control_vel, control_steering, state
    a_cov_shift = False

class MPPI_Node(Node):
    def __init__(self):
        super().__init__('lmppi_node')
        self.declare_parameter('is_sim', False)
        self.is_sim = self.get_parameter('is_sim').value
        
        # create subscribers
        if self.is_sim:
            self.pose_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.pose_callback, 1)
        else:
            self.pose_sub = self.create_subscription(Odometry, "/pf/pose/odom", self.pose_callback, 1)

        # publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 1)
        self.reference_pub = self.create_publisher(Float32MultiArray, "/reference_arr", 1)
        self.opt_traj_pub = self.create_publisher(Float32MultiArray, "/opt_traj_arr", 1)

        # variables
        self.delta = 0.0

        self.config = Config()
        self.config.load_file('./LMPPI_jax/planners/config.yaml')
        self.config.norm_params = np.array(self.config.norm_params).T
    
        if self.config.random_seed is None:
            self.config.random_seed = np.random.randint(0, 1e6)
        
        jrng = jax_utils.oneLineJaxRNG(self.config.random_seed)    
        map_info = np.genfromtxt(self.config.map_dir + 'map_info.txt', delimiter='|', dtype='str')
        track, self.config = Track.load_map(self.config.map_dir, map_info, self.config.map_ind, self.config, scale=self.config.map_scale, reverse=0)
        track.waypoints[:, 3] += 0.5 * math.pi
        # track.waypoints[:, 5] *= 0.7
        self.infer_env_st = InferEnv(track, self.config, DT=self.config.sim_time_step)
        self.mppi = MPPI(self.config, self.infer_env_st, jrng)        

        # Do a dummy call on the MPPI to initialize the variables
        state_c_0 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        reference_traj, waypoint_ind = self.infer_env_st.get_refernece_traj(state_c_0.copy(), self.config.ref_vel, self.config.n_steps)
        self.mppi.update(jnp.asarray(state_c_0), jnp.asarray(reference_traj))
        self.get_logger().info('MPPI initialized')

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

        # For demonstration, let’s assume we have these quaternion values
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        # Convert quaternion to Euler angles
        euler = tf_transformations.euler_from_quaternion(quaternion)

        # Extract the Z-angle (yaw)
        theta = euler[2]  # Yaw is the third element

        state = np.array([
            pose.position.x,
            pose.position.y,
            self.delta,
            twist.linear.x,
            theta,
            twist.angular.z,
            beta,
        ])
        #
        state_c_0 = np.asarray(state)
        print(twist.linear.x)
        reference_traj, waypoint_ind = self.infer_env_st.get_refernece_traj(state_c_0.copy(), max(twist.linear.x, self.config.ref_vel), self.config.n_steps)
        if self.reference_pub.get_subscription_count() > 0:
            ref_traj_cpu = numpify(reference_traj)
            arr_msg = to_multiarray_f32(ref_traj_cpu.astype(np.float32))
            self.reference_pub.publish(arr_msg)
        
        

        if twist.linear.x < self.config.init_vel*0.9:
            control = [0.0, self.config.init_vel]
            self.delta = 0.0
        else:
            # tick = time.time()
            ## MPPI call
            self.mppi.update(jnp.asarray(state_c_0), jnp.asarray(reference_traj))
            

            # if self.opt_traj_pub.get_subscription_count() > 0:
            opt_traj_cpu = numpify(self.mppi.traj_opt)
            arr_msg = to_multiarray_f32(opt_traj_cpu.astype(np.float32))
            self.opt_traj_pub.publish(arr_msg)

            mppi_control = numpify(self.mppi.a_opt[0]) * self.config.norm_params[0, :2]/2
            # self.get_logger().info(f"MPPI Control: {mppi_control}")
            control = [0.0, 0.0]
            # Integrate the accleartion to get the velocity
            control[1] = float(mppi_control[1]) * self.config.sim_time_step + twist.linear.x
            control[0] = float(mppi_control[0]) * self.config.sim_time_step + self.delta
            self.delta = control[0]
            # self.get_logger().info(f"Time taken to plan {time.time() - tick} | Hz {1/(time.time() - tick)}")

        # tick = time.time()
        if math.isnan(control[0]) or math.isnan(control[1]):
            control = np.array([0.0, self.config.init_vel])
            # Set the mppi_opt to control divided by the normalization parameter/2
            self.mppi.a_opt = np.ones_like(self.mppi.a_opt) * control
            self.delta = 0.0
            

        # Publish the control command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = control[0]
        # drive_msg.drive.speed = np.clip(control[1], -np.inf, self.config.max_vel)
        drive_msg.drive.speed = control[1]
        # if abs(drive_msg.drive.steering_angle) > 0.35:
        #     drive_msg.drive.speed *= 0.8
        # self.get_logger().info(f"Steering Angle: {drive_msg.drive.steering_angle}, Speed: {drive_msg.drive.speed}")
        self.drive_pub.publish(drive_msg)
        # self.get_logger().info(f"publish {time.time() - tick} | Hz {1/(time.time() - tick)}")
        

def main(args=None):
    rclpy.init(args=args)
    mppi_node = MPPI_Node()
    rclpy.spin(mppi_node)

    mppi_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()