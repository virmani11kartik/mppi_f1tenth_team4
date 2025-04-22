#!/usr/bin/env python3
import math
from dataclasses import dataclass, field
import pandas as pd

import cvxpy
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from sensor_msgs.msg import LaserScan
from mpc.utils import nearest_point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from scipy.interpolate import CubicSpline

# TODO CHECK: include needed ROS msg type headers and libraries


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 8  # finite time horizon length kinematic
    MPPI_HORIZON: int = 40           # Rollout length
    MPPI_NUM_SAMPLES: int = 100     # Number of trajectories to sample
    MPPI_LAMBDA: float = 1.0         # Temperature for softmin
    MPPI_NOISE_STD: list = field(default_factory=lambda: np.array([0.4, np.deg2rad(4.0)]))  # Noise for [accel, steer]


    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.05  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0

class MPC(Node):
    """ 
    Implement Kinematic MPC on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mpc_node')
        # TODO: create ROS subscribers and publishers
        #       use the MPC as a tracker (similar to pure pursuit)
        self.subscriber = self.create_subscription(
            Odometry,
            # '/pf/pose/odom',  # for real car
            '/ego_racecar/odom',  # for simulator
            self.pose_callback,
            10)
        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.marker_pub = self.create_publisher(MarkerArray, "mpc_markers", 10)
        self.marker_pub1 = self.create_publisher(MarkerArray, "mpc_markers1", 10)
        self.marker_pub2 = self.create_publisher(MarkerArray, "mpc_markers2", 10)
        # TODO: get waypoints here
        # Load waypoints from CSV file
        # waypoints_df = pd.read_csv("map3_levine_waypoints(1).csv")
        # raw_waypoints = waypoints_df[['x', 'y']].values
        raw_waypoints = np.load("waypoints_modified.npy")
        self.waypoints = self.process_waypoints(raw_waypoints)
        self.tracking_points = np.empty((0, 2))
        if self.waypoints.size > 0:
            self.fit_and_resample()

        self.config = mpc_config()
        self.odelta_v = None
        self.odelta = None
        self.oa = None
        self.init_flag = 0
    
    def quaternion_to_yaw(self, quaternion: Quaternion):
        """ Convert quaternion to yaw angle (assuming motion on a 2D plane). """
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y**2 + quaternion.z**2)
        return math.atan2(siny_cosp, cosy_cosp)

    def process_waypoints(self, raw_points):
        """Full waypoint processing pipeline"""
        if raw_points.size == 0:
            return np.empty((0, 2))
        # mask = rdp(raw_points, 
        #          epsilon=self.get_parameter('rdp_epsilon').value,
        #          return_mask=True)
        # simplified = raw_points[np.where(mask)[0]]
        simplified = raw_points
        if not np.allclose(simplified[0], simplified[-1]):
            simplified = np.vstack([simplified, simplified[0]])
            
        return simplified

    def fit_and_resample(self):
        x = self.waypoints[:, 0]
        y = self.waypoints[:, 1]
        # Arc-length parameterization
        dx = np.diff(x)
        dy = np.diff(y)
        dist = np.cumsum(np.hypot(dx, dy))
        dist = np.insert(dist, 0, 0)
        if dist[-1] == 0:  # Handle single-point case
            self.tracking_points = self.waypoints
            return   
        # Spline fitting
        spline_x = CubicSpline(dist/dist[-1], x)
        spline_y = CubicSpline(dist/dist[-1], y)
        # Generate dense spline points
        spline_dist = np.linspace(0, dist[-1], 3000)
        spline_points = np.vstack([spline_x(spline_dist/dist[-1]), 
                                 spline_y(spline_dist/dist[-1])]).T
        # Create tracking points
        self.create_tracking_points(spline_points, spline_dist)

    def create_tracking_points(self, spline_points, spline_dist):
        interval = 0.1
        num_points = int(spline_dist[-1] / interval)
        query_dist = np.linspace(0, spline_dist[-1], num_points)        
        # Linear interpolation of spline points
        self.tracking_points = np.array([
            np.interp(query_dist, spline_dist, spline_points[:, 0]),
            np.interp(query_dist, spline_dist, spline_points[:, 1])
        ]).T
    
    def pose_callback(self, odom_msg):
        pass
        # TODO: extract pose from ROS msg
        vehicle_state = State()
        ###################################################################
        vehicle_state.x = odom_msg.pose.pose.position.x
        vehicle_state.y = odom_msg.pose.pose.position.y
        vehicle_state.yaw = self.quaternion_to_yaw(odom_msg.pose.pose.orientation)
        vehicle_state.v = np.hypot(odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y)
        #self.get_logger().info(f"Pose callback triggered. x={vehicle_state.x:.2f}, v={vehicle_state.v:.2f}")

        ref_x = self.tracking_points[:, 0]
        ref_y = self.tracking_points[:, 1]
        dx = np.gradient(ref_x)
        dy = np.gradient(ref_y)
        ref_yaw = np.arctan2(dy, dx)
        # ref_yaw = np.clip(ref_yaw, -np.pi/2, np.pi/2)
        # ref_yaw = np.unwrap(ref_yaw)  # Smooth discontinuity
        # ref_yaw = np.convolve(ref_yaw, np.ones(5)/5, mode='same')
        dtheta = np.gradient(ref_yaw)
        ds = np.sqrt(dx**2 + dy**2) + 1e-6
        curvature = np.abs(dtheta / ds)
        ref_v = np.clip(3.0 / (1 + curvature + 1e-6), self.config.MIN_SPEED, self.config.MAX_SPEED)
        ref_v = ref_v * 1.8
        # ref_v = np.full_like(ref_x, 1.5)
        ###################################################################
        # TODO: Calculate the next reference trajectory for the next T steps
        #       with current vehicle pose.
        #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        ref_path = self.calc_ref_trajectory(vehicle_state, ref_x, ref_y, ref_yaw, ref_v)
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]
        # TODO: solve the MPPI control problem
        (self.oa, self.odelta_v, state_predict, samp_traj) = self.mppi_control(x0, ref_path)

        self.publish_mppi_prediction(state_predict)
        self.publish_sampled_trajectories(samp_traj)
        self.publish_yaw(ref_path)
        self.publish_path()
        # TODO: publish drive message.
        steer_output = self.odelta_v[0]
        speed_output = vehicle_state.v + self.oa[0] * self.config.DTK
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steer_output
        self.get_logger().info(f"speed={speed_output:.2f}, delta_v[0]={self.odelta_v[0]:.2f}, integrated delta={steer_output:.2f}")
        drive_msg.drive.speed = speed_output
        self.publisher.publish(drive_msg)

    def mppi_control(self, x0, ref_path):
        """
        MPPI control
        :param x0: current state
        :param ref_path: reference trajectory
        :return: optimal acceleration and steering angle
        """
        N = self.config.MPPI_NUM_SAMPLES
        T = min(self.config.MPPI_HORIZON, ref_path.shape[1] - 1)
        lambda_ = self.config.MPPI_LAMBDA
        noise_std = self.config.MPPI_NOISE_STD
        u_mean = np.zeros((T, 2)) 
        costs = np.zeros(N)
        u_samples = np.zeros((N, T, 2))
        trajectories = np.zeros((N, T + 1, 4)) 

        for k in range(N):
            state = State(x=x0[0], y=x0[1], v=x0[2], yaw=x0[3])
            total_cost = 0.0

            traj = np.zeros((T + 1, 4))
            traj[0] = [state.x, state.y, state.v, state.yaw]

            for t in range(T):
                noise = np.random.normal(0, noise_std, size=(2,))
                u = u_mean[t] + noise

                # Clip control limits
                u[0] = np.clip(u[0], -self.config.MAX_ACCEL, self.config.MAX_ACCEL)
                u[1] = np.clip(u[1], -self.config.MAX_STEER, self.config.MAX_STEER)

                u_samples[k, t] = u
                state = self.update_state(state, u[0], u[1])
                traj[t + 1] = [state.x, state.y, state.v, state.yaw]

                # Track reference
                cost_pos = np.linalg.norm(traj[t + 1, :2] - ref_path[:2, t])
                cost_yaw = (traj[t + 1, 3] - ref_path[3, t]) ** 2
                cost_speed = (traj[t + 1, 2] - ref_path[2, t]) ** 2
                total_cost += 30.0 * cost_pos + 10.0 * cost_speed + 5.0 * cost_yaw
                if t == T - 1:
                    total_cost += 100.0 * cost_pos + 40.0 * cost_yaw


            costs[k] = total_cost
            trajectories[k] = traj
        # Softmin
        min_cost = np.min(costs)
        exp_weights = np.exp(-(costs - min_cost) / lambda_)
        weights = exp_weights / np.sum(exp_weights)

        # Weighted average of first controls
        u0 = np.sum(weights[:, None] * u_samples[:, 0, :], axis=0)
        oa = [u0[0]] * T
        odelta = [u0[1]] * T
        best_index = np.argmin(costs)
        best_traj = trajectories[best_index].T  # shape: [4 x T+1]

        return oa, odelta, best_traj, trajectories


    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        # travel = abs(state.v) * self.config.DTK
        travel = 0.08
        dind = travel / self.config.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        #################
        # ind_list = np.mod(ind_list, ncourse)
        #################
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        cyaw[cyaw - state.yaw > 4.5] = (
            cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi)
        )
        cyaw[cyaw - state.yaw < -4.5] = (
            cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi)
        )
        ref_traj[3, :] = cyaw[ind_list]

        return ref_traj

    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    
    def yaw_to_quaternion(self, yaw):
        return Quaternion(
            x=0.0,
            y=0.0,
            z=np.sin(yaw / 2.0),
            w=np.cos(yaw / 2.0)
        )

    def publish_yaw(self,ref_path):
            """Visualize all path components"""
            marker_array = MarkerArray()

            arrow_length = 0.8  # meters

            x = ref_path[0, :]
            y = ref_path[1, :]
            yaw = ref_path[3, :]

            for i in range(len(x)):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "yaw_oriented_arrows"
                marker.id = i
                marker.type = Marker.ARROW
                marker.action = Marker.ADD

                marker.pose.position = Point(x=float(x[i]), y=float(y[i]), z=0.0)
                marker.pose.orientation = self.yaw_to_quaternion(yaw[i])

                # Arrow dimensions
                marker.scale.x = arrow_length     # shaft length
                marker.scale.y = 0.1             # shaft diameter
                marker.scale.z = 0.1             # head diameter

                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
                marker.lifetime.sec = 0  # keep until manually deleted

                marker_array.markers.append(marker)
        
            
            self.marker_pub.publish(marker_array)   

    def publish_path(self):
        # Original waypoints (red)
        marker_array1 = MarkerArray()
        self.add_marker(marker_array1, self.waypoints, 0, ColorRGBA(r=1.0), 0.3)
        # Tracking points (blue)
        self.add_marker(marker_array1, self.tracking_points, 1, ColorRGBA(b=1.0), 0.08)

        self.marker_pub1.publish(marker_array1)

    def add_marker(self, array, points, id, color, scale):
        """Helper to create point markers"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.POINTS
        marker.id = id
        marker.scale.x = marker.scale.y = scale
        marker.color = color
        marker.color.a = 0.3
        
        for pt in points:
            p = Point(x=float(pt[0]), y=float(pt[1]), z=0.0)
            marker.points.append(p)
        array.markers.append(marker)
    
    def publish_mppi_prediction(self, pred_path):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.LINE_STRIP
        marker.scale.x = 0.1
        marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)  # Purple for MPPI
        marker.id = 3
        for i in range(pred_path.shape[1]):
            p = Point()
            p.x, p.y = pred_path[0, i], pred_path[1, i]
            marker.points.append(p)
        self.marker_pub.publish(MarkerArray(markers=[marker]))

    def publish_sampled_trajectories(self, trajectories):
        """
        Visualize all sampled MPPI trajectories in RViz as faint lines.
        Each trajectory is a (T+1, 4) rollout: [x, y, v, yaw]
        """
        marker_array2 = MarkerArray()

        for i, traj in enumerate(trajectories):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.id = 1000 + i  # Avoid ID collision
            marker.scale.x = 0.02
            marker.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.8)  # Grayish transparent
            marker.lifetime.sec = 0  # persistent

            for j in range(traj.shape[0]):
                p = Point()
                p.x, p.y = traj[j, 0], traj[j, 1]
                marker.points.append(p)

            marker_array2.markers.append(marker)

        self.marker_pub2.publish(marker_array2)


def main(args=None):
    rclpy.init(args=args)
    print("MPPI Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()