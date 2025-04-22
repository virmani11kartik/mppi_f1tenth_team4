#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline

@dataclass
class MPPIConfig:
    NXK: int = 4  # state: [x, y, v, yaw]
    NU: int = 2   # control: [accel, delta]
    TK: int = 8   # horizon
    DT: float = 0.1
    K: int = 500
    lambda_: float = 1.0
    noise_sigma: np.ndarray = field(default_factory=lambda: np.diag([0.3, 0.2]))
    max_speed: float = 6.0
    min_speed: float = 0.0
    max_accel: float = 3.0
    max_steer: float = 0.4189
    wb: float = 0.33
    Q: np.ndarray = field(default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0]))
    Qf: np.ndarray = field(default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0]))

@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    yaw: float = 0.0

class MPPIController:
    def __init__(self, config):
        self.config = config
        self.u_mean = np.zeros((self.config.NU, self.config.TK))

    def update_state(self, state, a, delta):
        delta = np.clip(delta, -self.config.max_steer, self.config.max_steer)
        state.x += state.v * math.cos(state.yaw) * self.config.DT
        state.y += state.v * math.sin(state.yaw) * self.config.DT
        state.yaw += (state.v / self.config.wb) * math.tan(delta) * self.config.DT
        state.v += a * self.config.DT
        state.v = np.clip(state.v, self.config.min_speed, self.config.max_speed)
        return state

    def predict_trajectory(self, x0, u):
        traj = np.zeros((self.config.NXK, self.config.TK + 1))
        traj[:, 0] = x0
        state = State(*x0)
        for t in range(self.config.TK):
            state = self.update_state(state, u[0, t], u[1, t])
            traj[:, t + 1] = [state.x, state.y, state.v, state.yaw]
        return traj

    def compute_cost(self, traj, ref):
        cost = 0.0
        for t in range(self.config.TK):
            err = traj[:, t] - ref[:, t]
            cost += err.T @ self.config.Q @ err
        err = traj[:, self.config.TK] - ref[:, self.config.TK]
        cost += err.T @ self.config.Qf @ err
        return cost

    def control(self, x0, ref):
        K, TK = self.config.K, self.config.TK
        noise = np.random.multivariate_normal(np.zeros(self.config.NU), self.config.noise_sigma, size=(K, TK))
        costs = np.zeros(K)
        u_samples = np.zeros((K, self.config.NU, TK))
        for k in range(K):
            u = self.u_mean + noise[k].T
            traj = self.predict_trajectory(x0, u)
            costs[k] = self.compute_cost(traj, ref)
            u_samples[k] = u
        min_cost = np.min(costs)
        weights = np.exp(-(costs - min_cost) / self.config.lambda_)
        weights /= np.sum(weights)
        delta_u = np.sum(weights[:, None, None] * (u_samples - self.u_mean[None, :, :]), axis=0)
        self.u_mean += delta_u
        return self.u_mean[:, 0], self.predict_trajectory(x0, self.u_mean)

class MPPINode(rclpy.node.Node):
    def __init__(self):
        super().__init__('mppi_node')
        self.config = MPPIConfig()
        self.controller = MPPIController(self.config)
        self.subscription = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.marker_pub = self.create_publisher(MarkerArray, "mppi_markers", 10)

        raw_waypoints = np.load("waypoints_modified.npy")
        self.tracking_points = self.fit_and_resample(raw_waypoints)
        self.ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))

    def fit_and_resample(self, raw_points):
        x = raw_points[:, 0]
        y = raw_points[:, 1]
        dx = np.diff(x)
        dy = np.diff(y)
        dist = np.cumsum(np.hypot(dx, dy))
        dist = np.insert(dist, 0, 0)
        spline_x = CubicSpline(dist / dist[-1], x)
        spline_y = CubicSpline(dist / dist[-1], y)
        spline_dist = np.linspace(0, dist[-1], 3000)
        spline_points = np.vstack([spline_x(spline_dist / dist[-1]), spline_y(spline_dist / dist[-1])]).T
        return spline_points
    
    def quaternion_to_yaw(self, q: Quaternion):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y ** 2 + q.z ** 2)
        return math.atan2(siny_cosp, cosy_cosp)


    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        v = np.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        x0 = np.array([x, y, v, yaw])
        self.update_ref_traj(x0)
        u, pred_traj = self.controller.control(x0, self.ref_traj)
        self.publish_drive(u, v)
        self.publish_prediction(pred_traj)

    def update_ref_traj(self, x0):
        ref = self.tracking_points
        dists = np.linalg.norm(ref - np.array([x0[0], x0[1]]), axis=1)
        nearest_idx = np.argmin(dists)
        lookahead = np.linspace(0, 1, self.config.TK + 1)
        inds = (nearest_idx + (lookahead * 20)).astype(int)
        inds = np.clip(inds, 0, len(ref) - 1)
        self.ref_traj[0, :] = ref[inds, 0]
        self.ref_traj[1, :] = ref[inds, 1]
        self.ref_traj[2, :] = np.clip(3.0, self.config.min_speed, self.config.max_speed)
        self.ref_traj[3, :] = 0.0

    def publish_drive(self, u, v):
        msg = AckermannDriveStamped()
        msg.drive.speed = v + u[0] * self.config.DT
        msg.drive.steering_angle = u[1]
        self.publisher.publish(msg)

    def publish_prediction(self, traj):
        self.publish_sampled_trajectories()  # visualize samples
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.LINE_STRIP
        marker.scale.x = 0.1
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        marker.points = [Point(x=traj[0, t], y=traj[1, t], z=0.0) for t in range(traj.shape[1])]
        self.marker_pub.publish(MarkerArray(markers=[marker]))
        
    
    def publish_sampled_trajectories(self):
            marker_array = MarkerArray()
            for k in range(10):  # show 10 sampled rollouts for visualization
                noise = np.random.multivariate_normal(np.zeros(self.config.NU), self.config.noise_sigma, size=(self.config.TK)).T
                u_sample = self.controller.u_mean + noise
                traj = self.controller.predict_trajectory(self.ref_traj[:, 0], u_sample)

                marker = Marker()
                marker.header.frame_id = "map"
                marker.type = Marker.LINE_STRIP
                marker.scale.x = 0.05
                marker.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.4)
                marker.id = k + 100
                marker.points = [Point(x=traj[0, t], y=traj[1, t], z=0.0) for t in range(traj.shape[1])]
                marker_array.markers.append(marker)
            self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    print("MPPI Node started")
    node = MPPINode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
